# -*- coding: utf-8 -*-
"""
Geology (borehole-defined) → OBJ/MTL surfaces + STL solids + glTF exporter (meters).

- Entry now accepts a BoreholeSet object.
- Bounds can be a class 'Bounds' or a 4-tuple; if None, auto-derive from boreholes (+/- pad).
- Interpolation method is an Enum: InterpMethod.LINEAR / IDW / KRIGING.
- When the material name/color is not specified, the soil layer name is used and
  the color is pseudo-randomly generated but stable per layer name.
- New: export to glTF 2.0 (.gltf + .bin) so most 3D software can import and
  then convert to FBX, GLB, etc.

Author: Qiwei Wan
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Sequence, Optional, Any, Union, Iterable
import math
import json
import random
import os
import base64

import numpy as np
import matplotlib.tri as mtri

from ..borehole import BoreholeSet, BoreholeLayer, Borehole

# Optional Kriging
try:
    # Ordinary Kriging from PyKrige (optional dependency)
    from pykrige.ok import OrdinaryKriging  # type: ignore[import]
    _HAS_PYKRIGE: bool = True
except Exception:
    # Fallback: keep the name defined so static checkers do not complain
    OrdinaryKriging = None  # type: ignore[assignment]
    _HAS_PYKRIGE = False


# ###########################################################################
# Public types
# ###########################################################################

@dataclass(frozen=True)
class Bounds:
    """Axis-aligned rectangle in XY (meters)."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax)


class InterpMethod(Enum):
    """Interpolation choices (indexable)."""
    LINEAR  = "linear"   # TIN piecewise planes (straight-line connection)
    IDW     = "idw"      # inverse distance weighting
    KRIGING = "kriging"  # ordinary kriging (requires PyKrige)


# ###########################################################################
# Core helpers: boreholes & layers
# ###########################################################################

def _to_borehole_list(
    bhset: Union[BoreholeSet, Sequence[Borehole], Iterable[Borehole]]
) -> List[Borehole]:
    """
    Normalize input to a concrete list of Borehole objects.

    - If BoreholeSet, use its 'boreholes' attribute (if present) else iterate.
    - If already a sequence/iterable of Borehole, copy to list.
    """
    # BoreholeSet with 'boreholes' attribute (preferred)
    if isinstance(bhset, BoreholeSet) and hasattr(bhset, "boreholes"):
        return list(getattr(bhset, "boreholes") or [])

    # Any object with 'boreholes' attribute
    if hasattr(bhset, "boreholes"):
        return list(getattr(bhset, "boreholes") or [])

    # Sequence/Iterable fallback
    if isinstance(bhset, (list, tuple)):
        return list(bhset)

    try:
        return list(bhset)  # type: ignore[arg-type]
    except Exception as exc:
        raise TypeError(
            "Unsupported borehole container; expected BoreholeSet or iterable of Borehole."
        ) from exc


def _abs_elev(ground_level: float, rel_depth: float) -> float:
    """Absolute elevation = ground level + relative depth (downward is negative)."""
    return float(ground_level + rel_depth)


def _layer_key(lyr: "BoreholeLayer") -> str:
    """Prefer soil_layer.name; else fallback to layer.name."""
    try:
        sl = getattr(lyr, "soil_layer", None)
        nm = getattr(sl, "name", None)
        if nm:
            return str(nm)
    except Exception:
        pass

    nm2 = getattr(lyr, "name", None)
    return str(nm2) if nm2 else "Layer"


def _collect_layer_points(
    boreholes: Sequence["Borehole"]
) -> Dict[str, Dict[str, List[Tuple[float, float, float]]]]:
    """
    Collect, per soil layer, the absolute elevations of top / bottom surfaces
    at each borehole XY location.

    Supports two BoreholeLayer conventions:

    1) New style (absolute elevations):
       - lyr.top_z, lyr.bottom_z are absolute Z (m).

    2) Old style (relative depths from ground level):
       - lyr.top, lyr.bottom are depths (usually <= 0, downward negative),
         combined with Borehole.ground_level via _abs_elev().
    """
    out: Dict[str, Dict[str, List[Tuple[float, float, float]]]] = {}

    for bh in boreholes:
        x = float(bh.location.x)
        y = float(bh.location.y)
        gl = float(getattr(bh, "ground_level", 0.0))

        for lyr in getattr(bh, "layers", []) or []:
            key = _layer_key(lyr)

            if hasattr(lyr, "top_z") and hasattr(lyr, "bottom_z"):
                # New API: absolute elevations already stored on the layer
                zt = float(lyr.top_z)
                zb = float(lyr.bottom_z)
            else:
                # Fallback: relative depths from ground level
                top_rel = float(getattr(lyr, "top", 0.0))
                bot_rel = float(getattr(lyr, "bottom", 0.0))
                zt = _abs_elev(gl, top_rel)
                zb = _abs_elev(gl, bot_rel)

            rec = out.setdefault(key, {"top": [], "bot": []})
            rec["top"].append((x, y, zt))
            rec["bot"].append((x, y, zb))

    return out


# ###########################################################################
# Bounds & grid helpers
# ###########################################################################

def _compute_bounds_from_boreholes(
    boreholes: Sequence["Borehole"], pad: float
) -> Bounds:
    """Auto bounds from borehole XY, then expand by 'pad' meters in both +/- directions."""
    xs = [float(bh.location.x) for bh in boreholes]
    ys = [float(bh.location.y) for bh in boreholes]

    if not xs or not ys:
        # Fallback small box if input empty
        return Bounds(-10.0 - pad, 10.0 + pad, -10.0 - pad, 10.0 + pad)

    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad

    # Ensure non-degenerate
    if xmax <= xmin:
        xmax = xmin + 2 * pad
    if ymax <= ymin:
        ymax = ymin + 2 * pad

    return Bounds(xmin, xmax, ymin, ymax)


def _normalize_bounds(
    boundary: Optional[Union[Bounds, Tuple[float, float, float, float]]],
    boreholes: Sequence["Borehole"],
    pad: float
) -> Bounds:
    """Accept Bounds, 4-tuple, or None. If None, derive from boreholes and expand by pad."""
    if boundary is None:
        return _compute_bounds_from_boreholes(boreholes, pad=pad)

    if isinstance(boundary, Bounds):
        return boundary

    if (isinstance(boundary, tuple) or isinstance(boundary, list)) and len(boundary) == 4:
        xmin, xmax, ymin, ymax = boundary
        return Bounds(float(xmin), float(xmax), float(ymin), float(ymax))

    raise TypeError("boundary must be Bounds, a 4-tuple (xmin,xmax,ymin,ymax), or None")


def _grid_xy(
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    step: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Regular XY grid over rectangle (meters)."""
    nx = max(2, int(math.ceil((xmax - xmin) / step)) + 1)
    ny = max(2, int(math.ceil((ymax - ymin) / step)) + 1)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY


def _convex_hull_mask(points_xy: np.ndarray, XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """Mask grid nodes outside convex hull of control points."""
    tri = mtri.Triangulation(points_xy[:, 0], points_xy[:, 1])
    f = mtri.LinearTriInterpolator(tri, np.ones(points_xy.shape[0]))
    Z = f(XX, YY)
    return Z.mask  # True = outside


# ###########################################################################
# IDs & colors
# ###########################################################################

def _sanitize_id(name: str) -> str:
    """Safe ASCII-ish id for OBJ/MTL names."""
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("_")
    return out if out else "LAYER"


def _random_color(seed: Optional[str] = None) -> Tuple[float, float, float]:
    """
    Pseudo-random diffuse color in [0, 1].
    If a seed (e.g. layer name) is given, the color is stable for that seed.
    """
    rng = random.Random(seed) if seed is not None else random
    # Avoid too dark colors: clamp to [0.25, 0.95]
    return tuple(0.25 + 0.7 * rng.random() for _ in range(3))  # type: ignore[return-value]


# ###########################################################################
# Path helper
# ###########################################################################

def _ensure_parent_dir(path: Optional[str]) -> None:
    """
    Ensure the parent directory of 'path' exists if 'path' contains a directory.
    If path is None or has no directory component, do nothing.
    """
    if not path:
        return
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    print(f"The model will be written as {os.path.abspath(path)}")


# ###########################################################################
# Interpolators
# ###########################################################################

def _interp_linear(
    xyz: List[Tuple[float, float, float]],
    XX: np.ndarray,
    YY: np.ndarray
) -> np.ma.MaskedArray:
    arr = np.asarray(xyz, float)
    tri = mtri.Triangulation(arr[:, 0], arr[:, 1])
    f = mtri.LinearTriInterpolator(tri, arr[:, 2])
    return f(XX, YY)


def _interp_idw(
    xyz: List[Tuple[float, float, float]],
    XX: np.ndarray,
    YY: np.ndarray,
    power: float = 2.0,
    eps: float = 1e-9,
    k: Optional[int] = None,
    mask_outside_by_hull: bool = True
) -> np.ma.MaskedArray:
    pts = np.asarray(xyz, float)  # (N,3)
    px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

    GX = XX.ravel()
    GY = YY.ravel()

    dx = GX[:, None] - px[None, :]
    dy = GY[:, None] - py[None, :]
    dist = np.sqrt(dx * dx + dy * dy) + eps

    if k is not None and k < pts.shape[0]:
        idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(dist.shape[0])[:, None]
        dist = dist[rows, idx]
        zsel = pz[idx]
    else:
        zsel = np.broadcast_to(pz[None, :], dist.shape)

    w = 1.0 / np.power(dist, power)
    z = np.sum(w * zsel, axis=1) / np.sum(w, axis=1)
    Z = z.reshape(XX.shape)

    if mask_outside_by_hull and pts.shape[0] >= 3:
        hull_mask = _convex_hull_mask(pts[:, :2], XX, YY)
        return np.ma.array(Z, mask=hull_mask)

    return np.ma.array(Z, mask=np.zeros_like(Z, dtype=bool))


def _interp_kriging(
    xyz: List[Tuple[float, float, float]],
    XX: np.ndarray,
    YY: np.ndarray,
    variogram_model: str = "spherical",
    enable_fallback: bool = True
) -> np.ma.MaskedArray:
    pts = np.asarray(xyz, float)

    # If PyKrige is not available, optionally fall back to IDW
    if not _HAS_PYKRIGE or OrdinaryKriging is None:  # type: ignore[truthy-function]
        if enable_fallback:
            return _interp_idw(
                xyz, XX, YY,
                power=2.0,
                k=None,
                mask_outside_by_hull=True,
            )
        raise RuntimeError(
            "PyKrige is not available. Install via 'pip install pykrige' "
            "or enable_fallback=True."
        )

    OK = OrdinaryKriging(
        pts[:, 0], pts[:, 1], pts[:, 2],
        variogram_model=variogram_model,
        enable_plotting=False,
        verbose=False,
    )
    z, _ = OK.execute("grid", XX[0, :], YY[:, 0])  # (ny, nx)
    Z = np.array(z, float)
    hull_mask = _convex_hull_mask(pts[:, :2], XX, YY)
    return np.ma.array(Z, mask=hull_mask)


def _dispatch_interp(
    method: InterpMethod,
    xyz: List[Tuple[float, float, float]],
    XX: np.ndarray,
    YY: np.ndarray,
    method_params: Optional[Dict[str, Any]],
) -> np.ma.MaskedArray:
    mp = method_params or {}
    if method == InterpMethod.LINEAR:
        return _interp_linear(xyz, XX, YY)
    if method == InterpMethod.IDW:
        return _interp_idw(
            xyz, XX, YY,
            power=mp.get("power", 2.0),
            eps=mp.get("eps", 1e-9),
            k=mp.get("k", None),
            mask_outside_by_hull=mp.get("mask_outside_by_hull", True),
        )
    if method == InterpMethod.KRIGING:
        return _interp_kriging(
            xyz, XX, YY,
            variogram_model=mp.get("variogram_model", "spherical"),
            enable_fallback=mp.get("enable_fallback", True),
        )
    raise ValueError(f"Unknown interpolation method: {method}")


# ###########################################################################
# Meshing utils: shells from grids
# ###########################################################################

def _grid_to_tris(XX: np.ndarray, YY: np.ndarray, ZZ: np.ndarray) -> np.ndarray:
    """Convert a height grid to a surface triangle mesh (top or bottom only)."""
    ny, nx = ZZ.shape
    faces: List[np.ndarray] = []

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            z00 = ZZ[iy, ix]
            z10 = ZZ[iy, ix + 1]
            z01 = ZZ[iy + 1, ix]
            z11 = ZZ[iy + 1, ix + 1]

            if any(np.ma.is_masked(v) for v in (z00, z10, z01, z11)):
                continue

            x00, y00 = XX[iy, ix],             YY[iy, ix]
            x10, y10 = XX[iy, ix + 1],         YY[iy, ix + 1]
            x01, y01 = XX[iy + 1, ix],         YY[iy + 1, ix]
            x11, y11 = XX[iy + 1, ix + 1],     YY[iy + 1, ix + 1]

            faces.append(np.array([[x00, y00, z00],
                                   [x10, y10, z10],
                                   [x01, y01, z01]], float))
            faces.append(np.array([[x01, y01, z01],
                                   [x10, y10, z10],
                                   [x11, y11, z11]], float))

    return np.array(faces, float) if faces else np.zeros((0, 3, 3), float)


def _grid_pair_to_solid(
    XX: np.ndarray,
    YY: np.ndarray,
    Ztop: np.ndarray,
    Zbot: np.ndarray
) -> np.ndarray:
    """
    Build a closed shell (top + bottom + sides) for one layer
    from top/bottom elevation grids.
    """
    ny, nx = Ztop.shape
    faces: List[np.ndarray] = []

    def add(a, b, c) -> None:
        faces.append(np.array([a, b, c], float))

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            zt00, zt10, zt01, zt11 = (
                Ztop[iy, ix], Ztop[iy, ix + 1],
                Ztop[iy + 1, ix], Ztop[iy + 1, ix + 1]
            )
            zb00, zb10, zb01, zb11 = (
                Zbot[iy, ix], Zbot[iy, ix + 1],
                Zbot[iy + 1, ix], Zbot[iy + 1, ix + 1]
            )

            if any(np.ma.is_masked(v) for v in
                   (zt00, zt10, zt01, zt11, zb00, zb10, zb01, zb11)):
                continue

            x00, y00 = XX[iy, ix],             YY[iy, ix]
            x10, y10 = XX[iy, ix + 1],         YY[iy, ix + 1]
            x01, y01 = XX[iy + 1, ix],         YY[iy + 1, ix]
            x11, y11 = XX[iy + 1, ix + 1],     YY[iy + 1, ix + 1]

            # top (2)
            add([x00, y00, zt00], [x10, y10, zt10], [x01, y01, zt01])
            add([x01, y01, zt01], [x10, y10, zt10], [x11, y11, zt11])
            # bottom (2) reversed
            add([x01, y01, zb01], [x10, y10, zb10], [x00, y00, zb00])
            add([x11, y11, zb11], [x10, y10, zb10], [x01, y01, zb01])
            # sides (8)
            add([x00, y00, zt00], [x01, y01, zt01], [x01, y01, zb01])
            add([x00, y00, zt00], [x01, y01, zb01], [x00, y00, zb00])

            add([x10, y10, zt10], [x11, y11, zt11], [x11, y11, zb11])
            add([x10, y10, zt10], [x11, y11, zb11], [x10, y10, zb10])

            add([x00, y00, zt00], [x10, y10, zt10], [x10, y10, zb10])
            add([x00, y00, zt00], [x10, y10, zb10], [x00, y00, zb00])

            add([x01, y01, zt01], [x11, y11, zt11], [x11, y11, zb11])
            add([x01, y01, zt01], [x11, y11, zb11], [x01, y01, zb01])

    return np.array(faces, float) if faces else np.zeros((0, 3, 3), float)


# ###########################################################################
# Writers: OBJ / MTL / STL
# ###########################################################################

def _write_mtl(mtl_path: str, materials: Dict[str, Dict[str, Any]]) -> None:
    _ensure_parent_dir(mtl_path)
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("# MTL generated for geology layers\n")
        for mid, props in materials.items():
            kd = props.get("Kd", (0.8, 0.8, 0.8))
            d = float(props.get("d", 1.0))
            f.write(f"newmtl {mid}\n")
            f.write(f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n")
            f.write(f"d {d:.4f}\n\n")


def _write_obj_with_mtl(
    obj_path: str,
    mtl_name: str,
    objects: Dict[str, np.ndarray],
    obj_to_mat: Dict[str, str]
) -> None:
    _ensure_parent_dir(obj_path)
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# OBJ geology export\n")
        f.write(f"mtllib {mtl_name}\n")

        vid = 1
        for oname, tris in objects.items():
            tris = np.asarray(tris, float)
            if tris.size == 0:
                continue

            mat = obj_to_mat.get(oname, "default")
            f.write(f"o {oname}\nusemtl {mat}\n")

            for tri in tris:
                for v in tri:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            n_tris = tris.shape[0]
            for i in range(n_tris):
                a = vid + i * 3 + 0
                b = vid + i * 3 + 1
                c = vid + i * 3 + 2
                f.write(f"f {a} {b} {c}\n")

            vid += n_tris * 3


def _write_stl_ascii(path: str, solids: Dict[str, np.ndarray]) -> None:
    _ensure_parent_dir(path)

    def nrm(tri: np.ndarray) -> np.ndarray:
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        n = np.cross(v1, v2)
        L = np.linalg.norm(n)
        return n / L if L > 1e-12 else np.array([0.0, 0.0, 1.0])

    with open(path, "w", encoding="utf-8") as f:
        f.write("solid geology\n")
        for name, tris in solids.items():
            tris = np.asarray(tris, float)
            if tris.size == 0:
                continue
            f.write(f"  # solid {name}\n")
            for tri in tris:
                n = nrm(tri)
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {tri[0, 0]:.6e} {tri[0, 1]:.6e} {tri[0, 2]:.6e}\n")
                f.write(f"      vertex {tri[1, 0]:.6e} {tri[1, 1]:.6e} {tri[1, 2]:.6e}\n")
                f.write(f"      vertex {tri[2, 0]:.6e} {tri[2, 1]:.6e} {tri[2, 2]:.6e}\n")
                f.write("    endloop\n  endfacet\n")
        f.write("endsolid geology\n")


# ###########################################################################
# Writer: glTF 2.0 (.gltf + .bin)
# ###########################################################################

def _write_gltf(
    gltf_path: str,
    objects: Dict[str, np.ndarray],
    obj_to_mat: Dict[str, str],
    mat_catalog: Dict[str, Dict[str, Any]]
) -> None:
    """
    Minimal glTF 2.0 exporter:
    - Single .gltf file, with vertex/index binary embedded as base64 data URI
      (no external .bin file).
    - One buffer containing all vertices + indices (per object).
    - Each object is a separate mesh + node.
    - Materials created from mat_catalog (baseColorFactor from Kd, alpha from d).
    """
    if not objects:
        return

    # 1) Materials
    materials = []
    mat_index_by_id: Dict[str, int] = {}
    for mat_id, props in mat_catalog.items():
        kd = props.get("Kd", (0.8, 0.8, 0.8))
        d = float(props.get("d", 1.0))
        rgba = [float(kd[0]), float(kd[1]), float(kd[2]), d]
        mat_index = len(materials)
        mat_index_by_id[mat_id] = mat_index
        materials.append({
            "name": str(mat_id),
            "pbrMetallicRoughness": {
                "baseColorFactor": rgba,
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        })

    # 2) Geometry → single buffer with many bufferViews/accessors
    buffer_views = []
    accessors = []
    meshes = []
    nodes = []

    bin_data = bytearray()

    def _align4() -> None:
        nonlocal bin_data
        pad = (-len(bin_data)) % 4
        if pad:
            bin_data.extend(b"\x00" * pad)

    for oname, tris in objects.items():
        tris = np.asarray(tris, dtype=np.float32)
        if tris.size == 0:
            continue

        verts = tris.reshape(-1, 3)      # (n_v, 3)
        n_verts = int(verts.shape[0])
        indices = np.arange(n_verts, dtype=np.uint32)  # local 0..n_v-1
        n_indices = int(indices.size)

        # Positions bufferView
        _align4()
        pos_offset = len(bin_data)
        bin_data.extend(verts.tobytes())
        pos_len = len(bin_data) - pos_offset

        pos_view_index = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": pos_offset,
            "byteLength": pos_len,
            "target": 34962,  # ARRAY_BUFFER
        })

        min_xyz = verts.min(axis=0).tolist()
        max_xyz = verts.max(axis=0).tolist()

        accessor_pos_index = len(accessors)
        accessors.append({
            "bufferView": pos_view_index,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": n_verts,
            "type": "VEC3",
            "min": min_xyz,
            "max": max_xyz,
        })

        # Indices bufferView
        _align4()
        idx_offset = len(bin_data)
        bin_data.extend(indices.tobytes())
        idx_len = len(bin_data) - idx_offset

        idx_view_index = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": idx_offset,
            "byteLength": idx_len,
            "target": 34963,  # ELEMENT_ARRAY_BUFFER
        })

        accessor_idx_index = len(accessors)
        accessors.append({
            "bufferView": idx_view_index,
            "byteOffset": 0,
            "componentType": 5125,  # UNSIGNED_INT
            "count": n_indices,
            "type": "SCALAR",
            "min": [0],
            "max": [n_verts - 1],
        })

        # Material index
        mat_id = obj_to_mat.get(oname, None)
        if mat_id:
            mat_index = mat_index_by_id.get(mat_id, 0)
        else:
            raise ValueError("mat_id is not defined.")

        # Mesh + node
        mesh_index = len(meshes)
        meshes.append({
            "name": str(oname),
            "primitives": [
                {
                    "attributes": {"POSITION": accessor_pos_index},
                    "indices": accessor_idx_index,
                    "material": mat_index,
                }
            ],
        })

        nodes.append({
            "name": str(oname),
            "mesh": mesh_index,
        })

    # 3) Single embedded buffer (base64 data URI)
    bin_bytes = bytes(bin_data)
    b64 = base64.b64encode(bin_bytes).decode("ascii")
    buffer = {
        "byteLength": len(bin_bytes),
        "uri": "data:application/octet-stream;base64," + b64,
    }

    scene = {"nodes": list(range(len(nodes)))}

    gltf = {
        "asset": {"version": "2.0", "generator": "borehole2model"},
        "scene": 0,
        "scenes": [scene],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "buffers": [buffer],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    _ensure_parent_dir(gltf_path)
    with open(gltf_path, "w", encoding="utf-8") as fj:
        json.dump(gltf, fj, ensure_ascii=False, indent=2)

# ###########################################################################
# Material registry & meshing orchestration
# ###########################################################################

def _build_material_registry(
    per_layer: Dict[str, Dict[str, List[Tuple[float, float, float]]]],
    materials: Optional[Dict[str, Dict[str, Any]]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """
    Build a mapping:
      - mat_catalog: material_id -> {Kd, d}
      - layer_to_mat: layer_name  -> material_id
    """
    mat_catalog: Dict[str, Dict[str, Any]] = {}
    layer_to_mat: Dict[str, str] = {}

    for layer_name in per_layer.keys():
        spec = (materials or {}).get(layer_name, {})

        # Material name: explicitly given or just use layer_name
        mat_id_src = spec.get("mat_id", layer_name)
        mat_id = _sanitize_id(mat_id_src)

        # Diffuse color: explicitly given or pseudo-random from layer_name
        if "Kd" in spec:
            Kd = spec["Kd"]
        else:
            Kd = _random_color(seed=layer_name)

        # Alpha
        d = float(spec.get("d", 1.0))

        mat_catalog[mat_id] = {"Kd": Kd, "d": d}
        layer_to_mat[layer_name] = mat_id

    return mat_catalog, layer_to_mat


def _build_layer_meshes(
    per_layer: Dict[str, Dict[str, List[Tuple[float, float, float]]]],
    XX: np.ndarray,
    YY: np.ndarray,
    layer_to_mat: Dict[str, str],
    method: InterpMethod,
    method_params: Optional[Dict[str, Any]],
    invert_fix_eps: float
) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Dict[str, np.ndarray]]:
    """
    For each layer:
      - Interpolate top / bottom surfaces.
      - Enforce bottom below top.
      - Build a closed shell (top+bottom+sides) as triangles.

    Returns:
      obj_objects: name -> triangles
      obj_to_mat : name -> material_id
      stl_solids : name -> triangles
    """
    obj_objects: Dict[str, np.ndarray] = {}
    obj_to_mat: Dict[str, str] = {}
    stl_solids: Dict[str, np.ndarray] = {}

    for layer_name, rec in per_layer.items():
        tpts = rec["top"]
        bpts = rec["bot"]

        # Need enough constraints for interpolation
        if len(tpts) < 3 or len(bpts) < 3:
            continue

        # Interpolate top and bottom
        Ztop = _dispatch_interp(method, tpts, XX, YY, method_params)
        Zbot = _dispatch_interp(method, bpts, XX, YY, method_params)

        # Enforce bottom below top (absolute elevation)
        both = (~Ztop.mask) & (~Zbot.mask)
        if not np.any(both):
            continue

        Zbot_fixed = Zbot.copy()
        Zbot_fixed[both] = np.minimum(Zbot[both], Ztop[both] - invert_fix_eps)

        # Build closed shell
        solid_tris = _grid_pair_to_solid(XX, YY, Ztop, Zbot_fixed)
        if solid_tris.size == 0:
            continue

        solid_name = _sanitize_id(layer_name)

        obj_objects[solid_name] = solid_tris
        obj_to_mat[solid_name] = layer_to_mat[layer_name]
        stl_solids[solid_name] = solid_tris

    return obj_objects, obj_to_mat, stl_solids


# ###########################################################################
# High-level API
# ###########################################################################

def export_geology(
    borehole_set: BoreholeSet,
    boundary: Optional[Union[Bounds, Tuple[float, float, float, float]]] = None,
    *,
    boundary_pad: float = 10.0,     # meters if boundary is None
    grid_step: float = 10.0,        # meters
    method: InterpMethod = InterpMethod.LINEAR,
    method_params: Optional[Dict[str, Any]] = None,
    invert_fix_eps: float = 0.05,   # meters (gap to avoid top/bottom inversion)
    obj_path: Optional[str] = "model/geology.obj",
    mtl_path: Optional[str] = "model/geology.mtl",
    stl_path: Optional[str] = "model/geology.stl",
    gltf_path: Optional[str] = "model/geology.gltf",
    materials: Optional[Dict[str, Dict[str, Any]]] = None,
    legend_json: Optional[str] = "geology_legend.json"
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Bounds]:
    """
    Export OBJ/MTL surfaces, STL solids, and glTF 2.0 for each geological layer.

    - 'borehole_set' is your BoreholeSet container.
    - 'boundary' can be Bounds, (xmin,xmax,ymin,ymax), or None
      (auto from boreholes + boundary_pad).
    - 'method' is InterpMethod enum.

    Returns:
      obj_objects: name -> triangles (closed shells)
      stl_solids : name -> triangles (same shells)
      used_bounds: Bounds actually used
    """
    # 1) Normalize boreholes
    boreholes: List[Borehole] = _to_borehole_list(borehole_set)

    # 2) Bounds
    used_bounds = _normalize_bounds(boundary, boreholes, pad=boundary_pad)
    xmin, xmax, ymin, ymax = used_bounds.as_tuple()

    # 3) Per-layer top/bottom constraints at borehole XY
    per_layer = _collect_layer_points(boreholes)
    if not per_layer:
        return {}, {}, used_bounds

    # 4) Grid
    XX, YY = _grid_xy(xmin, xmax, ymin, ymax, grid_step)

    # 5) Materials
    mat_catalog, layer_to_mat = _build_material_registry(per_layer, materials)

    # 6) Mesh generation (closed shells)
    obj_objects, obj_to_mat, stl_solids = _build_layer_meshes(
        per_layer, XX, YY, layer_to_mat,
        method=method,
        method_params=method_params,
        invert_fix_eps=invert_fix_eps,
    )

    # 7) File outputs
    if mtl_path:
        _write_mtl(mtl_path, mat_catalog)

    if obj_path:
        _write_obj_with_mtl(
            obj_path,
            (os.path.basename(mtl_path) if mtl_path else "materials.mtl"),
            obj_objects,
            obj_to_mat,
        )

    if stl_path:
        _write_stl_ascii(stl_path, stl_solids)

    if gltf_path:
        _write_gltf(gltf_path, obj_objects, obj_to_mat, mat_catalog)

    # 8) Legend (optional)
    if legend_json:
        _ensure_parent_dir(legend_json)
        legend = {
            "units": "meter",
            "interpolation": method.value,
            "bounds": {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
            },
            "layers": [
                {
                    "layer": ln,
                    "material": layer_to_mat[ln],
                    "color_Kd": mat_catalog[layer_to_mat[ln]]["Kd"],
                }
                for ln in per_layer.keys()
                if ln in layer_to_mat
            ],
        }
        with open(legend_json, "w", encoding="utf-8") as f:
            json.dump(legend, f, ensure_ascii=False, indent=2)

    return obj_objects, stl_solids, used_bounds


# ###########################################################################
# Backward-compatible alias
# ###########################################################################

def export_geology_obj_stl(
    borehole_set: BoreholeSet,
    xmin: float = -345,
    xmax: float = 345,
    ymin: float = -60,
    ymax: float = 60,
    *,
    boundary_pad: float = 10.0,     # meters if boundary is None
    grid_step: float = 10.0,        # meters
    method: InterpMethod = InterpMethod.LINEAR,
    method_params: Optional[Dict[str, Any]] = None,
    invert_fix_eps: float = 0.05,   # meters (gap to avoid top/bottom inversion)
    obj_path: Optional[str] = "models/geology.obj",
    mtl_path: Optional[str] = "models/geology.mtl",
    stl_path: Optional[str] = "models/geology.stl",
    gltf_path: Optional[str] = "models/geology.gltf",
    materials: Optional[Dict[str, Dict[str, Any]]] = None,
    legend_json: Optional[str] = "geology_legend.json"
):
    """
    Compatibility wrapper if you previously passed explicit xmin/xmax/ymin/ymax.
    """
    boundary: Optional[Union[Bounds, Tuple[float, float, float, float]]]
    if None not in (xmin, xmax, ymin, ymax):
        boundary = (float(xmin), float(xmax), float(ymin), float(ymax))
    else:
        boundary = None

    return export_geology(
        borehole_set,
        boundary=boundary,
        boundary_pad=boundary_pad,
        grid_step=grid_step,
        method=method,
        method_params=method_params,
        invert_fix_eps=invert_fix_eps,
        obj_path=obj_path,
        mtl_path=mtl_path,
        stl_path=stl_path,
        gltf_path=gltf_path,
        materials=materials,
        legend_json=legend_json,
    )


# ###########################################################################
# Examples (for manual testing)
# ###########################################################################

if __name__ == "__main__":
    # Example only; fill BoreholeSet with real boreholes before calling.
    bhset = BoreholeSet()

    # 1) No explicit bounds → auto pad by 10 m around boreholes
    obj_objs, stl_solids, used_bounds = export_geology(
        bhset,
        boundary=None,
        grid_step=10.0,
        method=InterpMethod.LINEAR,
    )

    # 2) Bounds as a 4-tuple
    obj_objs, stl_solids, used_bounds = export_geology(
        bhset,
        boundary=(-345.0, 345.0, -50.0, 50.0),
        grid_step=8.0,
        method=InterpMethod.IDW,
        method_params={"power": 2.0, "k": 12, "mask_outside_by_hull": True},
    )

    # 3) Bounds as a class; try kriging (falls back to IDW if PyKrige missing)
    box = Bounds(xmin=-345.0, xmax=345.0, ymin=-50.0, ymax=50.0)
    obj_objs, stl_solids, used_bounds = export_geology(
        bhset,
        boundary=box,
        grid_step=10.0,
        method=InterpMethod.KRIGING,
        method_params={"variogram_model": "spherical"},
        materials={
            "121_细砂": {"mat_id": "mat_121_sand", "Kd": (0.90, 0.80, 0.45)},
            "62_淤泥质粉质黏土": {"mat_id": "mat_62_mud", "Kd": (0.55, 0.45, 0.38)},
        },
        obj_path="geo/geo.obj",
        mtl_path="geo/geo.mtl",
        stl_path="geo/geo.stl",
        gltf_path="geo/geo.gltf",
        legend_json="geo/geo_legend.json",
    )
    print("Used bounds:", used_bounds)
