# structure_mappers.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ===== 按你的项目结构调整这些导入 =====
from ..geometry import Point, Line3D, Polygon3D
from ..materials.beammaterial import ElasticBeam, ElastoplasticBeam, BeamType
from ..materials.pilematerial import ElasticPile, ElastoplasticPile
from ..materials.platematerial import ElasticPlate
from ..materials.anchormaterial import (
    AnchorType, ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor,
)
from ..structures.anchor import Anchor
from ..structures.beam import Beam
from ..structures.embeddedpile import EmbeddedPile
from ..structures.retainingwall import RetainingWall
from ..structures.well import Well
from ..structures.load import (
    PointLoad, LineLoad, SurfaceLoad,
    DynPointLoad, DynLineLoad, DynSurfaceLoad,
    DistributionType,
)
from ..structures.soilblock import SoilBlock

# =============================================================================
# Logging (single-line)
# =============================================================================
def _format_handle(h: Any) -> str:
    if h is None:
        return "None"
    for k in ("Id", "ID", "id", "Guid", "GUID", "guid", "Name", "MaterialName", "Identification"):
        try:
            v = getattr(h, k, None)
            if v is not None:
                if hasattr(v, "value"):  # some bindings wrap values
                    v = v.value
                return f"{k}={v}"
        except Exception:
            continue
    s = str(h).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= 120 else (s[:117] + "...")

def _one_line(msg: str) -> str:
    return " ".join(str(msg).split())

def _log_create(kind: str, desc: str, handle: Any, extra: str = "") -> None:
    h = _format_handle(handle)
    msg = f"[CREATE][{kind}] {desc} handle={h}"
    if extra:
        msg += f" {extra}"
    print(_one_line(msg), flush=True)

def _log_delete(kind: str, desc: str, handle: Any, ok: bool, extra: str = "") -> None:
    h = _format_handle(handle)
    status = "OK" if ok else "FAIL"
    msg = f"[DELETE][{kind}] {desc} handle={h} result={status}"
    if extra:
        msg += f" {extra}"
    print(_one_line(msg), flush=True)

# =============================================================================
# Utilities
# =============================================================================
def _normalize_created_handle(created: Any) -> Any:
    if isinstance(created, (list, tuple)) and created:
        return created[0]
    return created

def _try_call(g_i: Any, names: Sequence[str], *args, **kwargs) -> Any:
    """
    Try calling g_i.<name>(*args, **kwargs) in order; return first success.
    Raises last exception if all candidates failed, or RuntimeError if none found.
    """
    last_exc: Optional[Exception] = None
    for nm in names:
        fn = getattr(g_i, nm, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                continue
    if last_exc:
        raise last_exc
    raise RuntimeError(f"No callable found among {names}")

def _try_delete_with_gi(g_i: Any, plx_obj: Any) -> bool:
    if plx_obj is None:
        return False
    try:
        if hasattr(plx_obj, "delete") and callable(plx_obj.delete):
            plx_obj.delete()
            return True
    except Exception:
        pass
    for fn_name in ("delete", "delobject", "deletematerial", "delmaterial", "remove"):
        try:
            fn = getattr(g_i, fn_name, None)
            if callable(fn):
                fn(plx_obj)
                return True
        except Exception:
            continue
    return False

def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
    """Robust property setting with None-skip + bulk/per-key fallbacks."""
    if not props:
        return
    filtered = {k: v for k, v in props.items() if v is not None}

    # bulk
    if hasattr(plx_obj, "setproperties"):
        try:
            kv: List[Any] = []
            for k, v in filtered.items():
                kv.extend([k, v])
            if kv:
                plx_obj.setproperties(*kv)  # type: ignore[misc]
                return
        except Exception:
            pass

    # per-key
    for k, v in filtered.items():
        try:
            setattr(plx_obj, k, v)
            continue
        except Exception:
            pass
        try:
            if hasattr(plx_obj, "setproperty"):
                plx_obj.setproperty(k, v)
        except Exception:
            pass

def _first_non_none(*vals):
    """Return the first value that is not None; otherwise None."""
    for v in vals:
        if v is not None:
            return v
    return None

# =============================================================================
# Geometry helpers
# =============================================================================
def _ensure_point(g_i: Any, p: Point) -> Any:
    h = getattr(p, "plx_id", None)
    if h is not None:
        return h
    x, y, z = p.get_point()
    try:
        h = g_i.point(x, y, z)
    except TypeError:
        h = g_i.point((x, y, z))
    p.plx_id = h
    return h

def _ensure_surface(g_i: Any, polygon: Polygon3D) -> Any:
    """
    Ensure PLAXIS surface exists; assumes polygon points exist or will be created.
    Also writes back to polygon.plx_id.
    """
    pts = list(polygon.get_points())
    if len(pts) < 3:
        raise ValueError("Surface requires at least three points.")
    plx_pts = [_ensure_point(g_i, p) for p in pts]
    try:
        surf = g_i.surface(*plx_pts)
    except TypeError:
        coords: List[float] = []
        for p in pts:
            x, y, z = p.get_point()
            coords.extend([x, y, z])
        surf = g_i.surface(*coords)
    polygon.plx_id = surf
    return surf

def _assign_material(plx_obj: Any, mat_obj: Any) -> None:
    """
    Assign a (already-created) material handle to a structure.
    Tries a range of common property names.
    """
    if mat_obj is None:
        return
    h = getattr(mat_obj, "plx_id", None)
    if h is None:
        return
    candidates = (
        "Material", "BeamMaterial", "PlateMaterial",
        "N2NMaterial", "AnchorMaterial", "EmbeddedBeamMaterial",
    )
    for k in candidates:
        try:
            setattr(plx_obj, k, h)
            return
        except Exception:
            try:
                if hasattr(plx_obj, "setproperty"):
                    plx_obj.setproperty(k, h)
                    return
            except Exception:
                continue
    _set_many_props(plx_obj, {"Material": h})

# =============================================================================
# Anchor
# =============================================================================
class AnchorMapper:
    @staticmethod
    def create(g_i: Any, obj: Anchor) -> Any:
        pA, pB = obj.get_points()
        hA, hB = _ensure_point(g_i, pA), _ensure_point(g_i, pB)
        created = _try_call(g_i, ("n2nanchor", "n2n_anchor", "anchor", "cable"), hA, hB)
        created = _normalize_created_handle(created)

        _set_many_props(created, {"Name": getattr(obj, "name", None)})

        t = obj.anchor_type
        if isinstance(t, (ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor)):
            _assign_material(created, t)
        elif isinstance(t, AnchorType):
            _set_many_props(created, {"MaterialType": t.value if hasattr(t, "value") else str(t)})

        obj.plx_id = created
        _log_create("Anchor", f"name={obj.name}", created)
        return created

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[Anchor, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, Anchor) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("Anchor", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

# =============================================================================
# Beam
# =============================================================================
class BeamMapper:
    @staticmethod
    def create(g_i: Any, obj: Beam) -> Any:
        pA, pB = obj.get_points()
        hA, hB = _ensure_point(g_i, pA), _ensure_point(g_i, pB)
        created = _try_call(g_i, ("beam", "createbeam", "line_beam"), hA, hB)
        created = _normalize_created_handle(created)

        _set_many_props(created, {"Name": getattr(obj, "name", None)})

        t = obj.beam_type
        if isinstance(t, (ElasticBeam, ElastoplasticBeam)):
            _assign_material(created, t)
        elif isinstance(t, BeamType):
            _set_many_props(created, {"MaterialType": t.value if hasattr(t, "value") else str(t)})

        obj.plx_id = created
        _log_create("Beam", f"name={obj.name}", created)
        return created

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[Beam, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, Beam) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("Beam", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

# =============================================================================
# Embedded Pile
# =============================================================================
class EmbeddedPileMapper:
    @staticmethod
    def create(g_i: Any, obj: EmbeddedPile) -> Any:
        pA, pB = obj.get_points()
        hA, hB = _ensure_point(g_i, pA), _ensure_point(g_i, pB)
        created = _try_call(g_i, ("embeddedbeam", "embedded_pile", "embeddedpile"), hA, hB)
        created = _normalize_created_handle(created)

        _set_many_props(created, {"Name": getattr(obj, "name", None)})

        t = obj.pile_type
        if isinstance(t, (ElasticPile, ElastoplasticPile)):
            _assign_material(created, t)
        else:
            _set_many_props(created, {"MaterialType": getattr(t, "value", t)})

        obj.plx_id = created
        _log_create("EmbeddedPile", f"name={obj.name}", created)
        return created

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[EmbeddedPile, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, EmbeddedPile) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("EmbeddedPile", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

# =============================================================================
# Retaining Wall (Plate)
# =============================================================================
class RetainingWallMapper:
    @staticmethod
    def create(g_i: Any, obj: RetainingWall) -> Any:
        surf_h = getattr(obj.surface, "plx_id", None) or _ensure_surface(g_i, obj.surface)
        created = _try_call(g_i, ("plate", "createplate"), surf_h)
        created = _normalize_created_handle(created)

        _set_many_props(created, {"Name": getattr(obj, "name", None)})

        t = obj.plate_type
        if isinstance(t, ElasticPlate):
            _assign_material(created, t)
        else:
            _set_many_props(created, {"MaterialType": getattr(t, "value", t)})

        obj.plx_id = created
        _log_create("RetainingWall", f"name={obj.name}", created)
        return created

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[RetainingWall, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, RetainingWall) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("RetainingWall", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

# =============================================================================
# Well
# =============================================================================
class WellMapper:
    @staticmethod
    def create(g_i: Any, obj: Well) -> Any:
        pA, pB = obj.get_points()
        hA, hB = _ensure_point(g_i, pA), _ensure_point(g_i, pB)
        created = _try_call(g_i, ("well", "flowwell", "gw_well", "well3d", "create_well"), hA, hB)
        created = _normalize_created_handle(created)

        _set_many_props(created, {"Name": getattr(obj, "name", None)})

        typ = getattr(getattr(obj, "well_type", None), "value", getattr(obj, "well_type", None))
        _set_many_props(created, {
            "WellType": typ,
            "Type": typ,
            "Mode": typ,
            "Hmin": getattr(obj, "h_min", None),
            "h_min": getattr(obj, "h_min", None),
            "MinHead": getattr(obj, "h_min", None),
            "MinHydraulicHead": getattr(obj, "h_min", None),
        })

        obj.plx_id = created
        _log_create("Well", f"name={obj.name} type={typ}", created)
        return created

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[Well, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, Well) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("Well", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok
# =============================================================================
# SoilBlock (best-effort volume creation)
# =============================================================================
class SoilBlockMapper:
    """Best-effort soil volume creation; bindings differ a lot across versions."""
    @staticmethod
    def _extrude_surface(g_i: Any, surf_h: Any, vec: Tuple[float, float, float]) -> Any:
        """
        Try common extrude entrypoints and signatures:
          extrude(surface, dx, dy, dz)  or  extrude(surface, (dx,dy,dz))
        """
        names = ("extrude", "extrusion", "extrude_surface", "extrudesurface")
        last_exc: Optional[Exception] = None
        for nm in names:
            fn = getattr(g_i, nm, None)
            if not callable(fn):
                continue
            # 1) surface, dx,dy,dz
            try:
                obj = fn(surf_h, vec[0], vec[1], vec[2])
                return _normalize_created_handle(obj)
            except Exception as e:
                last_exc = e
            # 2) surface, (dx,dy,dz)
            try:
                obj = fn(surf_h, tuple(vec))
                return _normalize_created_handle(obj)
            except Exception as e:
                last_exc = e
        if last_exc:
            raise last_exc
        raise RuntimeError("No extrude-like constructor found on g_i.")

    @staticmethod
    def create(g_i: Any, obj: SoilBlock) -> Any:
        geom = obj.geometry
        if geom is None:
            raise ValueError("SoilBlock geometry is None.")

        # ---------- A) Polygon3D -> Surface -> Extrude to volume ----------
        if isinstance(geom, Polygon3D):
            # 1) ensure surface exists
            surf_h = _first_non_none(getattr(geom, "plx_id", None), None)
            if surf_h is None:
                surf_h = _ensure_surface(g_i, geom)

            # 2) decide extrude vector
            vec = getattr(obj, "extrude_vec", None)
            if vec is not None:
                try:
                    dx, dy, dz = float(vec[0]), float(vec[1]), float(vec[2])
                    extrude_vec = (dx, dy, dz)
                except Exception:
                    raise TypeError("SoilBlock.extrude_vec must be a 3-tuple of numbers.")
            else:
                # fallback thickness: prefer height/thickness/depth, else default -5.0m in Z
                H = _first_non_none(
                    getattr(obj, "height", None),
                    getattr(obj, "thickness", None),
                    getattr(obj, "depth", None),
                    -5.0,
                )
                try:
                    H = float(H)
                except Exception:
                    H = -5.0
                # 默认向下挤出
                extrude_vec = (0.0, 0.0, -abs(H))

            # 3) extrude and assign material
            created = SoilBlockMapper._extrude_surface(g_i, surf_h, extrude_vec)
            _assign_material(created, getattr(obj, "material", None))
            setattr(obj, "plx_id", created)
            _log_create("SoilBlock", f"name={obj.name} extrude={extrude_vec}", created)
            return created

        # ---------- B) Already-closed volume (rare) -> try soil(...) ----------
        # 如果你的几何类型本身就是封闭体（例如自定义 Polyhedron 并能返回各包围面），
        # 你可以把它转换为一组 surface 句柄后传入 soil(...)。此处保留原来的 best-effort 调用。
        try:
            created = _try_call(g_i, ("soil", "create_soil", "volume", "solid"), geom)
            created = _normalize_created_handle(created)
            _assign_material(created, getattr(obj, "material", None))
            setattr(obj, "plx_id", created)
            _log_create("SoilBlock", f"name={obj.name} via=soil()", created)
            return created
        except Exception as e:
            raise RuntimeError(
                "SoilBlock creation failed. For Polygon3D please use extrusion; "
                "for closed volumes ensure you pass a set of bounding surfaces compatible with soil()."
            ) from e

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[SoilBlock, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, SoilBlock) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            setattr(obj, "plx_id", None)
        _log_delete("SoilBlock", f"name={getattr(obj,'name','raw')}", h, ok=ok)
        return ok