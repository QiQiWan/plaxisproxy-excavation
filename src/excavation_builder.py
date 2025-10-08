# -*- coding: utf-8 -*-
"""
excavation_Builder.py

Refactored ExcavationBuilder:
- Manages geometry, materials, and structures with explicit relationships.
- Best-effort use of existing project classes (RetainingWall, SoilBlock, ElasticPlate, Polygon3D, etc.).
- No new structure/component/material/geometry *types* are introduced; we only reorganize how relationships are tracked.
- Modular build steps with clear docstrings and inline comments.

Notes
-----
- This builder assumes the PLAXIS scripting interface object `g_i` is passed in.
- Where exact PLAXIS collection accessors differ (e.g., how to list Surfaces/Volumes),
  adapt the `_snapshot_*` helpers to your environment.
- Dimensions (domain, pit size, depth) are parameters so you can adjust them easily.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---- Try to use existing project data structures; if absent, fall back gracefully ----
Polygon3D = None
Point3D = None
RetainingWall = None
SoilBlock = None
ElasticPlate = None
SoilMaterial = None

try:
    # Adjust these import paths to your project layout if needed
    from ..components.geometry import Polygon3D, Point3D   # type: ignore
except Exception:
    pass

try:
    from ..components.structures import RetainingWall, SoilBlock  # type: ignore
except Exception:
    pass

try:
    from ..components.materials import ElasticPlate, SoilMaterial  # type: ignore
except Exception:
    pass


@dataclass
class ExcavationBuilder:
    """
    Build an excavation (foundation pit) and manage explicit relationships:
      - self.geometry:  geometric objects (points/lines/surfaces/volumes)
      - self.materials: material objects (plate, soil, etc.)
      - self.structures: conceptual structures (e.g., retaining_walls, soil_blocks),
                         each linking geometry with the applied material

    The builder does NOT introduce new object types. It *uses* your existing model classes
    if available (RetainingWall, SoilBlock, ElasticPlate, Polygon3D), otherwise stores
    simple dicts with references to PLAXIS handles and names.
    """
    g_i: Any
    domain_size: Tuple[float, float] = (40.0, 40.0)  # (Lx, Ly) overall domain
    pit_size: Tuple[float, float] = (20.0, 20.0)     # (Lx, Ly) pit footprint
    pit_depth: float = 10.0                          # excavation depth (positive)
    origin_xy: Tuple[float, float] = (0.0, 0.0)      # center of domain/pit on ground (z=0)

    # State containers
    geometry: Dict[str, Any] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)
    structures: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------- Public API -------------------------------

    def build_excavation_model(self) -> Dict[str, Any]:
        """
        High-level orchestration to build the excavation and record relationships.
        Returns a compact summary (PLAXIS handles / objects) useful for downstream steps.
        """
        self.define_model_boundary()
        self.create_soil_layers()          # keep minimal; adapt if you have stratigraphy inputs
        self.create_excavation_polygon()
        self.extrude_excavation()
        self.assign_materials()

        # Compact return for quick access/testing; full detail retained in self.*
        return {
            "excavation_volume": self.geometry.get("excavation_volume"),
            "bottom_surface": self.geometry.get("excavation_bottom_surface"),
            "retaining_wall_surfaces": (
                self._get_retaining_wall_surfaces()
            ),
            "materials": self.materials,
            "structures": self.structures,
        }

    # --------------------------- Build Steps (modular) ------------------------

    def define_model_boundary(self) -> None:
        """
        Define/resize model domain (soil contour) centered at origin on z=0.
        Adapt this to your project's standard way of setting boundaries.
        """
        Lx, Ly = self.domain_size
        cx, cy = self.origin_xy

        x1, x2 = cx - Lx / 2.0, cx + Lx / 2.0
        y1, y2 = cy - Ly / 2.0, cy + Ly / 2.0

        # Example: use SoilContour.Coordinates (PLAXIS 3D)
        # Ensure your environment supports this; otherwise replace with your own method.
        try:
            self.g_i.SoilContour.reset()
            self.g_i.SoilContour.Coordinates = (
                x1, y1, x2, y1, x2, y2, x1, y2
            )
        except Exception:
            # Fallback: store for later use
            self.geometry["soil_contour"] = (x1, y1, x2, y1, x2, y2, x1, y2)

        self.geometry["domain_bbox"] = (x1, y1, x2, y2)

    def create_soil_layers(self) -> None:
        """
        (Optional) Create a borehole and layers.
        This is intentionally minimal; plug in your actual stratigraphy routine.
        """
        cx, cy = self.origin_xy
        try:
            bh = self.g_i.borehole(cx, cy)
            self.geometry["borehole_point"] = bh
            # Example: add a layer at -20m (adapt to real inputs)
            # self.g_i.soillayer(-20)
        except Exception:
            # If borehole creation not desired/available, skip gracefully
            self.geometry["borehole_point"] = None

    def create_excavation_polygon(self) -> None:
        """
        Draw a rectangular pit footprint (polygon) centered at origin on ground (z=0).
        Stores polygon handle and (if available) a Polygon3D instance.
        """
        Lx, Ly = self.pit_size
        cx, cy = self.origin_xy

        x1, x2 = cx - Lx / 2.0, cx + Lx / 2.0
        y1, y2 = cy - Ly / 2.0, cy + Ly / 2.0
        z = 0.0

        # PLAXIS polyline/polygon creation on ground; adjust to your preferred call
        try:
            p1 = self.g_i.point(x1, y1, z)
            p2 = self.g_i.point(x2, y1, z)
            p3 = self.g_i.point(x2, y2, z)
            p4 = self.g_i.point(x1, y2, z)
            poly = self.g_i.polygon(p1, p2, p3, p4)
        except Exception:
            p1 = (x1, y1, z)
            p2 = (x2, y1, z)
            p3 = (x2, y2, z)
            p4 = (x1, y2, z)
            poly = (p1, p2, p3, p4)

        self.geometry["excavation_polygon"] = poly

        # If Polygon3D exists, capture a semantic geometry object (no new type introduced)
        if Polygon3D is not None and Point3D is not None:
            try:
                poly3d = Polygon3D(
                    points=[
                        Point3D(*p1[:3]), Point3D(*p2[:3]),
                        Point3D(*p3[:3]), Point3D(*p4[:3])
                    ],
                    name="PitFootprint"
                )
                # Record back-reference to PLAXIS handle if desired
                setattr(poly3d, "plx_id", poly)
                self.geometry["excavation_polygon_obj"] = poly3d
            except Exception:
                pass

    def extrude_excavation(self) -> None:
        """
        Extrude the pit polygon downward by pit_depth to create:
          - a new soil Volume
          - a bottom horizontal Surface
          - vertical side Surfaces (retaining walls)
        Explicitly records which surfaces are walls and which is bottom.
        """
        poly = self.geometry.get("excavation_polygon")
        if poly is None:
            raise RuntimeError("Excavation polygon not created.")

        # Snapshot scene before extrusion (to detect newly created faces/volumes)
        pre_surfs = self._snapshot_surfaces()
        pre_vols = self._snapshot_volumes()

        depth = float(self.pit_depth)
        # Extrude downward along -Z; use your project's exact API
        try:
            # Example: line extrude (vector dz)
            new_vol = self.g_i.extrude(poly, 0.0, 0.0, -depth)
        except Exception:
            new_vol = {"volume": "EXCAVATION_VOLUME_PLACEHOLDER"}

        # Snapshot after
        post_surfs = self._snapshot_surfaces()
        post_vols = self._snapshot_volumes()

        # Identify newly created entities
        added_surfs = post_surfs - pre_surfs
        added_vols = post_vols - pre_vols

        # Heuristic separation: the bottom is the (near-)horizontal face at z ~ -depth
        bottom_surf = self._find_bottom_surface(added_surfs, target_z=-depth)

        wall_surfs = set(added_surfs)
        if bottom_surf is not None and bottom_surf in wall_surfs:
            wall_surfs.remove(bottom_surf)

        # Persist geometry references
        self.geometry["excavation_volume"] = self._first_or_none(added_vols) or new_vol
        self.geometry["excavation_bottom_surface"] = bottom_surf
        self.structures["retaining_walls"] = {
            "surfaces": list(wall_surfs)  # PLAXIS surface handles
        }

        # If your structure/material classes exist, optionally wrap them (no *new* types)
        # Create RetainingWall objects per surface (deferred material binding to assign_materials)
        if RetainingWall is not None and Polygon3D is not None and Point3D is not None:
            rw_objs: List[Any] = []
            for idx, s in enumerate(wall_surfs, start=1):
                # We don't reconstruct exact 3D polygon from PLAXIS here; keep a placeholder Polygon3D
                try:
                    poly3d = Polygon3D(points=[], name=f"WallSurface{idx}")
                    setattr(poly3d, "plx_id", s)
                    rw = RetainingWall(name=f"RetainingWall{idx}", surface=poly3d, plate_type=None)  # material later
                    setattr(rw, "plx_surface", s)
                    rw_objs.append(rw)
                except Exception:
                    # If construction fails, skip wrapping; surfaces remain in dict above
                    pass
            if rw_objs:
                # Mirror surfaces list with object list for richer semantics
                self.structures["retaining_walls_objects"] = rw_objs

    def assign_materials(self) -> None:
        """
        Create/retrieve materials and bind them to geometry:
          - Assign 'Wall' plate material to all retaining wall surfaces.
        Uses existing material classes if available; otherwise stores PLAXIS handles.
        """
        # Ensure a "Wall" plate material exists; capture in self.materials
        wall_mat = self._ensure_wall_plate_material()

        # Assign to all wall surfaces
        wall_surfaces = self._get_retaining_wall_surfaces()
        for s in wall_surfaces:
            try:
                # PLAXIS: setmaterial(surface, material)
                self.g_i.setmaterial(s, wall_mat)
            except Exception:
                # If the API differs, adapt here.
                pass

        # If RetainingWall objects were created, bind the material there too
        if self.structures.get("retaining_walls_objects"):
            for rw in self.structures["retaining_walls_objects"]:
                try:
                    if ElasticPlate is not None and isinstance(wall_mat, ElasticPlate):
                        rw.plate_type = wall_mat
                    else:
                        # Store PLAXIS handle (e.g., g_i.Wall) if ElasticPlate not available
                        setattr(rw, "plx_material", wall_mat)
                except Exception:
                    pass

    # ------------------------------ Helper Methods ----------------------------

    def _snapshot_surfaces(self) -> set:
        """
        Return a set of PLAXIS surface handles (or IDs) currently present.
        Adapt attribute access to your environment.
        """
        try:
            # Common PLAXIS way: e.g., self.g_i.Soil.Surfaces
            surfs = set(self.g_i.Soil.Surfaces[:])
            return surfs
        except Exception:
            # Fallback: empty set; caller should handle
            return set()

    def _snapshot_volumes(self) -> set:
        """
        Return a set of PLAXIS volume handles (or IDs) currently present.
        Adapt attribute access to your environment.
        """
        try:
            vols = set(self.g_i.Soil.Volumes[:])
            return vols
        except Exception:
            return set()

    def _find_bottom_surface(self, candidate_surfs: set, target_z: float, tol: float = 1e-2) -> Optional[Any]:
        """
        Heuristic: find the bottom surface among the newly added ones.
        If you can query a representative Z (elevation) of a surface in your API, use it here.
        Otherwise, return None and keep all candidates as wall surfaces.
        """
        for s in candidate_surfs:
            try:
                # Example pseudo-API; replace with your own (e.g., s.Polygons[0].Points -> z)
                zrep = self._approx_surface_z(s)
                if zrep is not None and abs(zrep - target_z) <= tol:
                    return s
            except Exception:
                continue
        return None

    def _approx_surface_z(self, surface: Any) -> Optional[float]:
        """
        Attempt to query a representative z of a surface (centroid or first point).
        Replace with your project-specific implementation.
        """
        try:
            # Example pseudo access:
            # pts = surface.Polygon.Points  # -> list of (x,y,z)
            # return sum(p.z for p in pts)/len(pts)
            return None
        except Exception:
            return None

    def _first_or_none(self, items: Sequence[Any] | set) -> Optional[Any]:
        if not items:
            return None
        if isinstance(items, set):
            for it in items:
                return it
        return items[0]

    def _get_retaining_wall_surfaces(self) -> List[Any]:
        """
        Return the PLAXIS surfaces representing retaining walls (vertical sides).
        """
        entry = self.structures.get("retaining_walls")
        if entry and isinstance(entry, dict):
            surfs = entry.get("surfaces", [])
            if isinstance(surfs, list):
                return surfs
        return []

    def _ensure_wall_plate_material(self) -> Any:
        """
        Ensure there is a 'Wall' plate material; store and return it.
        Uses ElasticPlate if available; otherwise returns the PLAXIS handle.
        """
        # If already created, return
        if "Wall" in self.materials:
            return self.materials["Wall"]

        # Try to create/reuse via PLAXIS; adapt to your material creation flow.
        plx_handle = None
        try:
            # If exists (e.g., identified as g_i.Wall), reuse; otherwise create
            plx_handle = getattr(self.g_i, "Wall", None)
            if plx_handle is None:
                # Example: create a plate material with identification "Wall"
                self.g_i.platemat("Identification", "Wall")
                plx_handle = getattr(self.g_i, "Wall", None)
        except Exception:
            pass

        # If your project has ElasticPlate class, wrap it for richer semantics
        if ElasticPlate is not None:
            try:
                ep = ElasticPlate(name="Wall")
                # Bind PLAXIS back-ref if helpful for mappers
                setattr(ep, "plx_id", plx_handle if plx_handle is not None else "Wall")
                self.materials["Wall"] = ep
                return ep
            except Exception:
                pass

        # Fallback: store the PLAXIS handle or the identification name
        self.materials["Wall"] = plx_handle if plx_handle is not None else "Wall"
        return self.materials["Wall"]
