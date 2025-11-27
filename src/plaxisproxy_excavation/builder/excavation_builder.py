from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from math import ceil
from typing import Iterable, List, Dict, Any, Optional, Tuple, Union, Sequence
import numpy as np
from ..utils import NeighborPointPicker


from ..plaxishelper.plaxisrunner import PlaxisRunner
from ..excavation import FoundationPit, StructureType, _normalize_structure_type
from ..structures.basestructure import BaseStructure
from ..structures.soilblock import SoilBlock
from ..geometry import Polygon3D
from ..components import Mesh, Phase

from ..plaxishelper.plaxisoutput import PlaxisOutput
from ..plaxishelper.resulttypes import (
    Plate as PlateResult,
    Beam as BeamResult,
    EmbeddedBeam as EmbeddedBeamResult,
    NodeToNodeAnchor as NodeToNodeAnchorResult,
    Well as WellResult,
    Soil as SoilResult,
)

"""
Excavation Engineering Automation — Builder

This builder orchestrates mapping the FoundationPit model into PLAXIS
via a PlaxisRunner adapter. It builds only the *initial design*; phase
options/activations and per-phase water/well overrides are applied later.
"""

def _segments_from_wall_surfaces(walls, tol=1e-6):
    """
    Derive 2D axis-aligned segments from each wall surface by reading its outer ring.
    For a rectangular vertical wall:
    - If x is (nearly) constant and y varies: vertical segment (x, y1)–(x, y2).
    - If y is (nearly) constant and x varies: horizontal segment (x1, y)–(x2, y).
    Returns: [((x1,y1),(x2,y2)), ...] with snapping to 'tol'.
    """
    segs = []

    def snap(v): return round(float(v) / tol) * tol

    for w in (walls or []):
        surf = getattr(w, "surface", None)
        if surf is None:
            continue
        try:
            ring = surf.outer_ring               # Line3D
            pts = ring.get_points()              # [Point]
        except Exception:
            continue

        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        x_lo, x_hi = min(xs), max(xs)
        y_lo, y_hi = min(ys), max(ys)

        # Detect orientation by span size
        span_x = abs(x_hi - x_lo)
        span_y = abs(y_hi - y_lo)

        # Heuristics:
        # - rect_wall_x: x is constant (span_x ~ 0), y varies
        # - rect_wall_y: y is constant (span_y ~ 0), x varies
        if span_x <= tol and span_y > tol:
            x = snap((x_lo + x_hi) * 0.5)
            segs.append(((x, snap(y_lo)), (x, snap(y_hi))))
        elif span_y <= tol and span_x > tol:
            y = snap((y_lo + y_hi) * 0.5)
            segs.append(((snap(x_lo), y), (snap(x_hi), y)))
        else:
            # Non-rectangular or too small; ignore
            continue

    # Optional: merge collinear intervals along same x or y to reduce redundancy
    return _merge_collinear_xy(segs, tol=tol)


def _merge_collinear_xy(segs, tol=1e-6):
    """Merge collinear, touching axis-aligned segments: tiny, clear 1D union."""
    from collections import defaultdict

    def snap(v): return round(v / tol) * tol

    vertical = defaultdict(list)   # x -> [(y1,y2)]
    horizontal = defaultdict(list) # y -> [(x1,x2)]

    for (x1, y1), (x2, y2) in segs:
        x1, y1, x2, y2 = snap(x1), snap(y1), snap(x2), snap(y2)
        if abs(x1 - x2) <= tol:
            y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
            vertical[x1].append((y1, y2))
        elif abs(y1 - y2) <= tol:
            x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
            horizontal[y1].append((x1, x2))

    def merge_1d(intervals):
        if not intervals: return []
        intervals.sort()
        out = [list(intervals[0])]
        for a, b in intervals[1:]:
            if a <= out[-1][1] + tol:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [tuple(p) for p in out]

    merged = []
    for x, ys in vertical.items():
        for y1, y2 in merge_1d(ys):
            merged.append(((x, y1), (x, y2)))
    for y, xs in horizontal.items():
        for x1, x2 in merge_1d(xs):
            merged.append(((x1, y), (x2, y)))
    return merged


def _signed_area_xy(verts_xy):
    """Shoelace area on (x,y) verts; >0 means CCW."""
    if len(verts_xy) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(verts_xy, verts_xy[1:] + verts_xy[:1]):
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _log(self, msg):
    """Mirror logs to both print and a builder `msg` list if present."""
    try:
        print(msg)
    finally:
        if hasattr(self, "msg") and isinstance(self.msg, list):
            self.msg.append(str(msg))



class ExcavationBuilder:
    """
    Build the excavation model in PLAXIS and collect IDs/handles via PlaxisRunner.

    Responsibilities:
    - Validate the FoundationPit payload shape (non-breaking warnings where possible).
    - Map project info, materials, boreholes & layers, structures, loads, monitors.
    - Optionally mesh.
    - Create phase shells (do not apply settings/activations here).
    - Provide helpers to apply phases and update well parameters later.
    """

    def __init__(self, excavation_object: FoundationPit, PASSWORD: str, HOST="localhost", PORT=10000) -> None:
        # Always create our own runner instance using config; keep provided `app` for signature parity.
        self.PORT = PORT 
        self.PASSWORD = PASSWORD
        self.HOST = HOST
        self.App: PlaxisRunner = PlaxisRunner(PASSWORD, HOST, PORT)
        self.excavation_object: FoundationPit = excavation_object
        self.Output: Optional[PlaxisOutput] = None
        self._calc_done: bool = False  # True after a successful builder.calculate()

        # Caches for soil coordinates and result fields (per phase)
        # phase_name -> NeighborPointPicker (KD-tree on soil X/Y/Z)
        self._soil_coord_picker_cache: Dict[str, NeighborPointPicker] = {}
        # phase_name -> coords (N,3) array of (x,y,z)
        self._soil_coord_cache: Dict[str, np.ndarray] = {}
        # (phase_name, leaf_name, smoothing) -> values (N,) array
        self._soil_field_cache: Dict[Tuple[str, str, bool], np.ndarray] = {}

    @classmethod
    def create(cls, excavation_object: FoundationPit, PASSWORD: str, HOST="localhost", PORT=10000):
        return cls(excavation_object, PASSWORD, HOST, PORT)
    
    @staticmethod
    def create_input_client(PASSWORD: str, HOST="localhost", PORT=10000) -> PlaxisRunner:
        return PlaxisRunner( PASSWORD, HOST, PORT)
    
    @staticmethod
    def create_output_client(PASSWORD: str, HOST="localhost", PORT=10001) -> PlaxisOutput:
        return PlaxisOutput(PASSWORD, HOST, PORT) 

    # #########################################################################
    # Lifecycle
    # #########################################################################

    def initialize(self) -> None:
        """Initialize builder: validate pit object and ensure a fresh PLAXIS project."""
        self.check_completeness()
        self.App.connect().new()

    def calculate(self) -> None:
        """Go to the stages interface, and start a calculation task."""
        self.App.calculate()
        # If no exception was raised, mark as calculated
        self._calc_done = True

    # #########################################################################
    # Validation (aligned with current FoundationPit definition)
    # #########################################################################

    def check_completeness(self) -> None:
        """
        Validate the FoundationPit instance against the current data model.

        Hard fails:
          - excavation_object missing
          - project_information missing

        Soft warnings (non-blocking):
          - name missing (fallback to project_information.title if available)
          - empty/invalid materials/structures/loads dicts (will be initialized)
          - no boreholes (builder may create a minimal soil body in mapper layer)
          - no phases (can be applied later via `apply_phases()`)

        Notes:
          - `footprint` / `depth` are NOT part of FoundationPit → no checks here.
          - Water table and well overrides live in Phase objects → not required here.
        """
        pit = getattr(self, "excavation_object", None)
        if pit is None:
            raise TypeError("excavation_object is not set.")

        errors: list[str] = []
        warns: list[str] = []

        # #### Project information (required) ####
        proj = getattr(pit, "project_information", None)
        if proj is None:
            errors.append("Missing project information (pit.project_information).")
        else:
            # Light sanity (do not hard-fail on units/bounds here)
            try:
                _ = float(getattr(proj, "gamma_water"))
            except Exception:
                warns.append("ProjectInformation.gamma_water not numeric; mapper defaults may be used.")
            # Bounding box access is optional; warn if unavailable
            for attr in ("x_min", "x_max", "y_min", "y_max"):
                if not hasattr(proj, attr):
                    warns.append(f"ProjectInformation.{attr} not found; mapper may use defaults.")
                    break

        # #### Name (optional): prefer explicit pit.name; fallback to project title ####
        pit_name = getattr(pit, "name", None)
        if not pit_name or not str(pit_name).strip():
            fallback = getattr(proj, "title", None) if proj else None
            if fallback:
                try:
                    setattr(pit, "name", str(fallback))
                    warns.append("pit.name not provided; using project_information.title as name.")
                except Exception:
                    warns.append("pit.name not provided; continuing without explicit name.")
            else:
                warns.append("pit.name not provided; continuing without explicit name.")

        # #### Boreholes (optional) ####
        bhset = getattr(pit, "borehole_set", None)
        bh_count = 0
        if bhset is None:
            warns.append("No borehole_set; a minimal soil body may be created by the builder.")
        else:
            try:
                bh_count = len(getattr(bhset, "boreholes", []) or [])
                if bh_count == 0:
                    warns.append("No boreholes defined; a minimal soil body may be created by the builder.")
            except Exception:
                warns.append("borehole_set present but not iterable; ignoring.")

        # #### Materials (init empty buckets if needed) ####
        mats = getattr(pit, "materials", None)
        if not isinstance(mats, dict):
            warns.append("materials is not a dict; initializing empty material categories.")
            pit.materials = {
                "soil_materials": [],
                "plate_materials": [],
                "anchor_materials": [],
                "beam_materials": [],
                "pile_materials": [],
            }
        else:
            for k in ("soil_materials", "plate_materials", "anchor_materials", "beam_materials", "pile_materials"):
                if k not in mats:
                    mats[k] = []

        # #### Structures (init empty buckets if needed) ####
        st = getattr(pit, "structures", None)
        if not isinstance(st, dict):
            warns.append("structures is not a dict; initializing empty structure categories.")
            pit.structures = {
                "retaining_walls": [],
                "anchors": [],
                "beams": [],
                "wells": [],
                "embedded_piles": [],
            }
        else:
            for k in ("retaining_walls", "anchors", "beams", "wells", "embedded_piles", "soil_blocks"):
                if k not in st:
                    st[k] = []

        # #### Loads (init empty buckets if needed) ####
        loads = getattr(pit, "loads", None)
        if not isinstance(loads, dict):
            warns.append("loads is not a dict; initializing empty load categories.")
            pit.loads = {"point_loads": [], "line_loads": [], "surface_loads": []}
        else:
            for k in ("point_loads", "line_loads", "surface_loads"):
                if k not in loads:
                    loads[k] = []

        try:
            excava_depth = float(getattr(pit, "excava_depth", 0.0))
        except Exception:
            excava_depth = 0.0

        walls_list = []
        try:
            # 按对象结构获取当前已定义的围护墙集合
            walls_list = pit.structures.get(StructureType.RETAINING_WALLS.value, []) if isinstance(pit.structures, dict) else []
        except Exception:
            walls_list = []

        # default: do not create bottom surface
        self._bottom_surface_ok = False

        if excava_depth != 0.0 and walls_list and self._walls_form_closed_ring(walls_list):
            z_shortest = self._shortest_wall_bottom_z(walls_list)
            if excava_depth > z_shortest:
                self._bottom_surface_ok = True
            else:
                errors.append(f"[check] Bottom surface NOT allowed: excava_depth={excava_depth:.3f} <= shortest wall bottom={z_shortest:.3f}")
        else:
            if excava_depth == 0.0:
                errors.append("[check] excava_depth not set (0.0); bottom surface will not be created.")
            else:
                errors.append("[check] Walls do not form a closed ring; bottom surface will not be created.")

        # #### Phases (optional at build-time; required only for apply_phases) ####
        phases = getattr(pit, "phases", None)
        if phases is None:
            pit.phases = []
            warns.append("No phases defined; call builder.apply_phases() later to stage the model.")
        elif not isinstance(phases, (list, tuple)):
            warns.append("phases is not a list; converting to empty list for safety.")
            pit.phases = []

        # #### Finalize ####
        if errors:
            msg = ["FoundationPit completeness check failed:"]
            msg += [f" - {e}" for e in errors]
            if warns:
                msg.append("Notes (non-blocking):")
                msg += [f" * {w}" for w in warns]
            raise ValueError("\n".join(msg))

        for w in warns:
            print(f"[check_completeness] {w}")

        walls = len(pit.structures.get(StructureType.RETAINING_WALLS.value, [])) if isinstance(pit.structures, dict) else 0
        print(
            f"[check_completeness] OK: "
            f"name='{getattr(pit, 'name', '<unnamed>')}', "
            f"boreholes={bh_count}, walls={walls}, phases={len(pit.phases)}."
        )

    # ################################ Excavation volumn check ################################

    @staticmethod
    def _walls_form_closed_ring(walls, tol: float = 1e-6) -> bool:
        """
        Minimal closure check in XY:
        - need at least 3 walls;
        - extents in both X and Y must be > tol.
        """
        if not walls or len(walls) < 3:
            return False
        xs, ys = [], []
        for w in walls:
            poly = getattr(w, "surface", None)
            pts = poly.get_points() if (poly and hasattr(poly, "get_points")) else []
            for p in pts:
                xs.append(float(getattr(p, "x", 0.0)))
                ys.append(float(getattr(p, "y", 0.0)))
        if not xs or not ys:
            return False
        return (max(xs) - min(xs) > tol) and (max(ys) - min(ys) > tol)

    @staticmethod
    def _shortest_wall_bottom_z(walls) -> float:
        """
        'Shortest' wall = the one with the HIGHEST bottom z among walls.
        For each wall, bottom z = min z of its surface points.
        Return max of those bottoms (or -inf if none).
        """
        bottoms = []
        for w in walls or []:
            poly = getattr(w, "surface", None)
            pts = poly.get_points() if (poly and hasattr(poly, "get_points")) else []
            if not pts:
                continue
            zmin = min(float(getattr(p, "z", 0.0)) for p in pts)
            bottoms.append(zmin)
        return max(bottoms) if bottoms else float("-inf")

    @staticmethod
    def _collect_wall_xy_points(walls):
        """
        Collect all surface vertices of retaining walls (projected to XY).
        """
        pts2d = []
        for w in walls or []:
            poly = getattr(w, "surface", None)
            pts = poly.get_points() if (poly and hasattr(poly, "get_points")) else []
            for p in pts:
                pts2d.append((float(p.x), float(p.y)))
        return pts2d

    @staticmethod
    def _convex_hull_xy(points):
        """
        Monotone chain convex hull in 2D.
        Returns vertices in counter-clockwise order, without repeating the first.
        """
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        return hull

    # ########## New public helpers: wall footprint / bottom polygon ##########

    def get_wall_footprint_polygon3d(self, z_value: float = 0.0, tol: float = 1e-6):
        """
        Return a Polygon3D at z = z_value representing the 2D footprint enclosed by
        retaining wall surfaces. Returns None if a closed loop cannot be reconstructed.
        """
        pit = getattr(self, "excavation_object", None)
        walls = []
        if pit is not None and isinstance(getattr(pit, "structures", None), dict):
            walls = pit.structures.get(StructureType.RETAINING_WALLS.value, []) or []
        return self._make_bottom_polygon3d(walls, z_value, tol=tol)

    def get_wall_footprint_xy(self, z_value: float = 0.0, tol: float = 1e-6):
        """
        Return the footprint as a list of (x, y) tuples in order (closed ring, last!=first).
        Returns [] if not available.
        """
        poly3d = self.get_wall_footprint_polygon3d(z_value=z_value, tol=tol)
        if not poly3d:
            return []
        try:
            pts = poly3d.outer_ring.get_points()  # [Point]
            xy = [(float(p.x), float(p.y)) for p in pts]
            # 去掉首尾重复
            if len(xy) >= 2 and xy[0] == xy[-1]:
                xy = xy[:-1]
            return xy
        except Exception:
            return []


    def _make_bottom_polygon3d(self, walls, z_value, tol=1e-6):
        """
        Build the excavation bottom polygon at z = z_value directly from wall surfaces.
        Assumptions:
        - Each wall surface is a rectangular vertical plane made by rect_wall_x / rect_wall_y.
        - The outline is a single closed rectilinear loop.

        Steps:
        1) For each wall, read surface.outer_ring points.
        2) Detect if the wall is x-const or y-const; create a 2D segment at z=z_value.
        3) Check graph degrees (each vertex must have degree 2).
        4) Trace the unique loop and lift to 3D.

        Returns:
        Polygon3D or None if outline cannot be reconstructed.
        """
        from src.plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D

        segs = _segments_from_wall_surfaces(walls, tol=tol)
        if not segs:
            _log(self, "[bottom] No wall segments derived from surfaces.")
            return None

        # Build adjacency (snapped) and validate degrees
        def snap(v): return round(float(v) / tol) * tol
        def kxy(x, y): return (snap(x), snap(y))

        adj = {}
        for (x1, y1), (x2, y2) in segs:
            a, b = kxy(x1, y1), kxy(x2, y2)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        if not adj:
            _log(self, "[bottom] Empty adjacency.")
            return None

        # Every node must have degree 2 for a single simple loop
        bad = [(n, len(N)) for n, N in adj.items() if len(N) != 2]
        if bad:
            _log(self, f"[bottom] Open or branched graph: {len(bad)} non-2-degree nodes (e.g. {bad[:3]}).")
            return None

        # Trace the loop deterministically (start at lexicographically smallest node)
        start = min(adj.keys())
        loop = [start]
        prev, curr = None, start

        # Next neighbor helper: avoid immediately going back to prev
        def next_of(u, p):
            nbrs = sorted(adj[u])     # deterministic
            if p is None:
                return nbrs[0]
            return nbrs[1] if len(nbrs) == 2 and nbrs[0] == p else nbrs[0]

        # Walk until we return to start
        for _ in range(len(adj) + len(segs) + 4):
            nxt = next_of(curr, prev)
            loop.append(nxt)
            if nxt == start:
                break
            prev, curr = curr, nxt

        # Validate closure and produce simple (x,y) list without the duplicated end
        if len(loop) < 4 or loop[0] != loop[-1]:
            _log(self, "[bottom] Failed to close the loop.")
            return None

        verts2d = loop[:-1]  # drop duplicate start
        # Ensure CCW orientation for consistent normals
        if _signed_area_xy(verts2d) < 0:
            verts2d.reverse()

        # Build Polygon3D
        pts3d = [Point(x, y, z_value) for (x, y) in verts2d] + [Point(verts2d[0][0], verts2d[0][1], z_value)]
        ring = Line3D(PointSet(pts3d))
        try:
            poly = Polygon3D([ring], require_horizontal=False)
        except Exception as e:
            _log(self, f"[bottom] Polygon3D construction failed: {e}")
            return None

        _log(self, f"[bottom] Bottom loop OK: {len(verts2d)} vertices; area={poly.area() if hasattr(poly,'area') else '?'}")
        return poly



    # #########################################################################
    # Build (initial design only)
    # #########################################################################
    # #### 0) iterate helper ###################################################
    def _iter_dict_lists(self, dct):
        if isinstance(dct, dict):
            for _k, _lst in dct.items():
                for _it in (_lst or []):
                    if _it is not None:
                        yield _k, _it

    # #### 1) Project info ###################################################
    def build_project_info(self) -> None:
        pit = self.excavation_object
        app = self.App
        if getattr(pit, "project_information", None) is not None:
            try:
                app.apply_project_information(pit.project_information)
            except Exception as e:
                print(f"[build] Warning: apply_project_information failed: {e}")

    # #### 2) Materials ######################################################
    def build_materials(self) -> int:
        pit = self.excavation_object
        app = self.App
        total_mats = 0
        try:
            for _k, mat in self._iter_dict_lists(getattr(pit, "materials", {})):
                app.create_material(mat)
                total_mats += 1
        except Exception as e:
            print(f"[build] Warning: create_material failed on some items: {e}")
        return total_mats

    # #### 3) Boreholes & layer relinking ###################################
    def relink_borehole_layers(self) -> None:
        try:
            self._relink_borehole_layers_to_library()
        except Exception as e:
            print(f"[build] Warning: relink borehole layers failed: {e}")

    def build_boreholes(self) -> None:
        pit = self.excavation_object
        app = self.App
        if getattr(pit, "borehole_set", None) is not None:
            try:
                app.create_boreholes(pit.borehole_set)
            except Exception as e:
                print(f"[build] Warning: create_boreholes failed: {e}")

    # #### 4) Soil blocks (pre-existing) ####################################
    def build_initial_soil_blocks(self) -> None:
        pit = self.excavation_object
        app = self.App
        for blk in (getattr(pit, "structures", {}) or {}).get("soil_blocks", []) or []:
            try:
                app.create_soil_block(blk)
            except Exception as e:
                print(f"[build] Warning: create_soil_block failed: {e}")

    # #### 5) Structures broken down ########################################
    def build_walls(self) -> int:
        pit = self.excavation_object
        app = self.App
        cnt = 0
        for wall in (getattr(pit, "structures", {}) or {}).get(StructureType.RETAINING_WALLS.value, []) or []:
            try:
                app.create_retaining_wall(wall)
                cnt += 1
            except Exception as e:
                print(f"[build] Warning: create_retaining_wall failed: {e}")
        return cnt

    def build_bottom_surface(self) -> bool:
        pit = self.excavation_object
        app = self.App
        try:
            ok = bool(getattr(self, "_bottom_surface_ok", False))
        except Exception:
            ok = False
        if not ok:
            return False
        try:
            excava_depth = float(getattr(pit, "excava_depth", 0.0))
            bottom_poly = self.get_wall_footprint_polygon3d(z_value=excava_depth, tol=1e-6)
            if bottom_poly is None:
                print("[build] Warning: could not derive bottom polygon from walls (need >=3 points).")
                return False
            app.create_surface(bottom_poly, name="ExcavationBottom", auto_close=True)
            print(f"[build] Bottom surface created at z={excava_depth:.3f} using wall-enclosed polygon.")
            return True
        except Exception as e:
            print(f"[build] Warning: bottom polygon creation failed: {e}")
            return False

    def build_beams(self) -> int:
        pit = self.excavation_object
        app = self.App
        cnt = 0
        for beam in (getattr(pit, "structures", {}) or {}).get(StructureType.BEAMS.value, []) or []:
            try:
                app.create_beam(beam)
                cnt += 1
            except Exception as e:
                print(f"[build] Warning: create_beam failed: {e}")
        return cnt

    def build_anchors(self) -> int:
        pit = self.excavation_object
        app = self.App
        cnt = 0
        for anc in (getattr(pit, "structures", {}) or {}).get(StructureType.ANCHORS.value, []) or []:
            try:
                app.create_anchor(anc)
                cnt += 1
            except Exception as e:
                print(f"[build] Warning: create_anchor failed: {e}")
        return cnt

    def build_piles(self) -> int:
        pit = self.excavation_object
        app = self.App
        cnt = 0
        for pile in (getattr(pit, "structures", {}) or {}).get(StructureType.EMBEDDED_PILES.value, []) or []:
            try:
                app.create_embedded_pile(pile)
                cnt += 1
            except Exception as e:
                print(f"[build] Warning: create_embedded_pile failed: {e}")
        return cnt

    def build_wells(self) -> int:
        """Create wells at geometry stage. Parameter overrides belong to phases."""
        pit = self.excavation_object
        app = self.App
        cnt = 0
        for well in (getattr(pit, "structures", {}) or {}).get(StructureType.WELLS.value, []) or []:
            try:
                app.create_well(well)
                cnt += 1
            except Exception as e:
                print(f"[build] Warning: create_well failed: {e}")
        return cnt

    def cut_and_register_inside_soil_blocks(self) -> None:
        pit = self.excavation_object
        if not (getattr(pit, "structures", {}) or {}).get(StructureType.SOIL_BLOCKS.value):
            self._cut_inside_soil_blocks(pit)
        for blk in (getattr(pit, "structures", {}) or {}).get(StructureType.SOIL_BLOCKS.value, []):
            self.App.create_soil_block(blk)

    # #### 6) Loads ##########################################################
    def build_loads(self) -> int:
        pit = self.excavation_object
        app = self.App
        loads = getattr(pit, "loads", {}) or {}
        total = 0
        try:
            for _k, ld in self._iter_dict_lists(loads):
                app.create_load(ld)
                total += 1
        except Exception as e:
            print(f"[build] Warning: create_load failed on some items: {e}")
        for mul in getattr(pit, "load_multipliers", []) or []:
            try:
                app.create_load_multiplier(mul)
            except Exception as e:
                print(f"[build] Warning: create_load_multiplier failed: {e}")
        return total

    # #### 7) Monitors #######################################################
    def build_monitors(self) -> int:
        pit = self.excavation_object
        app = self.App
        monitors = getattr(pit, "monitors", []) or []
        if not monitors:
            return 0
        try:
            if getattr(app, "g_i", None) is None:
                raise RuntimeError("Not connected (g_i is None).")
            try:
                from ..plaxishelper.monitormapper import MonitorMapper  # type: ignore
            except Exception:
                from plaxisproxy_excavation.plaxishelper.monitormapper import MonitorMapper  # type: ignore
            MonitorMapper.create_monitors(app.g_i, monitors)  # type: ignore
            return len(monitors)
        except Exception as e:
            print(f"[build] Warning: monitor mapping skipped: {e}")
            return 0

    # #### 8) Mesh ###########################################################
    def build_mesh(self, mesh: bool) -> bool:
        if not mesh:
            return False
        pit = self.excavation_object
        app = self.App
        mesh_cfg = getattr(pit, "mesh", None)
        if mesh_cfg is None:
            mesh_cfg = Mesh()
        else:
            print("[build] Info: no mesh config on FoundationPit; use the default!")
        try:
            app.mesh(mesh_cfg)
            return True
        except Exception as e:
            print(f"[build] Warning: mesh() failed: {e}")
            return False

    # #### 9) Phase shells (no apply) #######################################
    def build_phase_shells(self) -> int:
        pit = self.excavation_object
        app = self.App
        phases = list(getattr(pit, "phases", []) or getattr(pit, "stages", []) or [])
        if not phases:
            return 0
        created = 0
        try:
            app.goto_stages()
            prev = app.get_initial_phase()
            for ph in phases:
                # Special handling for soil blocks with suffix _in
                for blk in pit.structures.get(StructureType.SOIL_BLOCKS.value, []):
                    if blk.name.endswith("_in"):
                        phase_name = blk.name[:-3]
                        for ph2 in phases:
                            if ph2.name == phase_name:
                                ph2.deactivate_structures(blk)
                handle = app.create_phase(ph, inherits=prev)
                created += 1
                prev = handle
        except Exception as e:
            print(f"[build] Warning: phase creation failed: {e}")
        return created

    # =========================== Public build() =============================
    def build(self, mesh: bool = True) -> Dict[str, Any]:
        """Build ONLY the initial design using modular helpers."""
        pit = self.excavation_object
        app = self.App

        # Ensure connection & fresh project
        if not getattr(app, "is_connected", False) or getattr(app, "g_i", None) is None or getattr(app, "input_server", None) is None:
            app.connect().new()

        # Optional: completeness check
        if hasattr(self, "check_completeness") and callable(self.check_completeness):
            self.check_completeness()

        # 1) project
        self.build_project_info()
        # 2) materials
        total_mats = self.build_materials()
        # 3) boreholes
        self.relink_borehole_layers()
        self.build_boreholes()
        # 4) pre-declared soil blocks
        self.build_initial_soil_blocks()
        # 5) structures (walls → bottom → beams/anchors/piles/wells)
        _walls = self.build_walls()
        _bottom_ok = self.build_bottom_surface()
        _beams = self.build_beams()
        _ancs = self.build_anchors()
        _piles = self.build_piles()
        _wells = self.build_wells()
        # optional cut/registration for inside soil blocks
        self.cut_and_register_inside_soil_blocks()
        # 6) loads
        total_loads = self.build_loads()
        # 7) monitors
        total_monitors = self.build_monitors()
        # 8) mesh
        meshed = self.build_mesh(mesh)
        # 9) phase shells only
        phases_created = self.build_phase_shells()

        return {
            "materials": total_mats,
            "structures": {
                StructureType.RETAINING_WALLS.value: _walls,
                StructureType.BEAMS.value: _beams,
                StructureType.ANCHORS.value: _ancs,
                StructureType.EMBEDDED_PILES.value: _piles,
                StructureType.WELLS.value: _wells,
                StructureType.SOIL_BLOCKS.value: len((getattr(pit, "structures", {}) or {}).get(StructureType.SOIL_BLOCKS.value, []) or []),
            },
            "loads": total_loads,
            "monitors": total_monitors,
            "phases_created": phases_created,
            "meshed": meshed,
        }

    # ================= Keep all your original non-build helpers =============
    # (apply_phases, soil block utilities, Output helpers, save/load, results, etc.)
    # Paste them below unchanged from your current file.


    # #########################################################################
    # Soil-material relinking helpers (borehole layers → library)
    # #########################################################################

    def _debug_dump_bh_material_links(self) -> None:
        """Print a compact report of SoilLayer.material links for debugging."""
        pit = self.excavation_object
        print("\n[DEBUG] Borehole → Layer → SoilLayer.material")
        bhset = getattr(pit, "borehole_set", None)
        if not bhset or not getattr(bhset, "boreholes", None):
            print("  (no boreholes)")
            return

        for bi, bh in enumerate(bhset.boreholes):
            layers = getattr(bh, "layers", []) or []
            print(f"  BH[{bi}] {getattr(bh, 'name', '<noname>')}: {len(layers)} layers")
            for li, ly in enumerate(layers):
                sl = getattr(ly, "soil_layer", None)
                m = getattr(sl, "material", None) if sl is not None else None
                m_name = getattr(m, "name", None)
                in_lib = 'yes' if m in (self.excavation_object.materials.get('soil_materials', []) or []) else 'no'
                print(f"    L{li} · SL={getattr(sl, 'name', None)} · mat={m_name} · lib?={in_lib} · plx_id={getattr(m, 'plx_id', None)}")

    def _index_soil_library(self) -> Dict[str, Any]:
        """Build a name→material map from pit.materials['soil_materials'] (case-sensitive)."""
        mats = (self.excavation_object.materials or {}).get("soil_materials", []) or []
        idx: Dict[str, Any] = {}
        for m in mats:
            n = getattr(m, "name", None)
            if n and n not in idx:
                idx[n] = m
        return idx

    def _relink_borehole_layers_to_library(self) -> int:
        """
        Ensure every SoilLayer.material points to the SAME instance as in the soil library.
        Returns the number of relinked layers.
        """
        pit = self.excavation_object
        bhset = getattr(pit, "borehole_set", None)
        if not bhset or not getattr(bhset, "boreholes", None):
            return 0

        idx = self._index_soil_library()
        fixed = 0

        for bh in bhset.boreholes:
            for ly in getattr(bh, "layers", []) or []:
                sl = getattr(ly, "soil_layer", None)
                if sl is None:
                    continue
                m = getattr(sl, "material", None)

                # If missing or dict-like, try resolve by SoilLayer name or material.name
                if m is None or isinstance(m, dict):
                    target = idx.get(getattr(m, "name", "")) if m else None
                    if target is None:
                        target = idx.get(getattr(sl, "name", ""))
                    if target is not None:
                        sl.material = target
                        fixed += 1
                    continue

                # If it's an instance but not the library one (different identity), fix by name
                m_name = getattr(m, "name", "")
                lib_m = idx.get(m_name)
                if lib_m is not None and (m is not lib_m):
                    sl.material = lib_m
                    fixed += 1

        if fixed:
            print(f"[materials] Relinked {fixed} soil-layer → material references to library.")
        return fixed

    # region Delete strutures
    # #########################################################################
    # Delete structures
    # #########################################################################

    def delete_well(self, well: Any) -> bool:
        """
        Delete the specified well.
        """
        return self.App.delete_well(well)
    
    def delete_all_wells(self) -> bool:
        """
        Delete all of the wells.
        """
        return self.App.delete_all_wells()
    
    #endregion

    # #########################################################################
    # SoilBlocks helpers
    # #########################################################################

    def _cut_inside_soil_blocks(self, pit: FoundationPit):
        """
        Identify enclosed soil bodies (inside the diaphragm walls) per soil layer,
        extrude and register them as SoilBlock objects.
        This assumes the retaining walls form a closed rectangle in plan.
        """
        if not pit.borehole_set:
            raise ValueError("Borehole set is required to compute soil layer depths.")

        walls = pit.structures.get(StructureType.RETAINING_WALLS.value, [])
        if len(walls) < 4:
            raise ValueError("Need at least 4 walls to define an enclosed region.")

        # Assume rectangular pit: extract envelope from wall geometry (z=top surface)
        all_pts = []
        for w in walls:
            poly = w.surface
            if hasattr(poly, "get_points"):
                all_pts.extend(poly.get_points())
        if not all_pts:
            raise ValueError("Cannot extract polygon points from walls.")

        z_top = max(p.z for p in all_pts)
        x_min = min(p.x for p in all_pts)
        x_max = max(p.x for p in all_pts)
        y_min = min(p.y for p in all_pts)
        y_max = max(p.y for p in all_pts)

        # Build rectangular top polygon and extrude layer-by-layer
        pit_polygon = Polygon3D.from_rectangle(x_min, y_min, x_max, y_max, z=z_top)

        # Extract ordered layers from canonical soil layers (not borehole layers)
        layers = pit.borehole_set.unique_soil_layers
        layers = sorted(layers, key=lambda l: float(l.top_z or 0.0), reverse=True)

        for i, layer in enumerate(layers):
            if layer.top_z is None or layer.bottom_z is None:
                continue  # skip incomplete layers
            z1 = layer.top_z
            z2 = layer.bottom_z
            name = f"L{i+1}_in"
            geom = pit_polygon.extrude(z1, z2)
            blk = SoilBlock(name=name, geometry=geom, material=layer.material)
            pit.add_structure(StructureType.SOIL_BLOCKS, blk)
    # #########################################################################
    # Phase helpers
    # #########################################################################

    def apply_phases(self, phases: List[Phase] = [], *, warn_on_missing: bool = False) -> Dict[str, Any]:
        """
        Create and APPLY staged-construction phases.

        What this does:
          - Creates phases in sequence (each inherits from the previous one).
          - Applies each phase via PlaxisRunner.apply_phase(...), which should handle:
            * phase options/settings
            * structure activation/deactivation
            * per-phase water table (if provided on the Phase)
            * per-phase well overrides (if your PhaseMapper supports it)
        """
        app = self.App
        pit = self.excavation_object

        # Ensure we are connected to PLAXIS and a project is open
        if not getattr(app, "is_connected", False) or getattr(app, "g_i", None) is None or getattr(app, "input_server", None) is None:
            app.connect().new()

        # Resolve the phase list
        if phases is None:
            phases = list(getattr(pit, "phases", []) or getattr(pit, "stages", []) or [])
        else:
            phases = list(phases or [])

        if not phases:
            print("[apply_phases] No phases to apply; skipping.")
            return {"created": 0, "applied": 0, "errors": []}

        # Switch to Staged Construction and get the initial phase handle
        app.goto_stages()
        prev_handle = app.get_initial_phase()

        created = 0
        applied = 0
        errors: List[str] = []
        handles: List[Any] = []

        for ph in phases:
            ph_name = getattr(ph, "name", "<unnamed>")

            # 1) create phase inheriting from the previous one, if the phase exists, skip
            handle = ph.plx_id
            if handle is None:
                try:
                    handle = app.create_phase(ph, inherits=prev_handle)
                    created += 1
                    handles.append(handle)
                except Exception as e:
                    errors.append(f"create_phase('{ph_name}') failed: {e}")
                    # Cannot apply if creation failed; keep prev_handle unchanged and continue
                    continue
                

            # 2) apply the phase (options, water table, well overrides, activations, etc.)
            try:
                app.apply_phase(ph, warn_on_missing=warn_on_missing)
                applied += 1
                prev_handle = handle  # advance the inheritance chain
            except Exception as e:
                errors.append(f"apply_phase('{ph_name}') failed: {e}")
                # Even if apply failed, still advance inheritance to maintain sequence
                prev_handle = handle

        return {"created": created, "applied": applied, "errors": errors, "handles": handles}

    def delete_all_phases(self):
        """
        Delete all of the phases in plaxis software.
        """
        self.App.delete_all_phases()
        self.excavation_object.clear_all_phases()

    # =========================================================================================
    # Excavation soillayers methods
    # =========================================================================================

    def get_excavation_soil_names(self, *, prefer_volume: bool = True) -> list[str]:
        """
        Return candidate enclosed (pit) child clusters to excavate, discovered
        in Staged Construction after build().
        Selection per parent group:
          - If volumes are available and prefer_volume=True: choose smallest volume.
          - Else: choose smallest numeric suffix (e.g., *_1).
        """
        app = self.App
        # make sure we are in stages (safe if already)
        try:
            app.goto_stages()
        except Exception:
            pass
        return list(app.get_excavation_soil_names(prefer_volume=prefer_volume))

    def get_remaining_soil_names(self, *, prefer_volume: bool = True) -> list[SoilBlock]:
        """
        Return remaining (non-excavated) child clusters, typically the larger pieces.
        Useful for 'freeze during dewatering' policies.
        """
        app = self.App
        try:
            app.goto_stages()
        except Exception:
            pass
        soil_blocks = []
        soils_blocks_dict = app.get_excavation_soils(prefer_volume=prefer_volume)
        for k, v in soils_blocks_dict.items():
            sb = SoilBlock(v, f"Soil block {str(k)}")
            sb.plx_id = k
            soil_blocks.append(sb)

        return soil_blocks

    # #########################################################################
    # Soil blocks registration & application (names -> deactivate in a phase)
    # #########################################################################

    def apply_soil_blocks(self, phase):
        """
        Resolve `phase.soil_blocks` (names -> PLAXIS soil handles) and perform
        `deactivate` for each soil in the given phase handle (staged construction).

        Notes:
        - This is intentionally minimal and only performs 'deactivate', because
            your requirement states "冻结土体" but the action specified is deactivate.
        - If you later need "freeze (non-deformable)" instead, replace the action
            with your project's freeze routine here.
        """
        try:
            self.App.apply_deactivate_soilblock(phase)
        except Exception as e:
            print(f"[Warning] Deactivate soilblock in failed.")

    def apply_pit_soil_block(self):
        """
        Apply deactivation of registered soil blocks for **all phases** in the
        current excavation sequence. And transfer the excavated soil layers to the next phase.
        """
        for phase in self.excavation_object.phases:
            if phase.inherits is not None:
                phase.add_soils(*phase.inherits.soil_blocks)
            self.apply_soil_blocks(phase)

    def get_all_child_soils(self) -> list[SoilBlock]:
        """
        Return all split child soils as SoilBlock objects after build().

        Each SoilBlock:
        - name: PLAXIS soil cluster name (e.g., "Soil_2_1")
        - description: simple label including the PLAXIS handle
        - plx_id: the PLAXIS handle for direct operations later
        """
        app = self.App
        # make sure we are in stages (safe if already)
        try:
            app.goto_stages()
        except Exception:
            pass

        soil_blocks: list[SoilBlock] = []
        # app.get_all_child_soils() is expected to return {handle -> name}
        soils_map = app.get_all_child_soils()
        for name, handle in soils_map.items():
            sb = SoilBlock(name, f"Soil block {handle}")
            sb.plx_id = handle
            soil_blocks.append(sb)

        return soil_blocks

    def get_all_child_soils_dict(self) -> dict[str, SoilBlock]:
        app = self.App
        try:
            app.goto_stages()
        except Exception:
            pass

        soil_blocks_dict: dict[str, SoilBlock] = {}
        # app.get_all_child_soils() is expected to return {handle -> name}
        soils_map = app.get_all_child_soils()
        for name, handle in soils_map.items():
            sb = SoilBlock(name, f"Soil block {handle}")
            sb.plx_id = handle
            soil_blocks_dict[name] = sb

        return soil_blocks_dict

    def get_all_child_soil_names(self) -> list[str]:
        """
        Convenience wrapper returning only names (list).
        """
        app = self.App
        try:
            app.goto_stages()
        except Exception:
            pass
        return app.get_all_child_soil_names()

    ################################################################################
    # Output Helper
    ################################################################################
    def is_output_connected(self) -> bool:
        """
        True if builder has a PlaxisOutput facade that can open Output on demand.
        (Stateless: no long-lived g_o/server is kept.)
        """
        client = getattr(self, "Output", None)
        return bool(client and getattr(client, "is_connected", False))


    def close_output_viewer(self) -> None:
        """
        Close and clear the current Output session (if any).
        """
        client = getattr(self, "Output", None)
        try:
            if client:
                client.close()
        except Exception:
            pass
        finally:
            self.Output = None


    def _normalize_phase_for_view(self, phase: Optional[Union[str, int, Any]]) -> Optional[Union[str, int, Any]]:
        """
        Convert a project Phase object to its name; default to the last project phase if phase is None.
        This helps Input.g_i.view(phase) open Output at a sensible phase and also keeps Output phase resolution simple.
        """
        if phase is None:
            phases = self.list_project_phases()
            if phases:
                last = phases[-1]
                return getattr(last, "name", last)
            return None
        if hasattr(phase, "name"):
            return getattr(phase, "name")
        return phase

    def create_output_viewer(self, phase: object = None, *, reuse: bool = True):
        """
        Backward helper: ensure Output bound to 'phase' (or last project phase).
        """
        if phase is None:
            phases = self.list_project_phases()
            if not phases:
                raise ValueError("No project phases available; provide a phase to bind Output.")
            phase = phases[-1]
        return self._ensure_output_for_phase(phase)

    def reconnect_output_viewer(
        self,
        phase: Optional[Union[str, int, Any]] = None,
        *,
        fixed_port: int = 10001,
    ) -> Optional[PlaxisOutput]:
        """
        Force a fresh Output connection, optionally launching the viewer for a given phase.
        """
        self.close_output_viewer()
        return self.create_output_viewer(phase=phase, reuse=False)

    def get_output_client(
        self,
        view_phase: Optional[Any] = None,
        *,
        reuse: bool = True,
    ) -> Optional[PlaxisOutput]:
        """
        Returns a connected Output client bound to `view_phase`.
        """
        return self.create_output_viewer(phase=view_phase, reuse=reuse)

    def mark_phase_calculate(self, phase, should_cal: bool = True):
        """
        Update a phase whether it should be calculated or not.
        """
        if phase in self.excavation_object.phases:
            print(f"The phase named {phase.name} does not exist, please check it.")
        self.App.mark_phase_should_calculate(phase, should_cal)

    def mark_all_phases_calculate(self):
        """
        Mark all of phases of the project should be calculate.
        """
        for phase in self.excavation_object.phases:
            self.mark_phase_calculate(phase)

    def list_project_phases(self) -> List[Any]:
        """
        Return project phases from the excavation object (names map 1:1 to Output phases).
        """
        pit = (
            getattr(self, "excavation_object", None)
            or getattr(self, "pit", None)
            or getattr(self, "_pit", None)
        )
        return list(getattr(pit, "phases", []) or [])

    def _ensure_phase_is_calculated(self, phase: Phase) -> bool:
        """
        Ensure the given phase has been calculated before querying Output.
        Strategy (in order):
        1) If we've run builder.calculate() in this session -> trust flag.
        2) Use Runner-provided probes when available:
            - is_calculated(phase)

        """
        # Fast-path: we already ran calculate() in this builder session
        if getattr(self, "_calc_done", False):
            return False

        return self.App.is_calculated(phase)


    def _ensure_output_for_phase(self, phase: object) -> Optional[PlaxisOutput]:
        """
        Ensure there is a connected Output client bound to the given phase.
        - If Output not connected: create and bind.
        - If Output bound to a different phase: rebind (preferred) or reconnect.
        Returns: self.Output (connected to 'phase').
        """
        # 1) Make sure Input (g_i) is ready
        g_i = getattr(self.App, "g_i", None)
        if g_i is None:
            # Connect Input once if needed
            self.App.connect()
            g_i = getattr(self.App, "g_i", None)
            if g_i is None:
                raise RuntimeError("Failed to connect to PLAXIS Input (g_i is None).")

        # 2) If no Output or not connected → create & bind
        if not self.is_output_connected():
            from ..plaxishelper.plaxisoutput import PlaxisOutput  # local import to avoid cycles
            po = PlaxisOutput(host=self.HOST, password=self.PASSWORD)
            self.Output = po.connect_via_input(g_i, phase)
            return self.Output

        # 3) If already connected, verify current bound phase vs requested one
        try:
            cur_id = getattr(self.Output, "_current_phase_id", None)
            # Let PlaxisOutput resolve phase id (keeps id resolution inside Output)
            resolve = getattr(self.Output, "_resolve_phase_id", None)
            tgt_id = resolve(phase) if callable(resolve) else getattr(phase, "plx_id", phase)
        except Exception:
            # If anything goes wrong resolving, force a reconnect to be safe
            self.close_output_viewer()
            from ..plaxishelper.plaxisoutput import PlaxisOutput
            po = PlaxisOutput(host=self.HOST, password=self.PASSWORD)
            self.Output = po.connect_via_input(g_i, phase)
            return self.Output

        if cur_id != tgt_id:
            # Prefer a light rebind (Output handles closing and recreating s_o/g_o)
            try:
                set_default_phase = getattr(self.Output, "set_default_phase", None)
                if callable(set_default_phase):
                    set_default_phase(g_i, phase)
            except Exception:
                # Fallback: hard reconnect
                self.close_output_viewer()
                from ..plaxishelper.plaxisoutput import PlaxisOutput
                po = PlaxisOutput(host=self.HOST, password=self.PASSWORD)
                self.Output = po.connect_via_input(g_i, phase)

        return self.Output
    
    ###########################################################################
    # Update the structures
    ###########################################################################
    def update_structures(self, structure_type: StructureType, structure_list: List, rebuild: bool = False):
        """
        Update the structures from a list.

        Args:
            rebuild: Rebuild the total simulation model.
        """
        self.excavation_object.update_structures(structure_type, structure_list)
        if rebuild:
            if structure_type == StructureType.ANCHORS:
                self.check_completeness()
                self.build_anchors()
            if structure_type == StructureType.BEAMS:
                self.check_completeness()
                self.build_beams()
            if structure_type == StructureType.EMBEDDED_PILES:
                self.check_completeness()
                self.build_piles()
            if structure_type == StructureType.RETAINING_WALLS:
                self.check_completeness()
                self.build_walls()
                self.build_bottom_surface()
            if structure_type == StructureType.WELLS:
                self.check_completeness()
                self.build_wells()

    def initial_phase(self, phase: Phase):
        """
        Initial the status of a phase.
        """
        phase.init_phase()
    
    def initial_all_phases(self):
        """
        Initial all the phases.
        """
        for phase in self.excavation_object.phases:
            self.initial_phase(phase)

    def initial_all_phase_soilblocks(self):
        for phase in self.excavation_object.phases:
            phase.init_soilblocks()

    #region Get Results
    ###########################################################################
    # Get Results
    ###########################################################################

    def get_results(
        self,
        *,
        structure: Optional[BaseStructure] = None,  # optional: structure object or its plx_id
        leaf: Enum,                                  # resulttypes enum member, or "Plate.UX" string
        phase: Optional[Phase] = None,               # Phase object or its plx_id
        smoothing: bool = False,
        raw: bool = False
    ) -> Union[List[float], float, str, Iterable, None]:
        """
        One-call result fetch with pre-checks:
        - Ensure the phase was calculated (or print a warning).
        - Ensure Output is bound to the given phase (create/rebind if needed).
        - Delegate to PlaxisOutput.get_results(...).

        Parameters
        ##########
        structure : BaseStructure | None
            If provided, results are fetched for this structure.
            If None, results are fetched for "all soils"/global leaf.
        leaf : Enum
            Result type enum (e.g. g_o.ResultTypes.Soil.Uy).
        phase : Phase | None
            Phase object or id. Must not be None here unless you implement
            a "default phase" fallback inside _ensure_* helpers.
        smoothing : bool
            Whether to use smoothed results.
        """

        # If you want to support a "current/default phase", you can
        # replace this check with your own fallback logic.
        if phase is None:
            raise ValueError("phase must be provided for get_results().")

        # 1) must be calculated first
        if not self._ensure_phase_is_calculated(phase):
            print(f"{phase.name} was not calculated yet!")

        # 2) ensure Output bound to this phase (creates/rebinds as needed)
        po = self._ensure_output_for_phase(phase)

        # 3) fetch and return via PlaxisOutput
        if isinstance(po, PlaxisOutput):
            # structure provided -> use (structure, leaf, smoothing)
            if structure is not None:
                return po.get_results(structure, leaf, smoothing=smoothing, raw=raw)
            # no structure -> use (leaf, smoothing)
            return po.get_results(leaf, smoothing=smoothing, raw=raw)

        return None
    
    # Cache the position and scalar field of the soil.

    def _get_or_build_soil_picker(
        self,
        phase: Phase,
    ) -> Tuple[NeighborPointPicker, np.ndarray]:
        """
        For a given phase, fetch Soil.X/Y/Z once and build a KD-tree based
        NeighborPointPicker on the soil grid.

        Returns
        -------
        picker : NeighborPointPicker
            KD-tree accelerated neighbor search on (x, y, z).
        coords : np.ndarray, shape (N, 3)
            Coordinates corresponding to soil result points.
        """
        phase_name = getattr(phase, "name", str(phase))

        # cached?
        picker = self._soil_coord_picker_cache.get(phase_name)
        coords = self._soil_coord_cache.get(phase_name)
        if picker is not None and coords is not None:
            return picker, coords

        # Use generic get_results wrapper; raw=True -> full field as 1D array
        xs_raw = self.get_results(
            leaf=SoilResult.X,
            phase=phase,
            smoothing=False,
            raw=True,
        )
        ys_raw = self.get_results(
            leaf=SoilResult.Y,
            phase=phase,
            smoothing=False,
            raw=True,
        )
        zs_raw = self.get_results(
            leaf=SoilResult.Z,
            phase=phase,
            smoothing=False,
            raw=True,
        )

        if xs_raw is None or ys_raw is None or zs_raw is None:
            raise RuntimeError(f"Soil.X/Y/Z results are not available for phase {phase_name!r}.")

        xs = np.asarray(xs_raw, dtype=float).ravel()
        ys = np.asarray(ys_raw, dtype=float).ravel()
        zs = np.asarray(zs_raw, dtype=float).ravel()

        if not (xs.size == ys.size == zs.size):
            raise ValueError(
                f"Soil.X/Y/Z lengths mismatch for phase {phase_name!r}: "
                f"{xs.size}, {ys.size}, {zs.size}"
            )

        coords = np.vstack([xs, ys, zs]).T  # (N, 3)

        picker = NeighborPointPicker(
            xs=xs,
            ys=ys,
            zs=zs,
            use_kdtree=True,   # use KD-tree backend
        )

        self._soil_coord_picker_cache[phase_name] = picker
        self._soil_coord_cache[phase_name] = coords
        return picker, coords

    def _get_or_fetch_soil_field(
        self,
        phase: Phase,
        leaf: SoilResult,
        *,
        smoothing: bool = False,
    ) -> np.ndarray:
        """
        Fetch (or return cached) soil result field for a given phase and result type,
        as a flat numpy array.

        leaf examples: SoilResult.Uz, SoilResult.PActive, etc.
        """
        phase_name = getattr(phase, "name", str(phase))
        key = (phase_name, leaf.name, bool(smoothing))

        cached = self._soil_field_cache.get(key)
        if cached is not None:
            return cached

        vals_raw = self.get_results(
            leaf=leaf,
            phase=phase,
            smoothing=smoothing,
            raw=True,
        )
        if vals_raw is None:
            raise RuntimeError(
                f"No results returned for {leaf!r} in phase {phase_name!r} (raw=True)."
            )

        vals = np.asarray(vals_raw, dtype=float).ravel()
        self._soil_field_cache[key] = vals
        return vals
    
    def reset_soil_result_cache(self):
        """
        Reset all dicts of the soil results.
        """
        self._soil_coord_cache = {}
        self._soil_coord_picker_cache = {}
        self._soil_field_cache = {}
        

    def _estimate_water_depth_for_point_fast(
        self,
        picker: NeighborPointPicker,
        pvals: np.ndarray,
        x: float,
        y: float,
        z_ground: float,
        z_min: float,
        *,
        tol: float = 0.1,
    ) -> Optional[float]:
        """
        Fast groundwater depth estimation using a pre-fetched Soil.PActive field
        and a KDTree-based NeighborPointPicker.

        Approximation:
        - P(x,y,z) is approximated by the value at the nearest soil-field point
          in (x,y,z) space.
        - Otherwise, logic is the same as _estimate_water_depth_for_point:
            * assume P≈0 above water table, P<0 below,
            * bisection on "P(z) < 0" between z_ground and z_min.
        """
        # ensure z_top > z_bottom
        z_top = max(z_ground, z_min)
        z_bottom = min(z_ground, z_min)

        if x == -98.3 and y == -29.8:
            print(x, y)

        def p_at(z: float) -> Optional[float]:
            idx, _coord = picker.nearest_to_coord(x, y, float(z))
            v = pvals[idx]
            if isinstance(v, (int, float, np.floating)):
                return float(v)
            return None

        # 1) check top and bottom pore pressure
        p_top = p_at(z_top)
        p_bot = p_at(z_bottom)

        if p_top is None or p_bot is None:
            return None

        # Above water: pressure should be 0. If already negative, water is above ground.
        if p_top < 0.0:
            return 0.0

        # Below water: pressure must be negative; otherwise assumptions fail.
        if p_bot >= 0.0:
            # water table is deeper than z_min or model does not follow 0/negative pattern
            return None

        # Predicate: "below water table" (True if P < 0)
        def is_below(z: float) -> Optional[bool]:
            p = p_at(z)
            if p is None:
                return None
            return p < 0.0

        # initial booleans
        b_top = is_below(z_top)   # expected False
        b_bot = is_below(z_bottom)  # expected True

        if b_top is None or b_bot is None:
            return None

        # If top already "below" -> water above ground
        if b_top:
            return 0.0

        # If bottom not below -> no negative values in interval
        if not b_bot:
            return None

        # 2) bisection on boolean step: False (above) -> True (below)
        z_lo, b_lo = z_bottom, True
        z_hi, b_hi = z_top, False

        while (z_hi - z_lo) > tol:
            z_mid = 0.5 * (z_lo + z_hi)
            b_mid = is_below(z_mid)

            if b_mid is None:
                # if invalid, shrink interval conservatively towards top
                z_hi = z_mid
                continue

            if b_mid:
                # mid is already below water table (P<0)
                # interface lies between z_hi (above) and z_mid (below)
                z_lo, b_lo = z_mid, True
            else:
                # mid is still above water table (P>=0)
                # interface lies between z_mid (above) and z_lo (below)
                z_hi, b_hi = z_mid, False

        # 此时 z_hi ≈ 刚刚“差一点进入负水位”的高度（最后一个仍为 P>=0 的位置）
        z_water = z_hi
        return z_water

    def sample_water_depth_for_points_fast(
        self,
        phase,
        xy_points,
        *,
        z_ground: float,
        z_min: float,
        tol: float = 0.1,
        smoothing: bool = False,
    ):
        """
        Fast version of groundwater depth sampling using soil-field + NeighborPointPicker.

        For each (x, y), estimate depth where PActive crosses from ~0 to negative
        between z_ground and z_min using a boolean bisection, with P(z) obtained
        via nearest-neighbor lookup in a pre-fetched Soil.PActive field.
        """
        # 1) KDTree picker on Soil.X/Y/Z
        picker, _coords = self._get_or_build_soil_picker(phase)
        # 2) Soil.PActive field as array
        pvals = self._get_or_fetch_soil_field(
            phase,
            SoilResult.PActive,
            smoothing=smoothing,
        )

        depths: List[Optional[float]] = []
        for (x, y) in xy_points:
            d = self._estimate_water_depth_for_point_fast(
                picker,
                pvals,
                float(x),
                float(y),
                float(z_ground),
                float(z_min),
                tol=float(tol),
            )
            depths.append(d)
        return depths

    def sample_settlement_for_points_fast(
        self,
        phase,
        xy_points,
        *,
        z_ground: float,
        smoothing: bool = False,
    ):
        """
        Fast version of sample_settlement_for_points using a pre-fetched Soil.Uz
        field and a KDTree-based NeighborPointPicker on Soil.X/Y/Z.

        For each (x, y), we query the nearest (x, y, z_ground) soil point and
        read Uz from the cached field (raw=True).
        """
        # 1) KDTree picker on Soil.X/Y/Z
        picker, _coords = self._get_or_build_soil_picker(phase)
        # 2) Soil.Uz field as array
        uz_vals = self._get_or_fetch_soil_field(
            phase,
            SoilResult.Uz,
            smoothing=smoothing,
        )

        values: List[Optional[float]] = []
        for (x, y) in xy_points:
            idx, _coord = picker.nearest_to_coord(
                float(x),
                float(y),
                float(z_ground),
            )
            # uz_vals 是 float 数组，按索引取即可
            val = float(uz_vals[idx])
            values.append(val)
        return values

    # ########## helper: Pick up the deformation of walls ##################

    def get_plate_displacement_for_structure(
        self,
        phase,
        structure,
        include_uz: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Get plate results (X, Y, Z, Ux, Uy[, Uz]) for a given structure (e.g. a retaining wall)
        in a given phase, converted to float numpy arrays.

        This is a thin wrapper over builder.get_results, but centralizes:
        - result leaf selection (X, Y, Z, Ux, Uy, Uz)
        - list/scalar -> np.ndarray conversion
        - error handling (missing leaves -> empty arrays)
        """
        keys = ["X", "Y", "Z", "Ux", "Uy"]
        if include_uz:
            keys.append("Uz")

        result: Dict[str, np.ndarray] = {}

        for k in keys:
            # PlateResult.X / PlateResult.Ux / ...
            leaf = getattr(PlateResult, k, None)
            if leaf is None:
                # This result is not supported by current PLAXIS version
                result[k] = np.asarray([], dtype=float)
                continue

            try:
                raw = self.get_results(
                    structure=structure,
                    leaf=leaf,
                    phase=phase,
                    smoothing=False,
                )
            except Exception:
                arr = np.asarray([], dtype=float)
            else:
                # normalize to 1D float array
                if isinstance(raw, list):
                    arr = np.asarray(raw, dtype=float).ravel()
                elif isinstance(raw, (int, float, np.floating)):
                    arr = np.asarray([float(raw)], dtype=float)
                else:
                    arr = np.asarray(raw, dtype=float).ravel()

            result[k] = arr

        return result

    # ########## helper: estimate groundwater depth at one (x, y) ##########
    def _estimate_water_depth_for_point(
        self,
        po,
        phase,
        x: float,
        y: float,
        z_ground: float,
        z_min: float,
        *,
        tol: float = 0.1,
        smoothing: bool = False,
    ):
        """
        Estimate groundwater depth below z_ground at (x, y) assuming:

            Pore pressure = 0 above water table,
            Pore pressure < 0 below water table.

        We assume:
        - P(z_ground) ≈ 0
        - P(z_min) < 0
        - There is a unique transition from 0 to negative between them.

        Algorithm:
        - Binary search on the boolean predicate (P(z) < 0):
            * at z_ground: False
            * at z_min: True
        - Find the smallest depth where P becomes negative
          with vertical accuracy <= tol (m).

        Returns
        #######
        depth : float or None
            Groundwater depth below z_ground (>= 0), or None if assumptions
            are violated or results are not available.
        """
        # ensure z_top > z_bottom
        z_top = max(z_ground, z_min)
        z_bottom = min(z_ground, z_min)

        # 1) check top and bottom pore pressure
        p_top = po.get_single_result_at(
            phase, SoilResult.PActive, x, y, z_top, smoothing
        )
        p_bot = po.get_single_result_at(
            phase, SoilResult.PActive, x, y, z_bottom, smoothing
        )

        if not isinstance(p_top, (int, float)) or not isinstance(p_bot, (int, float)):
            return None

        # Above water: pressure should be 0. If already negative, water is above ground.
        if p_top < 0.0:
            return 0.0

        # Below water: pressure must be negative; otherwise assumptions fail.
        if p_bot >= 0.0:
            # water table is deeper than z_min or model does not follow 0/negative pattern
            return None

        # Predicate: "below water table" (True if P < 0)
        def is_below(z: float) -> bool | None:
            p = po.get_single_result_at(
                phase, SoilResult.PActive, x, y, z, smoothing
            )
            if not isinstance(p, (int, float)):
                return None
            return p < 0.0

        # initial booleans
        b_top = is_below(z_top)   # expected False
        b_bot = is_below(z_bottom)  # expected True

        if b_top is None or b_bot is None:
            return None

        # If top already "below" -> water above ground
        if b_top:
            return 0.0

        # If bottom not below -> no negative values in interval
        if not b_bot:
            return None

        # 2) bisection on boolean step: False (above) -> True (below)
        z_lo, b_lo = z_bottom, True   # below
        z_hi, b_hi = z_top, False     # above

        while (z_hi - z_lo) > tol:
            z_mid = 0.5 * (z_lo + z_hi)
            b_mid = is_below(z_mid)

            if b_mid is None:
                # if invalid, shrink interval conservatively
                z_hi = z_mid
                continue

            if b_mid:
                # still below water table -> transition is above
                z_hi, b_hi = z_mid, True
            else:
                # still above water table -> transition is deeper
                z_lo, b_lo = z_mid, False

        # At the end, z_hi is the highest depth where P<0 (first negative point)
        z_water = z_hi
        depth = z_top - z_water
        if depth < 0.0:
            depth = 0.0
        return depth
    

    def sample_water_depth_for_points(
        self,
        phase,
        xy_points,
        *,
        z_ground: float,
        z_min: float,
        tol: float = 0.1,
        smoothing: bool = False,
    ):
        """
        Compute groundwater depth for a list of (x, y) locations in a single phase,
        using a boolean bisection on:

            PorePressure(z) < 0  (below water table)

        Assumes:
        - P(z_ground) == 0
        - P(z_min) < 0

        Parameters
        ##########
        phase : Phase
            Calculation stage / Phase object (with .name).
        xy_points : iterable[(x, y)]
            List of plan coordinates.
        z_ground : float
            Ground surface elevation (upper bound).
        z_min : float
            Lower bound (must be below expected water table).
        tol : float
            Target accuracy in meters (default 0.1 m).
        smoothing : bool
            Whether to enable smoothing in getsingleresult.

        Returns
        #######
        List[float | None]
            Groundwater depths for each point; None if cannot be determined.
        """
        po = self._ensure_output_for_phase(phase)
        if po is None:
            raise RuntimeError("Output viewer is not available for this phase.")

        depths = []
        for (x, y) in xy_points:
            d = self._estimate_water_depth_for_point(
                po,
                phase,
                float(x),
                float(y),
                float(z_ground),
                float(z_min),
                tol=float(tol),
                smoothing=smoothing,
            )
            depths.append(d)
        return depths


    def sample_settlement_for_points(
        self,
        phase,
        xy_points,
        *,
        z_ground: float,
        smoothing: bool = False,
    ):
        """
        Sample vertical displacement (Soil.Uz) at z = z_ground for a list
        of (x, y) locations, in a single phase.

        Parameters
        ##########
        phase : Phase
            Calculation stage / Phase object (with .name).
        xy_points : iterable[(x, y)]
            List of plan coordinates.
        z_ground : float
            Elevation where settlement is sampled.
        smoothing : bool
            Whether to enable smoothing in getsingleresult.

        Returns
        #######
        List[float | None]
            Uz at each point; None if result is not available.
        """
        po = self._ensure_output_for_phase(phase)
        if po is None:
            raise RuntimeError("Output viewer is not available for this phase.")

        values = []
        for (x, y) in xy_points:
            uz = po.get_single_result_at(
                phase, SoilResult.Uz, float(x), float(y), float(z_ground), smoothing
            )
            if isinstance(uz, (int, float)):
                values.append(float(uz))
            else:
                values.append(None)
        return values

    #endregion

    #region Project operations
    ###########################################################################
    # Save and Load
    ###########################################################################

    def save_project(self, path: Optional[str] = None, *, overwrite: bool = False) -> str:
        """
        Save the current PLAXIS project via Runner.
        - If `path` is None, requires the project to already have a file name in PLAXIS.
        - If `path` is provided, performs a Save-As to that path.
        Returns the absolute path of the saved project.
        """
        return self.App.save_project(path=path, overwrite=overwrite)

    def save_as(self, path: str, *, overwrite: bool = True) -> str:
        """
        Convenience alias for Save-As. Overwrite defaults to True.
        Returns the absolute path of the saved project.
        """
        return self.App.save_as(path, overwrite=overwrite)

    def load_project(self, path: str) -> str:
        """
        Load an existing PLAXIS project via Runner.
        - Closes the current Output viewer (if any) to avoid stale connections.
        - Calls Runner.load_project(path).
        - Resets internal calculation flag (a newly loaded project may not be calculated).
        Returns the absolute path that was opened.
        """
        # Make sure we do not keep an Output session bound to the previous project
        try:
            self.close_output_viewer()
        except Exception:
            pass
        opened = self.App.load_project(path)
        # New project → results state unknown; require recalculation before queries
        try:
            self._calc_done = False
        except Exception:
            pass
        return opened

    def open_project(self, path: str) -> str:
        """
        Alias of load_project(path).
        """
        return self.load_project(path)

    def get_project_path(self) -> Optional[str]:
        """
        Return the last known project file path (best-effort).
        Delegates to Runner.get_project_path().
        """
        return self.App.get_project_path()

    # ######## Optional convenience: save using ProjectInformation path ##########

    def save_to_project_info_path(self, *, overwrite: bool = True) -> str:
        """
        Save-As using FoundationPit's ProjectInformation (dir + file_name) if available.
        - If dir is missing, defaults to current working directory.
        - If file_name is missing, falls back to 'project'.
        - Runner will append the proper extension when missing (e.g., .p3d).
        Returns the absolute path of the saved project.
        """
        pit = getattr(self, "excavation_object", None)
        proj = getattr(pit, "project_information", None) if pit is not None else None
        if proj is None:
            raise ValueError("ProjectInformation is not available on the excavation_object.")
        dir_ = getattr(proj, "dir", None) or "."
        fname = getattr(proj, "file_name", None) or "project"
        import os
        full = os.path.join(dir_, fname)
        return self.App.save_as(full, overwrite=overwrite)

    #endregion