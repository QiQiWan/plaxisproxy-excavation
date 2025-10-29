from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from math import ceil
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence

from ..plaxishelper.plaxisrunner import PlaxisRunner
from .plaxis_config import *
from ..excavation import FoundationPit, StructureType, _normalize_structure_type
from ..structures.soilblock import SoilBlock
from ..geometry import Polygon3D
from ..components.mesh import Mesh

from plaxis_config import HOST, PORT, PASSWORD

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
-------------------------------------------
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

    def __init__(self, app: PlaxisRunner, excavation_object: FoundationPit) -> None:
        # Always create our own runner instance using config; keep provided `app` for signature parity.
        self.App: PlaxisRunner = PlaxisRunner(PORT, PASSWORD, HOST)
        self.excavation_object: FoundationPit = excavation_object
        self.Output: PlaxisOutput
        self._calc_done: bool = False  # True after a successful builder.calculate()

    @classmethod
    def create(cls, app: PlaxisRunner, excavation_object: FoundationPit):
        return cls(app, excavation_object)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize builder: validate pit object and ensure a fresh PLAXIS project."""
        self.check_completeness()
        self.App.connect().new()

    def calculate(self) -> None:
        """Go to the stages interface, and start a calculation task."""
        self.App.calculate()
        # If no exception was raised, mark as calculated
        self._calc_done = True

    # -------------------------------------------------------------------------
    # Validation (aligned with current FoundationPit definition)
    # -------------------------------------------------------------------------

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

        # ---- Project information (required) ----
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

        # ---- Name (optional): prefer explicit pit.name; fallback to project title ----
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

        # ---- Boreholes (optional) ----
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

        # ---- Materials (init empty buckets if needed) ----
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

        # ---- Structures (init empty buckets if needed) ----
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

        # ---- Loads (init empty buckets if needed) ----
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

        # ---- Phases (optional at build-time; required only for apply_phases) ----
        phases = getattr(pit, "phases", None)
        if phases is None:
            pit.phases = []
            warns.append("No phases defined; call builder.apply_phases() later to stage the model.")
        elif not isinstance(phases, (list, tuple)):
            warns.append("phases is not a list; converting to empty list for safety.")
            pit.phases = []

        # ---- Finalize ----
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

    # -------------------------------- Excavation volumn check --------------------------------

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

    # ---------- New public helpers: wall footprint / bottom polygon ----------

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



    # -------------------------------------------------------------------------
    # Build (initial design only)
    # -------------------------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        """Build the base PLAXIS model from the FoundationPit object.

        IMPORTANT:
          - Builds ONLY the initial design:
              * project info
              * materials
              * relink borehole-layer materials → boreholes & layers
              * geometries & structures (including wells)
              * loads (+ optional multipliers)
              * monitor points
              * mesh
              * phase shells (created in sequence, NOT applied)
          - It does NOT apply per-phase options, water table, or well parameter changes.
            Those belong to phases and must be applied later via `apply_phases(...)`.
        """
        pit = self.excavation_object
        app = self.App

        # Ensure connection & a new project exists
        if not getattr(app, "is_connected", False) or getattr(app, "g_i", None) is None or getattr(app, "input_server", None) is None:
            app.connect().new()

        # Optional: completeness check (will raise on blocking issues)
        if hasattr(self, "check_completeness") and callable(self.check_completeness):
            self.check_completeness()

        # ---------------- helpers ----------------
        def _iter_dict_lists(dct):
            """Yield items from a dict-of-lists safely."""
            if isinstance(dct, dict):
                for _k, _lst in dct.items():
                    for _it in (_lst or []):
                        if _it is not None:
                            yield _k, _it

        # ---------------- 1) Project info ----------------
        if getattr(pit, "project_information", None) is not None:
            try:
                app.apply_project_information(pit.project_information)
            except Exception as e:
                print(f"[build] Warning: apply_project_information failed: {e}")

        # ---------------- 2) Materials (create first) ----------------
        total_mats = 0
        try:
            for _k, mat in _iter_dict_lists(getattr(pit, "materials", {})):
                app.create_material(mat)
                total_mats += 1
        except Exception as e:
            print(f"[build] Warning: create_material failed on some items: {e}")

        # ---------------- 3) Relink BH layers to library & create boreholes ----
        try:
            self._relink_borehole_layers_to_library()
        except Exception as e:
            print(f"[build] Warning: relink borehole layers failed: {e}")

        if getattr(pit, "borehole_set", None) is not None:
            try:
                app.create_boreholes(pit.borehole_set)  # mapper should normalize layers
            except Exception as e:
                print(f"[build] Warning: create_boreholes failed: {e}")

        # ---------------- 4) Geometries / soil blocks (optional) ---------------
        structures = getattr(pit, "structures", {}) or {}
        # for blk in structures.get("soil_blocks", []) or []:
        #     try:
        #         app.create_soil_block(blk)
        #     except Exception as e:
        #         print(f"[build] Warning: create_soil_block failed: {e}")

        # ---------------- 5) Structures (including wells) ----------------------
        for wall in structures.get(StructureType.RETAINING_WALLS.value, []) or []:
            try:
                app.create_retaining_wall(wall)
            except Exception as e:
                print(f"[build] Warning: create_retaining_wall failed: {e}")

        # ---------------- 5.1) Insert the bottom of the excavation -----------------------
        try:
            ok = bool(getattr(self, "_bottom_surface_ok", False))
        except Exception:
            ok = False

        # if ok:
        #     try:
        #         excava_depth = float(getattr(pit, "excava_depth", 0.0))
        #         walls_now = pit.structures.get("retaining_walls", []) if isinstance(pit.structures, dict) else []

        #         bottom_poly = self._make_bottom_polygon3d(walls_now, excava_depth)
        #         if bottom_poly is None:
        #             print("[build] Warning: could not derive bottom polygon from walls (need >=3 points).")
        #         else:
        #             app = self.App  # runner facade
        #             app.create_surface(bottom_poly, name="ExcavationBottom", auto_close=True)
        #             print(f"[build] Bottom surface created at z={excava_depth:.3f} using wall-enclosed polygon.")
        #     except Exception as e:
        #         print(f"[build] Warning: failed to create bottom surface: {e}")

        if ok:
            try:
                excava_depth = float(getattr(pit, "excava_depth", 0.0))
                bottom_poly = self.get_wall_footprint_polygon3d(z_value=excava_depth, tol=1e-6)
                if bottom_poly is None:
                    print("[build] Warning: could not derive bottom polygon from walls (need >=3 points).")
                else:
                    app = self.App
                    app.create_surface(bottom_poly, name="ExcavationBottom", auto_close=True)
                    print(f"[build] Bottom surface created at z={excava_depth:.3f} using wall-enclosed polygon.")
            except Exception as e:
                print(f"[build] Warning: bottom polygon creation failed: {e}")

        # ---------------- 5.2) Create beams -----------------------
        for beam in structures.get(StructureType.BEAMS.value, []) or []:
            try:
                app.create_beam(beam)
            except Exception as e:
                print(f"[build] Warning: create_beam failed: {e}")

        for anc in structures.get(StructureType.ANCHORS.value, []) or []:
            try:
                app.create_anchor(anc)
            except Exception as e:
                print(f"[build] Warning: create_anchor failed: {e}")

        for pile in structures.get(StructureType.EMBEDDED_PILES.value, []) or []:
            try:
                app.create_embedded_pile(pile)
            except Exception as e:
                print(f"[build] Warning: create_embedded_pile failed: {e}")

        # Wells — created once at structure stage (geometry-level only)
        for well in structures.get(StructureType.WELLS.value, []) or []:
            try:
                app.create_well(well)
            except Exception as e:
                print(f"[build] Warning: create_well failed: {e}")

        if not structures.get(StructureType.SOIL_BLOCKS.value):
            self._cut_inside_soil_blocks(self.excavation_object)

        for blk in structures.get(StructureType.SOIL_BLOCKS.value, []):
            app.create_soil_block(blk)

        # ---------------- 6) Loads (+ optional multipliers) --------------------
        loads = getattr(pit, "loads", {}) or {}
        total_loads = 0
        try:
            for _k, ld in _iter_dict_lists(loads):
                app.create_load(ld)
                total_loads += 1
        except Exception as e:
            print(f"[build] Warning: create_load failed on some items: {e}")

        for mul in getattr(pit, "load_multipliers", []) or []:
            try:
                app.create_load_multiplier(mul)
            except Exception as e:
                print(f"[build] Warning: create_load_multiplier failed: {e}")

        # ---------------- 7) Monitor points (best-effort) ----------------------
        monitors = getattr(pit, "monitors", []) or []
        if monitors:
            try:
                if getattr(app, "g_i", None) is None:
                    raise RuntimeError("Not connected (g_i is None).")
                try:
                    from ..plaxishelper.monitormapper import MonitorMapper  # type: ignore
                except Exception:
                    from plaxisproxy_excavation.plaxishelper.monitormapper import MonitorMapper  # type: ignore
                MonitorMapper.create_monitors(app.g_i, monitors)  # type: ignore
            except Exception as e:
                print(f"[build] Warning: monitor mapping skipped: {e}")

        # ---------------- 8) Mesh (optional) -----------------------------------
        meshed = False
        mesh_cfg = getattr(pit, "mesh", None)
        if mesh_cfg is None:
            mesh_cfg = Mesh()
        else:
            print("[build] Info: no mesh config on FoundationPit; use the default!")
        try:
            app.mesh(mesh_cfg)
            meshed = True
        except Exception as e:
            print(f"[build] Warning: mesh() failed: {e}")

        # ---------------- 9) Phase shells ONLY (no apply) ----------------------
        phases = list(getattr(pit, "phases", [])) or getattr(pit, "stages", []) or []
        created_phase_handles = []
        if phases:
            try:
                app.goto_stages()
                prev = app.get_initial_phase()
                for ph in phases:
                    for blk in pit.structures.get(StructureType.SOIL_BLOCKS.value, []):
                        if blk.name.endswith("_in"):
                            phase_name = blk.name[:-3]  # 去掉后缀“_in”得到阶段名
                            for ph in phases:  # phases为待应用的Phase列表
                                if ph.name == phase_name:
                                    ph.deactivate_structures(blk)
                    # Create the phase entity (inheriting sequence), DO NOT apply here
                    h = app.create_phase(ph, inherits=prev)
                    created_phase_handles.append(h)
                    prev = h
            except Exception as e:
                print(f"[build] Warning: phase creation failed: {e}")

        # Summary for logs/tests
        return {
            "materials": total_mats,
            "structures": {k: len(v or []) for k, v in (structures or {}).items()},
            "loads": total_loads,
            "monitors": len(monitors),
            "phases_created": len(created_phase_handles),
            "meshed": meshed,
        }

    # -------------------------------------------------------------------------
    # Soil-material relinking helpers (borehole layers → library)
    # -------------------------------------------------------------------------

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
                    target = idx.get(getattr(m, "name", None)) if m else None
                    if target is None:
                        target = idx.get(getattr(sl, "name", None))
                    if target is not None:
                        sl.material = target
                        fixed += 1
                    continue

                # If it's an instance but not the library one (different identity), fix by name
                m_name = getattr(m, "name", None)
                lib_m = idx.get(m_name)
                if lib_m is not None and (m is not lib_m):
                    sl.material = lib_m
                    fixed += 1

        if fixed:
            print(f"[materials] Relinked {fixed} soil-layer → material references to library.")
        return fixed

    # -------------------------------------------------------------------------
    # SoilBlocks helpers
    # -------------------------------------------------------------------------

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
    # -------------------------------------------------------------------------
    # Phase helpers
    # -------------------------------------------------------------------------

    def apply_phases(self, phases: Optional[List[Any]] = None, *, warn_on_missing: bool = False) -> Dict[str, Any]:
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
            # 1) create phase inheriting from the previous one
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
                app.apply_phase(handle, ph, warn_on_missing=warn_on_missing)
                applied += 1
                prev_handle = handle  # advance the inheritance chain
            except Exception as e:
                errors.append(f"apply_phase('{ph_name}') failed: {e}")
                # Even if apply failed, still advance inheritance to maintain sequence
                prev_handle = handle

        return {"created": created, "applied": applied, "errors": errors, "handles": handles}

    def apply_well_overrides_only(self, plan: Dict[Any, Dict[str, Dict[str, Any]]], *, warn_on_missing: bool = False) -> Dict[str, Any]:
        """Update well parameters for existing phases ONLY (no phase creation).

        plan format:
          {
            <phase_handle_or_name>: {
                "Well-1": {"q_well": 900.0, "h_min": 1.5},
                "Well-2": {"q_well": 700.0, "well_type": "Extraction"}
            },
            ...
          }

        - If the key is a phase handle, it will be used directly.
        - If the key is a string, we'll try to resolve an existing phase by that name via `PlaxisRunner.find_phase_handle`.
        - No new phase will be created here.
        """
        app = self.App
        if not getattr(app, "is_connected", False) or getattr(app, "g_i", None) is None or getattr(app, "input_server", None) is None:
            app.connect().new()

        applied = 0
        for phase_key, overrides in (plan or {}).items():
            handle = phase_key
            # resolve by name if a string key is provided
            if isinstance(phase_key, str):
                handle = app.find_phase_handle(phase_key)

            if handle is None:
                if warn_on_missing:
                    print(f"[apply_well_overrides_only] Phase '{phase_key}' not found; skipped.")
                continue

            try:
                app.apply_well_overrides(handle, overrides, warn_on_missing=warn_on_missing)
                applied += 1
            except Exception as e:
                print(f"[apply_well_overrides_only] Warning: overrides on phase '{phase_key}' failed: {e}")

        return {"phases_updated": applied}

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

    # -------------------------------------------------------------------------
    # Soil blocks registration & application (names -> deactivate in a phase)
    # -------------------------------------------------------------------------

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
        current excavation sequence.
        """
        for phase in self.excavation_object.phases:
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
    ) -> PlaxisOutput:
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
    ) -> PlaxisOutput:
        """
        Returns a connected Output client bound to `view_phase`.
        """
        return self.create_output_viewer(phase=view_phase, reuse=reuse)


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

    def _ensure_phase_is_calculated(self, phase: object) -> None:
        """
        Ensure the given phase has been calculated before querying Output.
        Strategy (in order):
        1) If we've run builder.calculate() in this session -> trust flag.
        2) Use Runner-provided probes when available:
            - is_phase_calculated(phase)
            - phase_has_results(handle)
            - is_project_calculated()
        3) If none available / uncertain -> raise with a clear message.
        """
        # Fast-path: we already ran calculate() in this builder session
        if getattr(self, "_calc_done", False):
            return

        app = self.App

        # 1) Direct probe: is_phase_calculated(phase)
        probe = getattr(app, "is_phase_calculated", None)
        if callable(probe):
            try:
                if bool(probe(phase)):
                    return
            except Exception:
                pass

        # 2) Indirect probe via handle + phase_has_results(handle)
        try:
            find_h = getattr(app, "find_phase_handle", None)
            has_res = getattr(app, "phase_has_results", None)
            if callable(find_h) and callable(has_res):
                h = find_h(phase)  # accept Phase object or identifier
                if h is not None and bool(has_res(h)):
                    return
        except Exception:
            pass

        # 3) Project-level probe
        proj_probe = getattr(app, "is_project_calculated", None)
        if callable(proj_probe):
            try:
                if bool(proj_probe()):
                    return
            except Exception:
                pass

        # If we reach here, we cannot establish that results exist for this phase
        raise RuntimeError(
            "Requested results before calculation. Please run builder.calculate() "
            "or otherwise ensure the target phase has finished calculation."
        )


    def _ensure_output_for_phase(self, phase: object) -> "PlaxisOutput":
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
            po = PlaxisOutput(host=HOST, password=PASSWORD)
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
            po = PlaxisOutput(host=HOST, password=PASSWORD)
            self.Output = po.connect_via_input(g_i, phase)
            return self.Output

        if cur_id != tgt_id:
            # Prefer a light rebind (Output handles closing and recreating s_o/g_o)
            try:
                self.Output.set_default_phase(g_i, phase)
            except Exception:
                # Fallback: hard reconnect
                self.close_output_viewer()
                from ..plaxishelper.plaxisoutput import PlaxisOutput
                po = PlaxisOutput(host=HOST, password=PASSWORD)
                self.Output = po.connect_via_input(g_i, phase)

        return self.Output


    def get_results(
        self,
        *,
        structure: object,   # structure object or its plx_id (Output resolves)
        leaf: object,        # resulttypes enum member, or "Plate.UX" string
        phase: object,       # Phase object or its plx_id (Output resolves)
        smoothing: bool = False,
    ):
        """
        One-call result fetch with pre-checks:
        - Ensure the phase was calculated (or raise).
        - Ensure Output is bound to the given phase (create/rebind if needed).
        - Return results from PlaxisOutput.get_results(...).
        """
        # 1) must be calculated first
        self._ensure_phase_is_calculated(phase)

        # 2) ensure Output bound to this phase (creates/rebinds as needed)
        po = self._ensure_output_for_phase(phase)

        # 3) fetch and return
        return po.get_results(structure, leaf, smoothing=smoothing)

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

    # -------- Optional convenience: save using ProjectInformation path ----------

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
