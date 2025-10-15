from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from math import ceil
from typing import List, Dict, Any, Optional

from .plaxisproxy_excavation.plaxishelper.plaxisrunner import PlaxisRunner
from .plaxis_config import *
from .plaxisproxy_excavation.excavation import FoundationPit


"""
Excavation Engineering Automation — Builder
-------------------------------------------
This builder orchestrates mapping the FoundationPit model into PLAXIS
via a PlaxisRunner adapter. It builds only the *initial design*; phase
options/activations and per-phase water/well overrides are applied later.
"""


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
            for k in ("retaining_walls", "anchors", "beams", "wells", "embedded_piles"):
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

        walls = len(pit.structures.get("retaining_walls", [])) if isinstance(pit.structures, dict) else 0
        print(
            f"[check_completeness] OK: "
            f"name='{getattr(pit, 'name', '<unnamed>')}', "
            f"boreholes={bh_count}, walls={walls}, phases={len(pit.phases)}."
        )

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
        for blk in structures.get("soil_blocks", []) or []:
            try:
                app.create_soil_block(blk)
            except Exception as e:
                print(f"[build] Warning: create_soil_block failed: {e}")

        # ---------------- 5) Structures (including wells) ----------------------
        for wall in structures.get("retaining_walls", []) or []:
            try:
                app.create_retaining_wall(wall)
            except Exception as e:
                print(f"[build] Warning: create_retaining_wall failed: {e}")

        for beam in structures.get("beams", []) or []:
            try:
                app.create_beam(beam)
            except Exception as e:
                print(f"[build] Warning: create_beam failed: {e}")

        for anc in structures.get("anchors", []) or []:
            try:
                app.create_anchor(anc)
            except Exception as e:
                print(f"[build] Warning: create_anchor failed: {e}")

        for pile in structures.get("embedded_piles", []) or []:
            try:
                app.create_embedded_pile(pile)
            except Exception as e:
                print(f"[build] Warning: create_embedded_pile failed: {e}")

        # Wells — created once at structure stage (geometry-level only)
        for well in structures.get("wells", []) or []:
            try:
                app.create_well(well)
            except Exception as e:
                print(f"[build] Warning: create_well failed: {e}")

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
                    from .plaxisproxy_excavation.plaxishelper.monitormapper import MonitorMapper  # type: ignore
                except Exception:
                    from plaxisproxy_excavation.plaxishelper.monitormapper import MonitorMapper  # type: ignore
                MonitorMapper.create_monitors(app.g_i, monitors)  # type: ignore
            except Exception as e:
                print(f"[build] Warning: monitor mapping skipped: {e}")

        # ---------------- 8) Mesh (optional) -----------------------------------
        meshed = False
        mesh_cfg = getattr(pit, "mesh", None)
        if mesh_cfg is not None:
            try:
                app.mesh(mesh_cfg)
                meshed = True
            except Exception as e:
                print(f"[build] Warning: mesh() failed: {e}")
        else:
            print("[build] Info: no mesh config on FoundationPit; skipping meshing step.")

        # ---------------- 9) Phase shells ONLY (no apply) ----------------------
        phases = list(getattr(pit, "phases", [])) or getattr(pit, "stages", []) or []
        created_phase_handles = []
        if phases:
            try:
                app.goto_stages()
                prev = app.get_initial_phase()
                for ph in phases:
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
