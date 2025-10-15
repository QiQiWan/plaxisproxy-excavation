# phasemapper.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

PhaseLike = Any          # your domain Phase object (has name/comment/settings/activate/deactivate/…)
PhaseHandle = Any        # PLAXIS-side handle for a phase
ModelHandle = Any        # PLAXIS-side handle for any created object


class PhaseMapper:
    """
    A defensive, version-tolerant mapper for PLAXIS Staged Construction phases.

    Public API (static):
      - goto_stages(g_i) -> None
      - get_initial_phase(g_i) -> PhaseHandle
      - find_phase_handle(g_i, name:str) -> Optional[PhaseHandle]
      - create(g_i, *, phase_obj:PhaseLike, inherits:Optional[PhaseHandle]) -> PhaseHandle
      - apply_phase(g_i, phase_handle:PhaseHandle, phase_obj:PhaseLike, warn_on_missing=False) -> None
      - apply_options(phase_handle:PhaseHandle, options:Dict[str,Any], warn_on_missing=False) -> None
      - apply_structures(g_i, phase_handle:PhaseHandle, activate, deactivate, warn_on_missing=False) -> None
      - apply_well_overrides_dict(g_i, phase_handle:PhaseHandle, overrides:Dict[str,Dict[str,Any]], warn_on_missing=False) -> None
      - apply_well_overrides(g_i, phase_handle:PhaseHandle, phase_like:PhaseLike, warn_on_missing=False) -> None
      - apply_plan(g_i, phases:Sequence[PhaseLike], warn_on_missing=False) -> List[PhaseHandle]
      - update(g_i, phase_obj:PhaseLike, base:Optional[PhaseHandle], warn_on_missing=False) -> Dict[str,Any]
    """

    # ---------------------------------------------------------------------
    # Stage navigation & discovery
    # ---------------------------------------------------------------------

    @staticmethod
    def goto_stages(g_i: Any) -> None:
        """Switch PLAXIS UI to Staged Construction; try common variants."""
        for fn in ("gotostages", "GoToStages", "goto_stages"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    f()
                    return
                except Exception:
                    pass
        # Some builds implicitly switch; do not hard-fail.

    @staticmethod
    def get_initial_phase(g_i: Any) -> PhaseHandle:
        """
        Return the Initial Phase handle (robust).
        Tries: g_i.InitialPhase, first of g_i.Phases, or search by Identification.
        """
        # 1) direct property
        initial = getattr(g_i, "InitialPhase", None)
        if initial is not None:
            return initial

        # 2) first in phases collection
        try:
            phases = getattr(g_i, "Phases", None)
            if phases:
                try:
                    return phases[0]  # indexing
                except Exception:
                    for p in phases:
                        return p
        except Exception:
            pass

        # 3) search by name
        try:
            for p in g_i.Phases[:]:
                name = getattr(p, "Identification", "") or getattr(p, "Name", "")
                if name in ("InitialPhase", "Initial Phase", "Initial phase"):
                    return p
        except Exception:
            pass

        raise RuntimeError("Initial phase handle not found.")

    @staticmethod
    def find_phase_handle(g_i: Any, name: str) -> Optional[PhaseHandle]:
        """Find a phase by Identification/Name."""
        if not name:
            return None
        try:
            col = getattr(g_i, "Phases", None)
            if not col:
                return None
            # Try slicing first (faster in most builds)
            try:
                for p in col[:]:
                    ident = getattr(p, "Identification", None) or getattr(p, "Name", None)
                    if ident == name:
                        return p
            except Exception:
                for p in col:
                    ident = getattr(p, "Identification", None) or getattr(p, "Name", None)
                    if ident == name:
                        return p
        except Exception:
            pass
        return None

    # ---------------------------------------------------------------------
    # Low-level creation helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _create_phase_from_base(g_i: Any, previous: PhaseHandle) -> PhaseHandle:
        """
        Create a new phase that inherits from 'previous'. Try several APIs:
          1) g_i.phase(previous)
          2) g_i.addphase(previous)
          3) g_i.Phase(previous)
          4) g_i.phase() then set PreviousPhase attribute
        """
        for fn in ("phase", "addphase", "Phase"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    return f(previous)
                except Exception:
                    pass

        # Empty create then point 'PreviousPhase' to previous
        for fn in ("phase", "addphase", "Phase"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    new_ph = f()
                    # set inheritance link
                    for attr in ("PreviousPhase", "Parent", "ParentPhase"):
                        try:
                            setattr(new_ph, attr, previous)
                            break
                        except Exception:
                            continue
                    return new_ph
                except Exception:
                    pass

        raise RuntimeError("No usable phase creation API found (phase/addphase/Phase).")

    @staticmethod
    def _maybe_set_identification(ph: PhaseHandle, name: Optional[str]) -> None:
        if not name:
            return
        for attr in ("Identification", "Name"):
            try:
                setattr(ph, attr, str(name))
                return
            except Exception:
                continue

    @staticmethod
    def _maybe_set_comment(ph: PhaseHandle, comment: Optional[str]) -> None:
        if not comment:
            return
        for attr in ("Comments", "Comment"):
            try:
                setattr(ph, attr, str(comment))
                return
            except Exception:
                continue

    # ---------------------------------------------------------------------
    # Public: create a PLAXIS phase from a Phase object
    # ---------------------------------------------------------------------

    @staticmethod
    def create(g_i: Any, *, phase_obj: PhaseLike, inherits: Optional[PhaseHandle]) -> PhaseHandle:
        """
        Create a new PLAXIS phase:
          - inherits from 'inherits' or Initial Phase if None
          - sets Identification/Comment if available
          - writes the new handle back to phase_obj.plx_id (if present)
        """
        base = inherits or PhaseMapper.get_initial_phase(g_i)
        new_ph = PhaseMapper._create_phase_from_base(g_i, base)

        PhaseMapper._maybe_set_identification(new_ph, getattr(phase_obj, "name", None))
        PhaseMapper._maybe_set_comment(new_ph, getattr(phase_obj, "comment", None))

        # bind back
        try:
            setattr(phase_obj, "plx_id", new_ph)
        except Exception:
            pass

        return new_ph

    # ---------------------------------------------------------------------
    # Options application
    # ---------------------------------------------------------------------

    @staticmethod
    def _set_phase_attr(phase: PhaseHandle, key: str, value: Any) -> bool:
        """
        Try to set phase attribute by key, with a few alias tricks:
          - replace Greek 'Σ' with 'Sum' / 'Sigma'
          - support dotted keys like 'Deform.ResetDisplacements'
        """
        # dot path (e.g., "Deform.ResetDisplacements")
        if "." in key:
            cur = phase
            parts = key.split(".")
            try:
                for p in parts[:-1]:
                    cur = getattr(cur, p)
                setattr(cur, parts[-1], value)
                return True
            except Exception:
                return False

        # direct attr or greek alias
        candidates = [key, key.replace("Σ", "Sum"), key.replace("Σ", "Sigma")]
        for k in candidates:
            try:
                setattr(phase, k, value)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def apply_options(phase_handle: PhaseHandle, options: Dict[str, Any],
                      *, warn_on_missing: bool = False) -> None:
        """
        Apply solver/stage options in a tolerant way.
        Accepts flat dict or nested dict; dotted keys are allowed.
        """
        if not isinstance(options, dict):
            return

        # flatten nested dicts with dotted paths
        def _flatten(prefix: str, d: Dict[str, Any], out: Dict[str, Any]) -> None:
            for k, v in d.items():
                kk = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(kk, v, out)
                else:
                    out[kk] = v

        flat: Dict[str, Any] = {}
        _flatten("", options, flat)

        for key, val in flat.items():
            if not PhaseMapper._set_phase_attr(phase_handle, key, val) and warn_on_missing:
                print(f"[PhaseMapper.apply_options] Unknown or unsupported option '{key}' (ignored).")

    # ---------------------------------------------------------------------
    # Structure activation/deactivation
    # ---------------------------------------------------------------------

    @staticmethod
    def _resolve_handle(obj: Any) -> Optional[ModelHandle]:
        """
        Resolve a PLAXIS handle from a domain object or a raw handle.
        Tries .plx_id first; otherwise returns obj if it already looks like a handle.
        """
        if obj is None:
            return None
        h = getattr(obj, "plx_id", None)
        return h if h is not None else obj

    @staticmethod
    def _activate(g_i: Any, handle: ModelHandle, phase: PhaseHandle) -> bool:
        # preferred API
        for fn in ("activate", "Activate"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    f(handle, phase=phase)
                    return True
                except Exception:
                    pass
        # attribute toggle fallback
        try:
            # many objects expose Active[phase] indexing
            active = getattr(handle, "Active", None)
            if active is not None:
                active[phase] = True
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _deactivate(g_i: Any, handle: ModelHandle, phase: PhaseHandle) -> bool:
        for fn in ("deactivate", "Deactivate"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    f(handle, phase=phase)
                    return True
                except Exception:
                    pass
        try:
            inactive = getattr(handle, "Active", None)
            if inactive is not None:
                inactive[phase] = False
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def apply_structures(g_i: Any, phase_handle: PhaseHandle,
                         activate: Iterable[Any], deactivate: Iterable[Any],
                         *, warn_on_missing: bool = False) -> None:
        """Activate/deactivate model objects for this phase."""
        # activate
        for obj in (activate or []):
            h = PhaseMapper._resolve_handle(obj)
            if h is None:
                if warn_on_missing:
                    print("[PhaseMapper.apply_structures] activate: missing handle; skipped.")
                continue
            if not PhaseMapper._activate(g_i, h, phase_handle) and warn_on_missing:
                print(f"[PhaseMapper.apply_structures] activate failed for '{getattr(obj,'name',obj)}'.")

        # deactivate
        for obj in (deactivate or []):
            h = PhaseMapper._resolve_handle(obj)
            if h is None:
                if warn_on_missing:
                    print("[PhaseMapper.apply_structures] deactivate: missing handle; skipped.")
                continue
            if not PhaseMapper._deactivate(g_i, h, phase_handle) and warn_on_missing:
                print(f"[PhaseMapper.apply_structures] deactivate failed for '{getattr(obj,'name',obj)}'.")

    # ---------------------------------------------------------------------
    # Water table (best-effort)
    # ---------------------------------------------------------------------

    @staticmethod
    def _extract_head_from_mapping(mp: Dict[str, Any]) -> Optional[float]:
        for k in ("head", "level", "z", "H", "h"):
            if k in mp:
                try:
                    return float(mp[k])
                except Exception:
                    return None
        return None

    @staticmethod
    def _extract_head(water_tbl: Any) -> Optional[float]:
        """
        Return a scalar head (z) when possible.
        Accepts WaterTable/WaterLevelTable objects, dict-like payloads, or a plain number.
        """
        # direct numeric
        if isinstance(water_tbl, (int, float)):
            try:
                return float(water_tbl)
            except Exception:
                return None

        # mapping/dict
        if isinstance(water_tbl, dict):
            return PhaseMapper._extract_head_from_mapping(water_tbl)

        # object with to_dict / to_payload
        for meth in ("to_payload", "to_dict", "as_dict"):
            fn = getattr(water_tbl, meth, None)
            if callable(fn):
                try:
                    mp = fn() or {}
                    if isinstance(mp, dict):
                        h = PhaseMapper._extract_head_from_mapping(mp)
                        if h is not None:
                            return h
                except Exception:
                    pass

        # object attributes (head / level / z / H / h)
        for k in ("head", "level", "z", "H", "h"):
            try:
                v = getattr(water_tbl, k, None)
                if v is not None:
                    return float(v)
            except Exception:
                continue

        return None

    @staticmethod
    def _apply_water_table(g_i: Any, phase_handle: Any, water_tbl: Any,
                           *, warn_on_missing: bool = False) -> None:
        """
        Best-effort application:
        1) If scalar head is extractable -> set as a phase property.
        2) Otherwise, try to pass the object to a solver API if present.
        """
        head = PhaseMapper._extract_head(water_tbl)
        if head is not None:
            # Try common phase attributes first
            for attr in ("WaterLevel", "PhreaticLevel", "WaterLevelHead", "Head"):
                try:
                    setattr(phase_handle, attr, head)
                    return
                except Exception:
                    pass
            # Or solver helpers with phase kw
            for fn_name in ("setwaterlevel", "SetWaterLevel", "set_water_level"):
                fn = getattr(g_i, fn_name, None)
                if callable(fn):
                    try:
                        fn(head, phase=phase_handle)
                        return
                    except Exception:
                        pass
            if warn_on_missing:
                print("[PhaseMapper] Could not set scalar head on phase; skipped.")
            return

        # If head not extractable, try to pass the whole object to solver-level API
        for fn_name in ("setwaterlevelobj", "SetWaterLevelObject", "set_water_level_object"):
            fn = getattr(g_i, fn_name, None)
            if callable(fn):
                try:
                    fn(water_tbl, phase=phase_handle)
                    return
                except Exception:
                    pass

        if warn_on_missing:
            print("[PhaseMapper] No supported API found to apply water table object; skipped.")

    # ---------------------------------------------------------------------
    # Wells: name-based overrides per phase
    # ---------------------------------------------------------------------

    @staticmethod
    def _iter_candidate_collections(g_i: Any) -> List[Any]:
        """
        Return possible containers that may hold well-like objects.
        We search by Identification across these.
        """
        names = [
            "Wells", "Pipes", "LineConstructs", "Lines",
            "EmbeddedBeams", "Beams", "Plates", "Walls",
            "Anchors", "Structures", "Objects"
        ]
        cols: List[Any] = []
        for n in names:
            try:
                c = getattr(g_i, n, None)
                if c:
                    try:
                        _ = iter(c)
                        cols.append(c)
                    except Exception:
                        pass
            except Exception:
                pass
        return cols

    @staticmethod
    def _find_model_object_by_name(g_i: Any, name: str) -> Optional[ModelHandle]:
        if not name:
            return None
        for col in PhaseMapper._iter_candidate_collections(g_i):
            # fast slice
            try:
                for obj in col[:]:
                    if getattr(obj, "Identification", None) == name:
                        return obj
            except Exception:
                try:
                    for obj in col:
                        if getattr(obj, "Identification", None) == name:
                            return obj
                except Exception:
                    continue
        return None

    @staticmethod
    def _set_object_param(g_i: Any, handle: ModelHandle, key: str, value: Any,
                          phase: Optional[PhaseHandle]) -> bool:
        """
        Try to set a parameter on a model object, optionally for a phase.
        """
        # 1) setattr directly on the object
        try:
            setattr(handle, key, value)
            return True
        except Exception:
            pass

        # 2) g_i.SetParameter(handle, key, value, phase=phase)
        for fn in ("setparameter", "SetParameter", "set"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    if phase is not None:
                        f(handle, key, value, phase=phase)
                    else:
                        f(handle, key, value)
                    return True
                except Exception:
                    continue

        return False

    @staticmethod
    def apply_well_overrides_dict(g_i: Any, phase_handle: PhaseHandle,
                                  overrides: Dict[str, Dict[str, Any]],
                                  *, warn_on_missing: bool = False) -> None:
        """
        Apply well parameter overrides per phase using a simple mapping:
            { "Well-1": {"q_well": 0.008, "h_min": -10.0, "well_type": "Extraction"} }
        Notes:
        - Discharge is assumed evenly distributed along well length by PLAXIS.
        - If wells intersect other geometry, solver warnings may appear (modeling issue).
        """
        if not isinstance(overrides, dict):
            return

        # canonical key map
        alias_map = {
            "q_well": ("q_well", "QWell", "qwell", "Discharge", "Q", "Qwell"),
            "h_min": ("h_min", "HMin", "HeadMin"),
            "well_type": ("well_type", "WellType", "Mode", "Type"),
        }

        for well_name, params in overrides.items():
            handle = PhaseMapper._find_model_object_by_name(g_i, well_name)
            if handle is None:
                if warn_on_missing:
                    print(f"[PhaseMapper.well_overrides] Well '{well_name}' not found; skipped.")
                continue

            for canonical, candidates in alias_map.items():
                # if canonical or any alias is present, set canonical key
                value_present = None
                if canonical in params:
                    value_present = params[canonical]
                else:
                    for a in candidates:
                        if a in params:
                            value_present = params[a]
                            break
                if value_present is None:
                    continue

                # try all alias keys on the handle
                applied = False
                for key_try in candidates:
                    if PhaseMapper._set_object_param(g_i, handle, key_try, value_present, phase_handle):
                        applied = True
                        break
                if not applied:
                    # final attempt with canonical name
                    PhaseMapper._set_object_param(g_i, handle, canonical, value_present, phase_handle)

    @staticmethod
    def apply_well_overrides(g_i: Any, phase_handle: PhaseHandle, phase_like: PhaseLike,
                             *, warn_on_missing: bool = False) -> None:
        """Compatibility wrapper expecting an object with `.well_overrides`."""
        mapping = getattr(phase_like, "well_overrides", None)
        if isinstance(mapping, dict):
            PhaseMapper.apply_well_overrides_dict(g_i, phase_handle, mapping, warn_on_missing=warn_on_missing)

    # ---------------------------------------------------------------------
    # High-level "apply phase"
    # ---------------------------------------------------------------------

    @staticmethod
    def apply_phase(g_i: Any, phase_handle: PhaseHandle, phase_obj: PhaseLike,
                    *, warn_on_missing: bool = False) -> None:
        """
        Apply one phase in four steps:
          1) options/settings
          2) water table (optional)
          3) structure activation/deactivation
          4) well overrides (optional)
        """
        # bind back (useful for later updates)
        try:
            setattr(phase_obj, "plx_id", phase_handle)
        except Exception:
            pass

        # 1) options
        options = {}
        if hasattr(phase_obj, "settings_payload") and callable(getattr(phase_obj, "settings_payload")):
            try:
                options = phase_obj.settings_payload() or {}
            except Exception:
                options = {}
        elif hasattr(phase_obj, "settings"):
            s = getattr(phase_obj, "settings")
            if hasattr(s, "to_dict"):
                try:
                    options = s.to_dict() or {}
                except Exception:
                    options = {}
            else:
                options = s if isinstance(s, dict) else {}

        PhaseMapper.apply_options(phase_handle, options, warn_on_missing=warn_on_missing)

        # 2) water table
        wt = getattr(phase_obj, "water_table", None)
        PhaseMapper._apply_water_table(g_i, phase_handle, wt, warn_on_missing=warn_on_missing)

        # 3) structures
        PhaseMapper.apply_structures(
            g_i,
            phase_handle,
            getattr(phase_obj, "activate", []) or [],
            getattr(phase_obj, "deactivate", []) or [],
            warn_on_missing=warn_on_missing,
        )

        # 4) well overrides
        PhaseMapper.apply_well_overrides(g_i, phase_handle, phase_obj, warn_on_missing=warn_on_missing)

    # ---------------------------------------------------------------------
    # Batch helpers (quality-of-life)
    # ---------------------------------------------------------------------

    @staticmethod
    def apply_plan(g_i: Any, phases: Sequence[PhaseLike], *,
                   warn_on_missing: bool = False) -> List[PhaseHandle]:
        """
        Create and apply a chain of phases in order (each inherits from previous).
        Returns the created phase handles (in order).
        """
        if not phases:
            return []

        PhaseMapper.goto_stages(g_i)
        previous = PhaseMapper.get_initial_phase(g_i)

        handles: List[PhaseHandle] = []
        for ph in phases:
            h = PhaseMapper.create(g_i, phase_obj=ph, inherits=previous)
            PhaseMapper.apply_phase(g_i, h, ph, warn_on_missing=warn_on_missing)
            handles.append(h)
            previous = h
        return handles

    @staticmethod
    def update(g_i: Any, phase_obj: PhaseLike, base: Optional[PhaseHandle] = None,
               *, warn_on_missing: bool = False) -> Dict[str, Any]:
        """
        Idempotent update:
          - If phase_obj has handle: apply on it.
          - Else create inheriting from `base` (or Initial), then apply.
        """
        report = {"created": False, "applied": False, "handle": None}

        h = getattr(phase_obj, "plx_id", None)
        if h is None:
            base = base or PhaseMapper.get_initial_phase(g_i)
            h = PhaseMapper.create(g_i, phase_obj=phase_obj, inherits=base)
            report["created"] = True

        try:
            PhaseMapper.apply_phase(g_i, h, phase_obj, warn_on_missing=warn_on_missing)
            report["applied"] = True
        finally:
            report["handle"] = h
        return report
