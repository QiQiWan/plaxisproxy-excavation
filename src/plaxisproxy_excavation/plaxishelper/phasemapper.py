# static, tolerant mapper for StageSettings + Phase structure toggles.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, List

# Phase + default settings to construct the initial Phase object
from ..components.phase import Phase
from ..components.phasesettings import PlasticStageSettings, LoadType


# ----------------------------- module helpers -----------------------------
def _is_mapping(x: Any) -> bool:
    """Duck typing for dict-like proxies (Remote Scripting often exposes these)."""
    try:
        return isinstance(x, MutableMapping) or (hasattr(x, "keys") and hasattr(x, "__getitem__"))
    except Exception:
        return False


def set_dotted_attr(root: Any, path: str, value: Any) -> bool:
    """
    Try to set `root.a.b.c = value` for a dotted path.
    Supports attribute objects and dict-like objects. Returns True on success. Never raises.
    """
    try:
        parts = [p for p in str(path).split(".") if p]
        if not parts:
            return False
        obj = root
        for p in parts[:-1]:
            if hasattr(obj, p):
                obj = getattr(obj, p)
                continue
            if _is_mapping(obj) and p in obj:
                obj = obj[p]  # type: ignore[index]
                continue
            return False  # cannot traverse
        leaf = parts[-1]
        if hasattr(obj, leaf):
            try:
                setattr(obj, leaf, value)
                return True
            except Exception:
                pass
        if _is_mapping(obj):
            try:
                obj[leaf] = value  # type: ignore[index]
                return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def merged_candidates(base: Mapping[str, Tuple[str, ...]],
                      overlay: Optional[Mapping[str, Sequence[str]]]) -> Dict[str, Tuple[str, ...]]:
    """Merge the base candidates with an optional overlay (no mutation on inputs)."""
    out: Dict[str, Tuple[str, ...]] = dict(base)
    if overlay:
        for k, v in overlay.items():
            out[k] = tuple(v)
    return out


def default_initial_phase(g_i: Any) -> Optional[Any]:
    """Best-effort handle to find an initial phase in g_i."""
    prev = getattr(g_i, "InitialPhase", None) or getattr(g_i, "initialphase", None) or None
    if prev is not None:
        return prev
    try:
        phases = getattr(g_i, "phases", None) or getattr(g_i, "Phases", None)
        if phases:
            return phases[0]
    except Exception:
        return None
    return None


def resolve_runtime_handle(g_i: Any, obj: Any) -> Optional[Any]:
    """
    Tolerant runtime handle resolver:
    1) Guess a collection by class name plural (Beam -> Beams, Anchor -> Anchors, ...),
       then try name-based lookup (dict-like or iterables with .name).
    2) Fallback to a few common collections (Structures, Beams, Anchors, EmbeddedPiles, Plates, Walls, Wells).
    3) Last resort: try global 'Find'/'ByName' on g_i or g_i.Model.
    """
    name = getattr(obj, "name", None)
    cls_name = type(obj).__name__
    candidates = [
        f"{cls_name}s",                   # Beam -> Beams, Anchor -> Anchors
        f"{cls_name}Set",
        f"{cls_name}Collection",
        "Structures", "Beams", "Anchors", "EmbeddedPiles", "Plates", "Walls", "Wells",
    ]

    def _from_collection(coll: Any, nm: str) -> Optional[Any]:
        if coll is None:
            return None
        if _is_mapping(coll) and nm in coll:
            return coll[nm]  # type: ignore[index]
        try:
            for it in coll:
                if getattr(it, "name", getattr(it, "Name", None)) == nm:
                    return it
        except Exception:
            pass
        if hasattr(coll, nm):
            return getattr(coll, nm)
        return None

    for cname in candidates:
        coll = getattr(g_i, cname, None) or getattr(getattr(g_i, "Model", None), cname, None)
        if coll and name:
            h = _from_collection(coll, name)
            if h is not None:
                return h

    for fn in ("Find", "ByName", "get"):
        finder = getattr(g_i, fn, None) or getattr(getattr(g_i, "Model", None), fn, None)
        if callable(finder) and name:
            try:
                h = finder(name)
                if h is not None:
                    return h
            except Exception:
                pass
    return None


def _handle_name(h: Any, fallback: str = "") -> str:
    """Extract a human-readable name from a PLAXIS handle."""
    try:
        n = getattr(h, "Name", None)
        if n is not None:
            return str(n)
    except Exception:
        pass
    try:
        n = getattr(h, "name", None)
        if n is not None:
            return str(n)
    except Exception:
        pass
    return fallback


# ----------------------------- reporting -----------------------------
@dataclass
class PhaseMapReport:
    assigned: List[Tuple[str, str]] = field(default_factory=list)  # (key, dotted_path) for options
    skipped:  List[str] = field(default_factory=list)              # logical keys w/o candidates
    failed:   List[Tuple[str, str]] = field(default_factory=list)  # (key, dotted_path) attempts that failed
    struct_on:  List[str] = field(default_factory=list)            # names activated
    struct_off: List[str] = field(default_factory=list)            # names deactivated
    water_applied: bool = False

@dataclass
class PhaseUpdateResult:
    handle: Any                          # the (final) phase handle that was updated or recreated
    report: PhaseMapReport               # options + structure toggles report
    recreated: bool = False              # whether we had to recreate the phase
    previous_handle: Optional[Any] = None# old handle if recreated
    prev_changed: bool = False           # whether PreviousPhase was changed (or intended to)

# ----------------------------- static mapper -----------------------------
class PhaseMapper:
    """Pure static utility. Do NOT instantiate."""

    # ----------- options candidates  (aligned to your phasesettings.to_settings_dict) -----------
    CANDIDATES: Dict[str, Tuple[str, ...]] = {
        # General
        "calc_type": ("CalculationType", "Deform.CalculationType"),
        "load_type": ("Loading.LoadingType", "Deform.LoadingType", "Deform.Loading"),
        "pore_cal_type": ("Flow.PoreCalculationType", "Loading.PoreCalculationType", "Loading.PoreCalculation"),
        # ΣM
        "ΣM_stage": ("Deform.SumMStage", "Deform.SigmaMStage", "Deform.SumMstage"),
        "ΣM_weight": ("Deform.SumMWeight", "Deform.SigmaMWeight", "Deform.SumMweight"),
        "SigmaMstage": ("Deform.SumMStage",),
        "SigmaMweight": ("Deform.SumMWeight",),
        # Loading
        "time_interval": ("Loading.TimeInterval", "Deform.TimeInterval"),
        "estimated_end_time": ("Loading.EstimatedEndTime", "Deform.EstimatedEndTime"),
        "first_step": ("Loading.FirstStep", "Deform.FirstStep"),
        "last_step": ("Loading.LastStep", "Deform.LastStep"),
        "special_option": ("Loading.SpecialOption", "Deform.SpecialOption"),
        # Consolidation extras
        "p_stop": ("Loading.MinExcessPorePressure", "Loading.MinimumExcessPorePressure", "Loading.PStop"),
        "degree_of_consolidation": ("Loading.DegreeOfConsolidation", "Loading.SolidationDegree"),
        "solidation_degree": ("Loading.DegreeOfConsolidation", "Loading.SolidationDegree"),
        # Deform
        "ignore_undr_behavior": ("Deform.IgnoreUndrainedBehaviour", "Deform.IgnoreUndrainedBehavior"),
        "force_fully_drained": ("Deform.ForceFullyDrainedNewClusters", "Deform.ForceFullyDrained"),
        "reset_displacemnet": ("Deform.ResetDisplacements",),
        "reset_small_strain": ("Deform.ResetSmallStrains", "Deform.ResetSmallStrain"),
        "reset_state_variable": ("Deform.ResetStateVariables", "Deform.ResetStateVariable"),
        "reset_time": ("Deform.ResetTime",),
        "update_mesh": ("Deform.UpdatedMesh", "Deform.UpdateMesh"),
        "ignore_suction_F": ("Deform.IgnoreSuction",),
        "cavitation_cutoff": ("Deform.CavitationCutOff", "Deform.CavitationCutoff"),
        "cavitation_limit": ("Deform.CavitationStress", "Deform.CavitationLimit"),
        # Numerical
        "solver": ("Numerical.SolverType", "Numerical.Solver"),
        "max_cores_use": ("Numerical.MaxNumberOfCores", "Numerical.MaxCores"),
        "max_number_of_step_store": ("Numerical.MaxNumberOfStepStore",),
        "use_compression_result": ("Numerical.UseCompression", "Numerical.UseCompressionResult"),
        "use_default_iter_param": ("Numerical.UseDefaultIterativeParameters", "Numerical.UseDefaultIterParam"),
        "max_steps": ("Numerical.MaxSteps",),
        "time_step_determination": ("Numerical.TimeStepDetermination",),
        "first_time_step": ("Numerical.FirstTimeStep",),
        "min_time_step": ("Numerical.MinTimeStep",),
        "max_time_step": ("Numerical.MaxTimeStep",),
        "tolerance_error": ("Numerical.Tolerance", "Numerical.ToleranceError"),
        "max_unloading_step": ("Numerical.MaxUnloadingSteps",),
        "max_load_fraction_per_step": ("Numerical.MaxLoadFractionPerStep", "Numerical.MaxLoadFractionStep"),
        "over_relaxation_factor": ("Numerical.Overrelaxation", "Numerical.OverRelaxation", "Numerical.OverrelaxationFactor"),
        "max_iterations": ("Numerical.MaxIterations",),
        "desired_min_iterations": ("Numerical.DesiredMinIterations",),
        "desired_max_iterations": ("Numerical.DesiredMaxIterations",),
        "Arc_length_control": ("Numerical.ArcLengthControl", "Numerical.Arc_length_control"),
        "use_subspace_accelerator": ("Numerical.UseSubspaceAcceleration", "Numerical.SubspaceAccelerator"),
        "subspace_size": ("Numerical.SubspaceSize",),
        "line_search": ("Numerical.UseLineSearch", "Numerical.LineSearch"),
        "use_gradual_error_reduction": ("Numerical.UseGradualErrorReduction", "Numerical.GradualErrorReduction"),
        "number_sub_steps": ("Numerical.NumberOfSubsteps", "Numerical.NumberOfSubSteps"),
        # Dynamics
        "dynamic_time_interval": ("Dynamics.TimeInterval", "Dynamic.TimeInterval", "Deform.TimeInterval"),
        "newmark_alpha": ("Dynamics.NewmarkAlpha", "Deform.NewmarkAlpha"),
        "newmark_beta": ("Dynamics.NewmarkBeta", "Deform.NewmarkBeta"),
        "mass_matrix": ("Dynamics.MassMatrix", "Deform.MassMatrix", "Deform.MassMatrixType"),
        # Flow
        "flow_use_default_iter_param": ("Flow.UseDefaultIterativeParameters", "Flow.UseDefaultIterParam"),
        "flow_max_steps": ("Flow.MaxSteps",),
        "flow_tolerance_error": ("Flow.Tolerance", "Flow.ToleranceError"),
        "flow_over_relaxation_factor": ("Flow.Overrelaxation", "Flow.OverRelaxation"),
        # Safety
        "safety_multiplier": ("Safety.SafetyMultiplier", "Numerical.SafetyMultiplier"),
        "msf": ("Safety.Msf", "Safety.MSF", "Loading.Msf"),
        "sum_msf": ("Safety.TargetSumMsf", "Safety.SumMsf"),
    }

    # ===================== public static API =====================

    # --- navigation helpers ---
    @staticmethod
    def goto_stages(g_i: Any) -> bool:
        """Enter the Stages mode to make sure phase creation API is available."""
        for fn in ("gotostages", "GoToStages", "to_stages"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    f(); return True
                except Exception:
                    pass
        return False

    @staticmethod
    def get_initial_phase(g_i: Any) -> Phase:
        """
        Return a concrete Phase object representing the current project's initial phase.
        - It carries a default PlasticStageSettings (values will be those you define as defaults).
        - `plx_id` on the returned Phase is set to the actual PLAXIS phase handle.
        """
        h = default_initial_phase(g_i)
        # robust name extraction
        nm = "InitialPhase"
        try:
            nm = str(getattr(h, "Name", nm))
        except Exception:
            pass
        st = PlasticStageSettings(load_type=LoadType.StageConstruction)  # default baseline settings
        ph = Phase(name=nm, comment="(initial phase)", settings=st, inherits=None)
        setattr(ph, "plx_id", h)
        return ph

    @staticmethod
    def _set_previous_phase(new_phase_handle: Any, prev_phase_handle: Any) -> bool:
        """
        Best-effort to set inheritance: new.PreviousPhase = prev.
        Return True if succeeded.
        """
        # 1) direct attribute
        try:
            new_phase_handle.PreviousPhase = prev_phase_handle
            return True
        except Exception:
            pass
        # 2) dotted setter fallback
        try:
            return set_dotted_attr(new_phase_handle, "PreviousPhase", prev_phase_handle)
        except Exception:
            return False

    @staticmethod
    def create(g_i: Any, phase_obj: "Phase", inherits: Optional[Any] = None) -> Any:
        """
        Create a new PLAXIS phase for `phase_obj`.
        Inheritance source (base phase) resolution priority:
          1) explicit `inherits` param (if provided),
          2) `phase_obj.inherits.plx_id`.
        Raises:
          ValueError if no base phase can be resolved.
        Side effects:
          - Binds `phase_obj.plx_id` to the created PLAXIS phase handle.
        """
        # 1) resolve base (prefer explicit override)
        base = inherits
        if base is None:
            inh = getattr(phase_obj, "inherits", None)
            if inh is not None:
                base = getattr(inh, "plx_id", None)

        if base is None:
            raise ValueError(
                f"Phase '{getattr(phase_obj, 'name', 'Phase')}' must inherit from a base phase: "
                "provide `phase_obj.inherits` or `inherits=...`."
            )

        # 2) preferred path: empty-new, then set PreviousPhase explicitly
        new_h = None
        for method_name in ("phase", "Phase", "phases.New", "Phases.New"):
            creator = getattr(g_i, method_name, None)
            if not creator:
                continue
            try:
                new_h = creator()  # empty creation
                break
            except Exception:
                continue

        # 3) fallback: create directly with base (older APIs)
        if new_h is None:
            for method_name in ("phase", "Phase", "phases.New", "Phases.New"):
                creator = getattr(g_i, method_name, None)
                if not creator:
                    continue
                try:
                    new_h = creator(base)
                    break
                except Exception:
                    continue

        if new_h is None:
            raise RuntimeError("Failed to create phase in PLAXIS.")

        # 4) enforce inheritance via property (even if fallback already used)
        try:
            PhaseMapper._set_previous_phase(new_h, base)
        except Exception:
            pass

        # 5) write back handle
        setattr(phase_obj, "plx_id", new_h)
        return new_h

    @staticmethod
    def apply_options(
        phase: Any,
        settings: Mapping[str, Any],
        *,
        warn_on_missing: bool = False,
        custom_candidates: Optional[Mapping[str, Sequence[str]]] = None,
        setter: Optional[Callable[[Any, str, Any], bool]] = None,
    ) -> PhaseMapReport:
        """Apply a settings dict (from StageSettingsBase.to_settings_dict()) to a Phase-like object."""
        rep = PhaseMapReport()
        if not settings:
            return rep
        cand = merged_candidates(PhaseMapper.CANDIDATES, custom_candidates)
        _set = setter or set_dotted_attr

        for key, val in settings.items():
            if val is None:
                continue
            paths = cand.get(key)
            if not paths:
                if warn_on_missing:
                    rep.skipped.append(key)
                continue
            assigned = False
            for path in paths:
                ok = _set(phase, path, val)
                if ok:
                    rep.assigned.append((key, path))
                    assigned = True
                    break
                else:
                    rep.failed.append((key, path))
            # Tiny semantic fallback for ΣM stage aliases
            if not assigned and key in ("ΣM_stage", "SigmaMstage"):
                for alt in ("Deform.SumMWeight", "Deform.SigmaMWeight"):
                    if _set(phase, alt, val):
                        rep.assigned.append((key, alt))
                        assigned = True
                        break
        return rep

    @staticmethod
    def apply_structures(
        g_i: Any,
        phase_handle: Any,
        activate: Optional[Sequence[Any]] = None,
        deactivate: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Toggle structures ON/OFF for a given phase.
        This resolver guarantees we pass a valid top-level STRUCTURE HANDLE to PLAXIS:
        - domain object -> obj.plx_id
        - geometry/nested handle -> climb owner to structure
        - pure name string -> resolve handle from model collections
        Call order tries (1) object-first then (2) legacy phase-first, then (3) property fallback.
        """
        def _name_of(h: Any, fallback: str = "unnamed") -> str:
            try:
                return str(getattr(h, "Name", getattr(h, "name", fallback)))
            except Exception:
                return fallback

        def _activate_one(h: Any) -> bool:
            # 1) object-first
            try:
                g_i.activate(h, phase_handle); return True
            except Exception:
                pass
            # 2) legacy order
            try:
                g_i.activate(phase_handle, h); return True
            except Exception:
                pass
            # 3) property fallback
            try:
                h.Active[phase_handle] = True; return True
            except Exception:
                return False

        def _deactivate_one(h: Any) -> bool:
            try:
                g_i.deactivate(h, phase_handle); return True
            except Exception:
                pass
            try:
                g_i.deactivate(phase_handle, h); return True
            except Exception:
                pass
            try:
                h.Active[phase_handle] = False; return True
            except Exception:
                return False

        def _iter_struct_handles(seq: Optional[Sequence[Any]]):
            if not seq:
                return
            for obj in seq:
                # string name → handle
                if isinstance(obj, str):
                    h = PhaseMapper._resolve_handle_by_name(g_i, obj)
                    if h is not None:
                        yield h
                    continue
                # domain or raw → canonical handle
                h, _ = PhaseMapper._canonical_structure_handle(g_i, obj)
                if h is not None:
                    yield h

        turned_on: List[str] = []
        turned_off: List[str] = []

        # Deactivate first, then activate (safer)
        for h in _iter_struct_handles(deactivate):
            if _deactivate_one(h):
                turned_off.append(_name_of(h))

        for h in _iter_struct_handles(activate):
            if _activate_one(h):
                turned_on.append(_name_of(h))

        return turned_on, turned_off

    @staticmethod
    def apply_water_table(phase_handle: Any, water_table_obj: Any) -> bool:
        """
        Apply a single WaterLevelTable to the target phase.
        Tries common property names on Flow/Loading/Model-level ancestors.
        """
        if water_table_obj is None:
            return False
        payload = water_table_obj.to_dict() if hasattr(water_table_obj, "to_dict") else water_table_obj
        paths = (
            "Flow.WaterLevelTable",
            "Flow.WaterTable",
            "Flow.GroundwaterTable",
            "Loading.WaterLevelTable",
            "Model.WaterLevelTable",
            "Model.WaterTable",
            "WaterLevelTable",
            "WaterTable",
        )
        for p in paths:
            if set_dotted_attr(phase_handle, p, payload):
                return True
        return False

    @staticmethod
    def apply_phase(
        g_i: Any,
        phase_handle: Any,
        phase_obj: Any,
        *,
        warn_on_missing: bool = False,
        custom_candidates: Optional[Mapping[str, Sequence[str]]] = None,
        option_setter: Optional[Callable[[Any, str, Any], bool]] = None,
        resolver: Optional[Callable[[Any, Any], Optional[Any]]] = None,  # reserved for future lookups
    ) -> PhaseMapReport:
        """
        Apply both settings and structure toggles (and water table) from a Phase object
        onto an existing PLAXIS phase handle.
        NOTE: creation/inheritance is done separately via `create(...)`.
        """
        # Bind plx_id if not yet bound (so Phase object can be reused across calls)
        if getattr(phase_obj, "plx_id", None) is None:
            setattr(phase_obj, "plx_id", phase_handle)

        # 1) options
        rep = PhaseMapReport()
        opt_payload = getattr(phase_obj, "settings_payload")() if hasattr(phase_obj, "settings_payload") else {}
        opt_rep = PhaseMapper.apply_options(
            phase_handle,
            opt_payload,
            warn_on_missing=warn_on_missing,
            custom_candidates=custom_candidates,
            setter=option_setter,
        )
        rep.assigned.extend(opt_rep.assigned)
        rep.skipped.extend(opt_rep.skipped)
        rep.failed.extend(opt_rep.failed)

        # 2) structures
        act_list = getattr(phase_obj, "activate", []) or []
        deact_list = getattr(phase_obj, "deactivate", []) or []
        on_names, off_names = PhaseMapper.apply_structures(
            g_i, phase_handle, activate=act_list, deactivate=deact_list
        )
        rep.struct_on.extend(on_names)
        rep.struct_off.extend(off_names)

        # 3) water table (single)
        wt = getattr(phase_obj, "water_table", None)
        if wt is not None:
            rep.water_applied = PhaseMapper.apply_water_table(phase_handle, wt)

        return rep

    @staticmethod
    def apply_plan(
        g_i: Any,
        phases: Sequence["Phase"],  # a sequence of Phase domain objects
        *,
        start_from: Optional[Any] = None,     # Phase or handle; defaults to InitialPhase if None
        warn_on_missing: bool = False,
        custom_candidates: Optional[Mapping[str, Sequence[str]]] = None,
        option_setter: Optional[Callable[[Any, str, Any], bool]] = None,
    ) -> List[PhaseMapReport]:
        """
        Create a chain of phases and apply each Phase (options + structures).
        Inheritance priority per phase:
          Phase.inherits.plx_id > last-created handle > start_from > InitialPhase.
        On success:
          - Each Phase gets `plx_id` bound to the created handle.
          - Each report contains options and structure toggles applied on that handle.
        """
        reports: List[PhaseMapReport] = []

        # Resolve starting handle
        prev_handle = getattr(start_from, "plx_id", start_from)
        if prev_handle is None:
            prev_handle = default_initial_phase(g_i)

        for ph in phases:
            # Resolve base/inherits for current phase
            base_handle = None
            inh = getattr(ph, "inherits", None)
            if inh is not None:
                base_handle = getattr(inh, "plx_id", None)
            if base_handle is None:
                base_handle = prev_handle

            # 1) Create phase (empty-new -> set PreviousPhase); this writes ph.plx_id
            new_handle = PhaseMapper.create(g_i, ph, inherits=base_handle)

            # 2) Apply settings + structure toggles (+ water table)
            rep = PhaseMapper.apply_phase(
                g_i,
                new_handle,
                ph,
                warn_on_missing=warn_on_missing,
                custom_candidates=custom_candidates,
                option_setter=option_setter,  # <-- correct param name
            )
            reports.append(rep)

            # 3) Continue the chain
            prev_handle = new_handle

        return reports

    # ===== helpers: pools and name-based resolution =====
    def _enum_structure_pools(g_i: Any):
        """Yield likely structure collections from g_i and g_i.Model."""
        pools = (
            "Structures", "Plates", "Walls", "Beams", "Anchors",
            "EmbeddedPiles", "EmbeddedBeams", "Wells", "Ribs", "Geogrids", "Interfaces",
        )
        model = getattr(g_i, "Model", None)
        for p in pools:
            coll = getattr(g_i, p, None)
            if coll is not None:
                yield coll
            if model is not None:
                coll_m = getattr(model, p, None)
                if coll_m is not None:
                    yield coll_m

    def _lookup_by_name_in_pool(coll: Any, name: str) -> Optional[Any]:
        """Try several access patterns to get an item named `name` out of a PLAXIS collection."""
        if not name or coll is None:
            return None
        # dict-like (index by key)
        try:
            if hasattr(coll, "keys") and name in coll:
                return coll[name]  # type: ignore[index]
        except Exception:
            pass
        # attribute access
        try:
            if hasattr(coll, name):
                return getattr(coll, name)
        except Exception:
            pass
        # linear scan
        try:
            for it in coll:
                nm = None
                try:
                    nm = getattr(it, "Name", None) or getattr(it, "name", None)
                except Exception:
                    pass
                if nm and str(nm) == name:
                    return it
        except Exception:
            pass
        return None

    def _resolve_handle_by_name(g_i: Any, name: str) -> Optional[Any]:
        """Search all known structure pools for an object whose Name equals `name`."""
        for coll in PhaseMapper._enum_structure_pools(g_i):
            h = PhaseMapper._lookup_by_name_in_pool(coll, name)
            if h is not None:
                return h
        # last resort: try global attribute (rare but cheap)
        try:
            if hasattr(g_i, name):
                return getattr(g_i, name)
        except Exception:
            pass
        return None

    def _canonical_structure_handle(g_i: Any, raw: Any) -> Tuple[Optional[Any], str]:
        """
        Normalize domain/geometry/nested handle to a top-level structure handle.
        Returns (handle_or_None, name_guess).
        """
        def _has_active(h: Any) -> bool:
            try:
                _ = h.Active   # probe only
                return True
            except Exception:
                return False

        # prefer an existing handle on domain object
        h = getattr(raw, "plx_id", raw)
        name_guess = ""
        try:
            name_guess = str(getattr(h, "Name", getattr(h, "name", getattr(raw, "name", ""))))
        except Exception:
            pass

        # case A: already a structure
        if _has_active(h):
            return h, name_guess

        # case B: climb likely owners
        cur = h
        for attr in ("Plate", "Beam", "Anchor", "EmbeddedPile", "Parent", "Owner", "Structure"):
            try:
                nxt = getattr(cur, attr, None)
            except Exception:
                nxt = None
            if not nxt:
                continue
            if _has_active(nxt):
                try:
                    name_guess = str(getattr(nxt, "Name", getattr(nxt, "name", name_guess)))
                except Exception:
                    pass
                return nxt, name_guess
            cur = nxt

        # case C: resolve by GUI name
        if name_guess:
            byname = PhaseMapper._resolve_handle_by_name(g_i, name_guess)
            if byname is not None:
                return byname, name_guess

        return None, name_guess or "unnamed"


    @staticmethod
    def update(
        g_i: Any,
        phase_obj: "Phase",
        *,
        warn_on_missing: bool = False,
        custom_candidates: Optional[Mapping[str, Sequence[str]]] = None,
        option_setter: Optional[Callable[[Any, str, Any], bool]] = None,
        allow_recreate: bool = False,
        sync_meta: bool = True,
    ) -> PhaseUpdateResult:
        """
        One-shot update after you modify a Phase object:
        - Ensures the phase exists (create if missing, honoring `phase_obj.inherits`).
        - Tries to align inheritance to `phase_obj.inherits`:
            * if current handle has a different PreviousPhase, try to set it;
            * if not supported and `allow_recreate=True`, recreate the phase and re-apply.
        - Re-applies options + structure toggles (+ water table).
        - Optionally sync phase name/comment to the handle.

        Returns PhaseUpdateResult with details (including whether a recreation happened).
        """
        # Resolve (or create) target handle
        h = getattr(phase_obj, "plx_id", None)
        recreated = False
        prev_changed = False
        old_h = None

        # Desired base
        desired_base = None
        inh = getattr(phase_obj, "inherits", None)
        if inh is not None:
            desired_base = getattr(inh, "plx_id", None)

        # If no handle yet -> create now
        if h is None:
            h = PhaseMapper.create(g_i, phase_obj, inherits=desired_base)
            recreated = True
        else:
            # Try to read current PreviousPhase
            need_change = False
            try:
                current_prev = getattr(h, "PreviousPhase", None)
                # If caller指定了目标基相位，且与当前不同，则尝试切换
                if desired_base is not None and current_prev is not None and current_prev != desired_base:
                    need_change = True
            except Exception:
                # Cannot read current PreviousPhase; if desired_base is provided, try change anyway
                need_change = desired_base is not None

            if need_change:
                prev_changed = True
                # Try to set on the same handle
                if not PhaseMapper._set_previous_phase(h, desired_base):
                    # If not supported & allowed, recreate the phase on the new base
                    if allow_recreate:
                        old_h = h
                        h = PhaseMapper.create(g_i, phase_obj, inherits=desired_base)
                        recreated = True
                    # else: keep going with the old linkage

        # Optionally sync meta (name/comment)
        if sync_meta:
            try:
                PhaseMapper._sync_phase_meta(h, phase_obj)
            except Exception:
                pass

        # Apply options + structure toggles (+ water table)
        rep = PhaseMapper.apply_phase(
            g_i,
            h,
            phase_obj,
            warn_on_missing=warn_on_missing,
            custom_candidates=custom_candidates,
            option_setter=option_setter,
        )

        return PhaseUpdateResult(
            handle=h,
            report=rep,
            recreated=recreated,
            previous_handle=old_h,
            prev_changed=prev_changed,
        )

    @staticmethod
    def _sync_phase_meta(phase_handle: Any, phase_obj: Any) -> Any:
        """
        Best-effort sync of name/comment from domain Phase to PLAXIS handle.
        Ignored silently if properties are read-only in your build.
        """
        nm = getattr(phase_obj, "name", None)
        if nm:
            # (some builds expose Name / Title)
            return set_dotted_attr(phase_handle, "Name", nm) or set_dotted_attr(phase_handle, "Title", nm)

        cm = getattr(phase_obj, "comment", None)
        if cm:
            # (some builds expose Comments / Description)
            return set_dotted_attr(phase_handle, "Comments", cm) or set_dotted_attr(phase_handle, "Description", cm)
