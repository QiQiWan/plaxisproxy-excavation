from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from ..components import *
from ..structures.basestructure import BaseStructure
from ..components.phasesettings import *
from .watertablemapper import WaterTableMapper
from collections import defaultdict
import re


FAMILY_REGISTRY: list[dict] = [
    {
        "tag": "plate",
        "regex": r"plate",                         # Plate/WALL 等本质上都是 Plate
        "containers": ["Plates", "Walls"],
        "child_attrs": ("Children", "Members", "SubObjects", "Items"),
        "suffix_regex": r"(?:_\d+)?",
    },
    {
        "tag": "n2n_anchor",
        "regex": r"(node.?to.?node.*anchor|n2n.*anchor|anchor)",
        "containers": ["NodeToNodeAnchors", "NodetoNodeAnchors", "Anchors", "Node_to_node_anchors"],
        "child_attrs": ("Children", "Members", "Items", "SubObjects"),
        "suffix_regex": r"(?:_\d+)?",
    },
    {
        "tag": "beam",
        "regex": r"\bbeam\b",
        "containers": ["Beams", "EmbeddedBeams"],
        "child_attrs": (),
        "suffix_regex": r"",
    },
    {
        "tag": "embedded_pile",
        "regex": r"(embedded.*(pile|beam))",
        "containers": ["EmbeddedBeams", "EmbeddedPiles"],
        "child_attrs": (),
        "suffix_regex": r"",
    },
    {
        "tag": "well",
        "regex": r"\bwell\b",
        "containers": ["Wells", "Pipes"],
        "child_attrs": (),
        "suffix_regex": r"",
    },
    {
        "tag": "soil",
        "regex": r"(soil|soilvolume|soilbody)",
        "containers": ["SoilVolumes", "Soils"],
        "child_attrs": (),
        "suffix_regex": r"",
    },
]

# Fast mapping for domain object class names
CLASS_TAGS = {
    "RetainingWall": "plate",
    "Beam": "beam",
    "EmbeddedPile": "embedded_pile",
    "Anchor": "n2n_anchor",
    "Well": "well",
    "SoilBlock": "soil",
}

# Internal alias for PLAXIS phase handle (only used in private helpers)
PhaseHandle = Any

# ### heplers: pick up objects by names ###

def _ident_str(x) -> str:
    if x is None: return ""
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)

def _canon(name: str) -> str:
    """normalize name: keep [A-Za-z0-9_], collapse _, strip edges, case-insensitive elsewhere."""
    s = str(name or "")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)   # non-word -> _
    s = re.sub(r"_+", "_", s)               # collapse __ -> _
    return s.strip("_")

def _family_base_strict(name: str) -> str:
    """
    Strict base: remove ONLY trailing '_<digits>' groups (if any).
      'A1'        -> 'A1'
      'A1_1'      -> 'A1'
      'PLATE_12'  -> 'PLATE'
      'WALL_2_3'  -> 'WALL_2'   (只去掉最后一个 _3；若要多级去掉，使用 *_loose)
    为了简洁与健壮，我们直接“一次性去掉所有结尾的 _数字段”：
      'WALL_2_3'  -> 'WALL' （更符合“同一族”的直觉）
    """
    c = _canon(name)
    return re.sub(r'(?:_\d+)+$', '', c)

def _family_base_loose(name: str) -> str:
    """
    Loose base: remove trailing '_<digits>' OR bare '<digits>' at end.
      'PLATE1'    -> 'PLATE'
      'A1'        -> 'A'     （仅作为兜底！）
    """
    c = _canon(name)
    return re.sub(r'(?:_?\d+)+$', '', c)

def _family_regex(base: str) -> re.Pattern:
    """
    Accept family object itself and any number of trailing '_<digits>' chunks.
      BASE, BASE_1, BASE_1_2, ...
    """
    b = _canon(base)
    # 允许 0 次或多次后缀；IGNORECASE
    return re.compile(rf'^{re.escape(b)}(?:_\d+)*$', re.IGNORECASE)

def _family_regexes_for_name(name: str) -> list[re.Pattern]:
    """
    Build a list of candidate regexes for the given name:
      1) strict:  base_strict + (?:_\\d+)*
      2) loose :  base_loose  + (?:_?\\d+)*   (fallback only)
    """
    base1 = _family_base_strict(name)
    base2 = _family_base_loose(name)
    pats = []
    if base1:
        pats.append(_family_regex(base1))                # ^BASE1(?:_\d+)*$
    if base2 and base2 != base1:
        # 宽松匹配允许无下划线的数字段：BASE2, BASE2_1, BASE2_12, BASE2_12_3, BASE2 1 -> 已在 _canon 中变成 BASE2_1
        pats.append(re.compile(rf'^{re.escape(_canon(base2))}(?:_?\d+)*$', re.IGNORECASE))
    return pats or [re.compile(r'^$', re.IGNORECASE)]

def _try_iter(col):
    try: return col[:]
    except Exception: return col

def _get_typename(o: Any) -> str:
    for k in ("TypeName", "typename", "type_name"):
        try:
            v = getattr(o, k, None)
            if v:
                return _ident_str(v).lower()
        except Exception:
            pass
    # Fallback: 类名或 repr 线索
    try:
        return (o.__class__.__name__ or "").lower()
    except Exception:
        return str(type(o)).lower()

def _match_rule_by_typename(typename: str) -> Optional[dict]:
    t = (typename or "").lower()
    for rule in FAMILY_REGISTRY:
        if re.search(rule["regex"], t):
            return rule
    return None

def _guess_type_tag(obj_or_handle: Any) -> Optional[str]:
    # If the handle can be obtained, directly determine it with TypeName.
    h = getattr(obj_or_handle, "plx_id", None) or obj_or_handle
    if h is not None:
        rule = _match_rule_by_typename(_get_typename(h))
        if rule:
            return rule["tag"]

    try:
        tag = CLASS_TAGS.get(obj_or_handle.__class__.__name__)
        if tag:
            return tag
    except Exception:
        pass
    return None

def _rule_by_tag(tag: Optional[str]) -> Optional[dict]:
    if not tag: 
        return None
    for r in FAMILY_REGISTRY:
        if r["tag"] == tag:
            return r
    return None

def _get_first_collection(g_i: Any, names: Sequence[str]) -> Optional[Any]:
    for n in names:
        try:
            c = getattr(g_i, n, None)
            if c:
                return c
        except Exception:
            pass
    return None

# ### helpers: flatten nested dicts###
def _flatten_options_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    def _rec(prefix: str, obj: Dict[str, Any]) -> None:
        for k, v in obj.items():
            kk = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _rec(kk, v)
            else:
                flat[kk] = v
    _rec("", d or {})
    return flat

# ### helper: map the phase settings ###
# Map normalized user keys -> exact PLAXIS property path "Phase|Deform|Flow|Dynamics|GroundwaterFlow|Deformations.<Prop>"
# Normalization rule: lower-case, remove spaces/dashes/underscores.
def _norm(s: Any) -> str:
    """Lowercase and strip spaces/underscores/dashes for robust key matching."""
    return str(s).strip().lower().replace("_", "").replace("-", "").replace(" ", "")

def normalize_key(key: str) -> str:
    return _norm(key)

def resolve_path(raw_key: str, alias_map: Dict[str, str]) -> str | None:
    """
    Accept either a friendly key ('ignore_undrained') or a dotted path
    ('Deform.IgnoreUndrainedBehaviour'). Returns the canonical PLAXIS path.
    """
    if "." in raw_key:
        # Align US/UK spelling just in case
        return raw_key.replace("Behavior", "Behaviour")
    return alias_map.get(normalize_key(raw_key))

def coerce_value(path: str, val: Any, enums: Dict[str, Dict[str, str]]) -> Any:
    """
    Convert user inputs into PLAXIS-friendly values:
    - booleans: accept 'true/false', 'on/off', 'yes/no', '1/0'
    - enums: normalized via enums[path]
    """
    if isinstance(val, str):
        lv = _norm(val)
        if lv in {"true", "yes", "on", "1"}:
            return True
        if lv in {"false", "no", "off", "0"}:
            return False

    table = enums.get(path)
    if table:
        key = _norm(val)
        if key in table:
            return table[key]
        return val  # already in correct literal
    return val

def split_target(path: str) -> Tuple[str | None, str]:
    """
    'Deform.MaxUnloadingSteps' -> ('Deform', 'MaxUnloadingSteps')
    'TimeInterval' -> (None, 'TimeInterval')
    """
    parts = path.split(".")
    return (None, parts[0]) if len(parts) == 1 else (parts[0], parts[1])

# ########## Canonical key maps (normalized_key -> 'Phase or Sub.Property') ##########
PHASE_MAP = {
    "timeinterval": "TimeInterval",
    "maxstepsstored": "MaxStepsStored",
    "maxcores": "MaxCores",
    "shouldcalculate": "ShouldCalculate",
    "deformcalctype": "DeformCalcType",
    "porecaltype": "PorePresCalcType",
    "estimatedendtime": "EstimatedEndTime",
}

DEFORM_MAP = {
    "usedefaultiterationparams": "Deform.UseDefaultIterationParams",
    "overrelaxation": "Deform.OverRelaxation",
    "arclengthcontrol": "Deform.ArcLengthControl",
    "uselinesearch": "Deform.UseLineSearch",
    "maxiterations": "Deform.MaxIterations",
    "desiredmaxiterations": "Deform.DesiredMaxIterations",
    "desiredminiterations": "Deform.DesiredMinIterations",
    "maxunloadingsteps": "Deform.MaxUnloadingSteps",
    "toleratederror": "Deform.ToleratedError",
    "usegradualerror": "Deform.UseGradualError",
    "timestepdetermtype": "Deform.TimeStepDetermType",
    "timesteptype": "Deform.TimeStepDetermType",  # alias
    "firsttimestep": "Deform.FirstTimeStep",
    "mintimestep": "Deform.MinTimeStep",
    "maxtimestep": "Deform.MaxTimeStep",
    "maxsteps": "Deform.MaxSteps",
    "maxloadfractionperstep": "Deform.MaxLoadFractionPerStep",
    "ignoreundrained": "Deform.IgnoreUndrainedBehaviour",
    "ignoreundrainedbehaviour": "Deform.IgnoreUndrainedBehaviour",
    "ignoreundrainedbehavior": "Deform.IgnoreUndrainedBehaviour",
    "ignoresuction": "Deform.IgnoreSuction",
    "useupdatedmesh": "Deform.UseUpdatedMesh",
    "useupdatedwaterpressures": "Deform.UseUpdatedWaterPressures",
    "usecavitationcutoff": "Deform.UseCavitationCutOff",
    "cavitationstress": "Deform.CavitationStress",
    "summweight": "Deform.SumMweight",
    "resetdisplacementstozero": "Deform.ResetDisplacementsToZero",
    "resetsmallstrain": "Deform.ResetSmallStrain",
    "resettime": "Deform.ResetTime",
    "resetstatevariables": "Deform.ResetStateVariables",
    "newmarkalpha": "Deform.NewmarkAlpha",
    "newmarkdelta": "Deform.NewmarkDelta",
    "loadingtype": "Deform.LoadingType",
}

FLOW_MAP = {
    "usedefaultiterationparams": "Flow.UseDefaultIterationParams",
    "overrelaxation": "Flow.OverRelaxation",
    "maxiterations": "Flow.MaxIterations",
    "desiredmaxiterations": "Flow.DesiredMaxIterations",
    "desiredminiterations": "Flow.DesiredMinIterations",
    "toleratederror": "Flow.ToleratedError",
    "firsttimestep": "Flow.FirstTimeStep",
    "mintimestep": "Flow.MinTimeStep",
    "maxtimestep": "Flow.MaxTimeStep",
    "maxsteps": "Flow.MaxSteps",
    "thermalcalctype": "Flow.ThermalCalcType",
}

DYNAMICS_MAP = {
    "boundaryxmin": "Dynamics.BoundaryXMin",
    "boundaryxmax": "Dynamics.BoundaryXMax",
    "boundaryymin": "Dynamics.BoundaryYMin",
    "boundaryymax": "Dynamics.BoundaryYMax",
    "boundaryzmin": "Dynamics.BoundaryZMin",
    "boundaryzmax": "Dynamics.BoundaryZMax",
    "normalrelaxcoeffc1": "Dynamics.NormalRelaxCoeffC1",
    "tangentialrelaxcoeffc2": "Dynamics.TangentialRelaxCoeffC2",
}

GROUNDWATERFLOW_MAP = {
    "boundaryxmin": "GroundwaterFlow.BoundaryXMin",
    "boundaryxmax": "GroundwaterFlow.BoundaryXMax",
    "boundaryymin": "GroundwaterFlow.BoundaryYMin",
    "boundaryymax": "GroundwaterFlow.BoundaryYMax",
    "boundaryzmin": "GroundwaterFlow.BoundaryZMin",
    "boundaryzmax": "GroundwaterFlow.BoundaryZMax",
}

DEFORMATIONS_BC_MAP = {
    "boundaryxmin": "Deformations.BoundaryXMin",
    "boundaryxmax": "Deformations.BoundaryXMax",
    "boundaryymin": "Deformations.BoundaryYMin",
    "boundaryymax": "Deformations.BoundaryYMax",
    "boundaryzmin": "Deformations.BoundaryZMin",
    "boundaryzmax": "Deformations.BoundaryZMax",
}

FLAT_ALIAS: Dict[str, str] = {
    **PHASE_MAP, **DEFORM_MAP, **FLOW_MAP, **DYNAMICS_MAP, **GROUNDWATERFLOW_MAP, **DEFORMATIONS_BC_MAP
}

# ########## Enum normalization tables ##########
ENUMS: Dict[str, Dict[str, str]] = {
    "DeformCalcType": {
        _norm("K0 procedure"): "K0 procedure",
        _norm("Gravity loading"): "Gravity loading",
        _norm("Flow only"): "Flow only",
        _norm("Plastic"): "Plastic",
        _norm("Consolidation"): "Consolidation",
        _norm("Safety"): "Safety",
        _norm("Dynamic"): "Dynamic",
        _norm("Fully coupled flow-deformation"): "Fully coupled flow-deformation",
    },
    "PorePresCalcType": {
        _norm("Phreatic"): "Phreatic",
        _norm("Use pressures from previous phase"): "Use pressures from previous phase",
        _norm("Steady state groundwater flow"): "Steady state groundwater flow",
        _norm("Transient groundwater flow"): "Transient groundwater flow",
    },
    "Deform.ArcLengthControl": {
        _norm("On"): "On", _norm("Off"): "Off", _norm("Auto"): "Auto",
    },
    "Deform.TimeStepDetermType": {
        _norm("Automatic"): "Automatic", _norm("Manual"): "Manual",
    },
    "Deform.LoadingType": {
        _norm("Staged construction"): "Staged construction",
        _norm("Minimum excess pore pressure"): "Minimum excess pore pressure",
        _norm("Degree of consolidation"): "Degree of consolidation",
        _norm("Target SumMsf"): "Target SumMsf",
        _norm("Incremental multipliers"): "Incremental multipliers",
    },
    "Flow.ThermalCalcType": {
        _norm("Steady state thermal flow"): "Steady state thermal flow",
        _norm("Transient thermal flow"): "Transient thermal flow",
    },
    "Dynamics.BoundaryXMin": { _norm("Viscous"): "Viscous", _norm("Free field"): "Free field", _norm("None"): "None" },
    "Dynamics.BoundaryXMax": { _norm("Viscous"): "Viscous", _norm("Free field"): "Free field", _norm("None"): "None" },
    "Dynamics.BoundaryYMin": { _norm("Viscous"): "Viscous", _norm("Free field"): "Free field", _norm("None"): "None" },
    "Dynamics.BoundaryYMax": { _norm("Viscous"): "Viscous", _norm("Free field"): "Free field", _norm("None"): "None" },
    "Dynamics.BoundaryZMin": { _norm("Viscous"): "Viscous", _norm("Compliant base"): "Compliant base", _norm("None"): "None" },
    "Dynamics.BoundaryZMax": { _norm("Viscous"): "Viscous", _norm("None"): "None" },
    "GroundwaterFlow.BoundaryXMin": { _norm("Open"): "Open", _norm("Closed"): "Closed" },
    "GroundwaterFlow.BoundaryXMax": { _norm("Open"): "Open", _norm("Closed"): "Closed" },
    "GroundwaterFlow.BoundaryYMin": { _norm("Open"): "Open", _norm("Closed"): "Closed" },
    "GroundwaterFlow.BoundaryYMax": { _norm("Open"): "Open", _norm("Closed"): "Closed" },
    "GroundwaterFlow.BoundaryZMin": { _norm("Open"): "Open", _norm("Closed"): "Closed" },
    "GroundwaterFlow.BoundaryZMax": { _norm("Open"): "Open", _norm("Closed"): "Closed" },
    "Deformations.BoundaryXMin": {
        _norm("Free"): "Free", _norm("Normally fixed"): "Normally fixed",
        _norm("Horizontally fixed"): "Horizontally fixed", _norm("Vertically fixed"): "Vertically fixed",
        _norm("Fully fixed"): "Fully fixed",
    },
    "Deformations.BoundaryXMax": {
        _norm("Free"): "Free", _norm("Normally fixed"): "Normally fixed",
        _norm("Horizontally fixed"): "Horizontally fixed", _norm("Vertically fixed"): "Vertically fixed",
        _norm("Fully fixed"): "Fully fixed",
    },
    "Deformations.BoundaryYMin": {
        _norm("Free"): "Free", _norm("Normally fixed"): "Normally fixed",
        _norm("Horizontally fixed"): "Horizontally fixed", _norm("Vertically fixed"): "Vertically fixed",
        _norm("Fully fixed"): "Fully fixed",
    },
    "Deformations.BoundaryYMax": {
        _norm("Free"): "Free", _norm("Normally fixed"): "Normally fixed",
        _norm("Horizontally fixed"): "Horizontally fixed", _norm("Vertically fixed"): "Vertically fixed",
        _norm("Fully fixed"): "Fully fixed",
    },
    "Deformations.BoundaryZMin": {
        _norm("Free"): "Free", _norm("Normally fixed"): "Normally fixed",
        _norm("Horizontally fixed"): "Horizontally fixed", _norm("Vertically fixed"): "Vertically fixed",
        _norm("Fully fixed"): "Fully fixed",
    },
    "Deformations.BoundaryZMax": {
        _norm("Free"): "Free", _norm("Normally fixed"): "Normally fixed",
        _norm("Horizontally fixed"): "Horizontally fixed", _norm("Vertically fixed"): "Vertically fixed",
        _norm("Fully fixed"): "Fully fixed",
    },
}

def _coerce_enum_value(enum_key: str, raw: Any) -> Any:
    """
    Map user-provided values to PLAXIS canonical enum strings according to ENUMS[enum_key].
    - Accepts case/space/underscore/dash-insensitive strings (via _norm).
    - If the table contains On/Off and user passes True/False (or 'true'/'false'/'on'/'off'), map accordingly.
    - If value is a list/tuple, map element-wise.
    """
    # element-wise for sequences
    if isinstance(raw, (list, tuple)):
        return type(raw)(_coerce_enum_value(enum_key, v) for v in raw)

    table = ENUMS.get(enum_key)
    if not table:
        return raw

    # boolean → On/Off if table supports it
    if isinstance(raw, bool):
        if _norm("On") in table and _norm("Off") in table:
            return table[_norm("On")] if raw else table[_norm("Off")]
        return raw

    # string normalization
    if isinstance(raw, str):
        key = _norm(raw)
        # allow typical boolean-like strings
        if key in {"true", "yes", "on", "1"} and _norm("On") in table:
            return table[_norm("On")]
        if key in {"false", "no", "off", "0"} and _norm("Off") in table:
            return table[_norm("Off")]
        # direct lookup
        if key in table:
            return table[key]
        # if already canonical (exact match), keep as-is
        return raw

    # numeric or others pass through
    return raw


def _coerce_for_property(label: str, prop: str, value: Any) -> Any:
    """
    Try enum coercion with two keys:
      1) '<Label>.<Prop>' (e.g., 'Deform.TimeStepDetermType')
      2) '<Prop>'         (e.g., 'DeformCalcType', 'PorePresCalcType')
    Falls back to original value if no enum table matches.
    """
    # prefer dotted path when label is a known subobject; ignore 'Phase'
    dotted = f"{label}.{prop}" if label and label.lower() != "phase" else prop

    # 1) dotted match
    coerced = _coerce_enum_value(dotted, value)
    if coerced is not value:
        return coerced

    # 2) plain property match
    coerced = _coerce_enum_value(prop, value)
    return coerced

def _batch_set(g_i: Any, obj: Any, bag: Dict[str, Any], label: str, warn: bool = False):
    """Apply properties on a PLAXIS proxy object.
    For each key, coerce its value via ENUMS (using '<Label>.<Prop>' or '<Prop>') before setting.
    Fast path: setattr()/setproperties per key; fallback: g_i.set(handle, value).
    """
    if not bag or obj is None:
        return

    # Try fast path (per-key to isolate offenders early)
    for k, v in bag.items():
        cv = _coerce_for_property(label, k, v)
        try:
            # Prefer setproperties if available (more robust on PLAXIS proxies)
            if hasattr(obj, "setproperties"):
                obj.setproperties(**{k: cv})  # type: ignore[attr-defined]
            else:
                setattr(obj, k, cv)
        except Exception:
            # Fallback to g_i.set on the underlying attribute handle
            try:
                h = getattr(obj, k)
                g_i.set(h, cv)
            except Exception as ee:
                if warn:
                    pref = f"{label}." if label and label.lower() != "phase" else ""
                    print(f"[phase-apply] skipped {pref}{k}: {ee}")


class PhaseMapper:
    """
    A defensive, version-tolerant mapper for PLAXIS Staged Construction phases.

    Public API (Phase-first):
      - goto_stages(g_i) -> None
      - wrap_initial_as_phase(g_i, **opts) -> Phase
      - create(g_i, phase: Phase, inherits: Optional[Phase|Any]) -> Phase
      - apply_phase(g_i, phase: Phase, warn_on_missing=False) -> None
      - apply_plan(g_i, phases: Sequence[Phase], warn_on_missing=False) -> List[Phase]
      - update(g_i, phase: Phase, base: Optional[Phase|Any], warn_on_missing=False) -> Dict[str,Any]

    Compatibility helpers (internal use OK):
      - get_initial_phase(g_i) -> PhaseHandle
      - find_phase_handle(g_i, name: str) -> Optional[PhaseHandle]
    """

    # #####################################################################
    # Stage navigation & discovery
    # #####################################################################

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
    def get_initial_phase(g_i: Any) -> Any:
        """
        Return the Initial Phase handle (robust).
        Tries: g_i.InitialPhase, first of g_i.Phases, or search by Identification.
        """
        PhaseMapper.goto_stages(g_i)
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

        # 3) best-effort name scan（通常不需要）
        try:
            for p in getattr(g_i, "Phases", []):
                name = getattr(p, "Identification", None) or getattr(p, "Name", None)
                if str(getattr(getattr(name, "value", name), "__str__", lambda: name)()) in ("Initial phase", "InitialPhase"):
                    return p
        except Exception:
            pass

        raise RuntimeError("Cannot resolve InitialPhase handle from g_i.")

    # ##################### Wrap InitialPhase as a Phase domain object #####################

    @staticmethod
    def wrap_initial_as_phase(
        g_i: Any,
        *,
        name: str = "P0_Initial",
        comment: str = "Wrapped InitialPhase (Staged construction)",
        # 仅允许的少量设置（其余采用 PlasticStageSettings 的强类型默认值）
        ΣM_stage: float = 0.0,
        ΣM_weight: float = 1.0,
        pore_cal_type: PoreCalType | str = PoreCalType.Phreatic,
        tolerance_error: float = 1.0e-2,
        max_steps: int = 100,
        time_interval: float = 0.0,
        solver: SolverType = SolverType.PICO,
    ) -> Phase:
        """
        Build a Phase-domain object that *represents the existing InitialPhase*.
        Forcely let 'Staged construction'(PlasticStageSettings) + a few parameters.
        Returnned Phase will bind the plx_id=InitialPhase handle.
        """
        initial_handle = PhaseMapper.get_initial_phase(g_i)

        # 归一化孔压计算方式（字符串/枚举均可）
        if isinstance(pore_cal_type, str):
            pore_mode = PoreCalType.Phreatic if pore_cal_type.lower().startswith("phreatic") else PoreCalType.LastStage
        else:
            pore_mode = pore_cal_type

        settings = PlasticStageSettings(
            load_type=LoadType.StageConstruction,
            pore_cal_type=pore_mode,
            ΣM_stage=ΣM_stage,
            ΣM_weight=ΣM_weight,
            tolerance_error=tolerance_error,
            max_steps=max_steps,
            time_interval=time_interval,
            solver=solver,
        )

        phase = Phase(
            name=name,
            comment=comment,
            settings=settings,
            activate=[],
            deactivate=[],
            inherits=None,  # 初始相无继承
        )
        setattr(phase, "plx_id", initial_handle)
        return phase

    @staticmethod
    def wrap_initial_with_toggles(
        g_i: Any,
        *,
        name: str = "P0_Initial",
        comment: str = "Wrapped InitialPhase (Staged construction)",
        activate: Sequence[BaseStructure] = (),
        deactivate: Sequence[BaseStructure] = (),
        **kwargs,
    ) -> Phase:
        """语法糖：封装 InitialPhase 并附带激活/冻结清单。"""
        ph = PhaseMapper.wrap_initial_as_phase(g_i, name=name, comment=comment, **kwargs)
        if activate:
            ph.activate_structures(activate)
        if deactivate:
            ph.deactivate_structures(deactivate)
        return ph

    # Compatibility helper
    @staticmethod
    def find_phase_handle(g_i: Any, name: str) -> Optional[PhaseHandle]:
        """Find a phase by Identification/Name (best-effort)."""
        if not name:
            return None
        try:
            col = getattr(g_i, "Phases", None)
            if not col:
                return None
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

    # #####################################################################
    # Private low-level helpers (handle land; not public)
    # #####################################################################

    @staticmethod
    def _create_phase_from_base(g_i: Any, previous: PhaseHandle) -> PhaseHandle:
        """Create a new phase that inherits from 'previous' (try several APIs)."""
        for fn in ("phase", "addphase", "Phase"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    return f(previous)
                except Exception:
                    pass

        # Empty create then set PreviousPhase/Parent
        for fn in ("phase", "addphase", "Phase"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    new_ph = f()
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

    @staticmethod
    def _ensure_phase_handle(g_i: Any, phase_obj: Phase, base: Optional[Phase | Any] = None) -> PhaseHandle:
        """
        Ensure a PLAXIS phase handle exists for the given Phase object.
        Rules:
        1. If phase_obj.plx_id already exists, return it directly.
        2. Otherwise, determine the base phase for inheritance: prioritize "base", followed by phase_obj.inherits, and finally fall back to "Initial".
        3. If the base phase for inh
        eritance is a Phase object and does not have a plx_id, first create a handle for it.
        4. Create a new phase and write the name/comment back to it.
        """
        h = getattr(phase_obj, "plx_id", None)
        if h is not None:
            return h

        base_src = base if base is not None else getattr(phase_obj, "inherits", None)

        base_handle = None
        if isinstance(base_src, Phase):
            base_handle = getattr(base_src, "plx_id", None)
            if base_handle is None:
                base_handle = PhaseMapper._ensure_phase_handle(g_i, base_src)
        elif base_src is not None:
            base_handle = base_src

        if base_handle is None:
            base_handle = PhaseMapper.get_initial_phase(g_i)

        new_ph = PhaseMapper._create_phase_from_base(g_i, base_handle)
        PhaseMapper._maybe_set_identification(new_ph, getattr(phase_obj, "name", None))
        PhaseMapper._maybe_set_comment(new_ph, getattr(phase_obj, "comment", None))
        setattr(phase_obj, "plx_id", new_ph)
        return new_ph

    # #####################################################################
    # Public: create/apply using Phase objects
    # #####################################################################

    @staticmethod
    def create(g_i: Any, phase: Phase, *, inherits: Optional[Phase | Any] = None,
            return_handle: bool = False, apply_structure: bool = True) -> Union[Phase, Any]:
        """
        Create a new PLAXIS phase for this Phase object (if needed), set Identification/Comment,
        bind handle back to phase.plx_id, and return the same Phase object.
        """
        PhaseMapper.goto_stages(g_i)
        base = inherits if inherits is not None else getattr(phase, "inherits", None)
        h = PhaseMapper._ensure_phase_handle(g_i, phase, base=base)
        PhaseMapper.apply_options(g_i, phase)
        if apply_structure:
            if (getattr(phase, "activate", None) or getattr(phase, "deactivate", None)) or getattr(phase, "wells_dict", None):
                try:
                    print(f"[INFO] {phase.name} is applying the structures. ~")
                    PhaseMapper.apply_structures(g_i, phase, warn_on_missing=True)
                except Exception as e:
                    # Do not interrupt the process; errors will be reported again in the subsequent apply_phase.
                    print(f"Applying the structures in a phase is failed with the error message: {e}")
                    if return_handle:
                        return h
                    return phase
        
        if phase.water_table:
            try:
                WaterTableMapper.create_table(g_i, phase.water_table)
            except Exception as e:
                print(f"Create water table failed in phase {phase.name} with error message {e}")

        return h if return_handle else phase



    @staticmethod
    def _set_phase_attr(phase_handle: PhaseHandle, key: str, value: Any) -> bool:
        """
        Try to set phase attribute by key, with alias tweaks:
          - dotted keys 'Deform.ResetDisplacements'
          - Greek 'Σ' => 'Sum' / 'Sigma'
        """
        # dot path
        if "." in key:
            cur = phase_handle
            parts = key.split(".")
            try:
                for p in parts[:-1]:
                    cur = getattr(cur, p)
                setattr(cur, parts[-1], value)
                return True
            except Exception:
                return False

        candidates = [key, key.replace("Σ", "Sum"), key.replace("Σ", "Sigma")]
        for k in candidates:
            try:
                setattr(phase_handle, k, value)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def apply_options(g_i: Any, phase: Phase, warn_on_missing: bool = False, **_ignored) -> None:
        """
        Static entry point: read settings from the phase object and push them
        into PLAXIS using setproperties(). Supported sources on the phase:
        - phase.settings      (dict or object with to_dict())
        - phase.stage_settings
        - phase.options
        - phase.stageOptions
        Keys can be friendly names (e.g., 'ignore_undrained') or canonical dotted
        paths (e.g., 'Deform.IgnoreUndrainedBehaviour').
        """
        # 1) Pull settings directly from the phase
        settings_obj = None
        for attr in ("settings", "stage_settings", "options", "stageOptions"):
            if hasattr(phase, attr):
                settings_obj = getattr(phase, attr)
                break

        if settings_obj is None:
            return

        # Convert to dict if needed
        if not isinstance(settings_obj, dict):
            if hasattr(settings_obj, "to_dict"):
                settings: Dict[str, Any] = settings_obj.to_dict()
            elif hasattr(settings_obj, "to_phase_dict"):
                settings = settings_obj.to_phase_dict()
            else:
                # Last resort: use object __dict__
                settings = dict(getattr(settings_obj, "__dict__", {}))
        else:
            settings = settings_obj

        if not settings:
            return

        # 2) Collect properties per target
        top_props: Dict[str, Any] = {}
        deform_props: Dict[str, Any] = {}
        flow_props: Dict[str, Any] = {}
        dynamics_props: Dict[str, Any] = {}
        gwflow_props: Dict[str, Any] = {}
        deforms_bc_props: Dict[str, Any] = {}

        skipped: list[tuple[str, str]] = []

        # 3) Translate & coerce
        for raw_key, val in settings.items():
            path = resolve_path(raw_key, FLAT_ALIAS)
            if not path:
                skipped.append((raw_key, "unmapped"))
                continue

            coerced = coerce_value(path, val, ENUMS)
            sub, prop = split_target(path)

            if sub is None:
                top_props[prop] = coerced
            elif sub == "Deform":
                deform_props[prop] = coerced
            elif sub == "Flow":
                flow_props[prop] = coerced
            elif sub == "Dynamics":
                dynamics_props[prop] = coerced
            elif sub == "GroundwaterFlow":
                gwflow_props[prop] = coerced
            elif sub == "Deformations":
                deforms_bc_props[prop] = coerced
            else:
                skipped.append((raw_key, f"unknown sub-object {sub}"))

        ph_plx = phase.plx_id
        # 4) Apply via setproperties()
        _batch_set(g_i, ph_plx, top_props, "Phase", warn_on_missing)
        _batch_set(g_i, getattr(ph_plx, "Deform", None), deform_props, "Deform", warn_on_missing)
        _batch_set(g_i, getattr(ph_plx, "Flow", None), flow_props, "Flow", warn_on_missing)
        _batch_set(g_i, getattr(ph_plx, "Dynamics", None), dynamics_props, "Dynamics", warn_on_missing)
        _batch_set(g_i, getattr(ph_plx, "GroundwaterFlow", None), gwflow_props, "GroundwaterFlow", warn_on_missing)
        _batch_set(g_i, getattr(ph_plx, "Deformations", None), deforms_bc_props, "Deformations", warn_on_missing)

        # 5) Optional diagnostics
        if warn_on_missing and skipped:
            for raw_key, reason in skipped:
                print(f"[phase-apply] skipped key '{raw_key}': {reason}")

    # #### structures activation/deactivation
    @staticmethod
    def _resolve_handle(obj: Any) -> Optional[Any]:
        """Resolve a PLAXIS object handle from a domain object or a raw handle."""
        if obj is None:
            return None
        h = getattr(obj, "plx_id", None)
        if h is not None:
            return h
        return obj if PhaseMapper._looks_like_handle(obj) else None

    @staticmethod
    def _activate(g_i: Any, handle: Any, phase_handle: Any) -> bool:
        for fn in ("activate", "Activate"):
            f = getattr(g_i, fn, None)
            if callable(f):
                # 位置参数（你调试验证成功的调用方式）
                try:
                    f(handle, phase_handle)
                    return True
                except Exception:
                    pass
                # 关键字回退
                try:
                    f(handle, phase=phase_handle)
                    return True
                except Exception:
                    pass
        # 属性法回退
        try:
            active = getattr(handle, "Active", None)
            if active is not None:
                active[phase_handle] = True
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _deactivate(g_i: Any, handle: Any, phase_handle: Any) -> bool:
        for fn in ("deactivate", "Deactivate"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    f(handle, phase_handle)
                    return True
                except Exception:
                    pass
                try:
                    f(handle, phase=phase_handle)
                    return True
                except Exception:
                    pass
        try:
            active = getattr(handle, "Active", None)
            if active is not None:
                active[phase_handle] = False
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def apply_structures(g_i: Any,
                        phase_or_handle: Union[Phase, Any],
                        activate: Iterable[Any] = (),
                        deactivate: Iterable[Any] = (),
                        *, warn_on_missing: bool = False) -> None:
        """
        Apply the active-deactive status of structures for a phase, and update the wells' parameters.
        Dual-mode：
        1) New: apply_structures(g_i, phase: Phase, ...)
        - If activate/deactivate is not explicitly passed in, it will read phase.activate/phase.deactivate
        2) Old: apply_structures(g_i, phase_handle, activate=... , deactivate=...)
        """
        if isinstance(phase_or_handle, Phase):
            ph_handle = PhaseMapper._ensure_phase_handle(g_i, phase_or_handle)
            src_activate = list(activate or []) if activate else (getattr(phase_or_handle, "activate", None) or [])
            src_deactivate = list(deactivate or []) if deactivate else (getattr(phase_or_handle, "deactivate", None) or [])
        else:
            ph_handle = phase_or_handle
            src_activate = list(activate or [])
            src_deactivate = list(deactivate or [])

        # 1) 解析为基对象句柄（含按名兜底）
        act, deact = [], []
        for o in src_activate:
            handles = PhaseMapper._ensure_object_handle(g_i, o, warn_on_missing=warn_on_missing)
            act.extend(handles)
        
        for o in src_deactivate:
            handles = PhaseMapper._ensure_object_handle(g_i, o, warn_on_missing=warn_on_missing)
            deact.extend(handles)
            
        # 2) 🔹分块展开（关键：把 PLATE → 所有 PLATE_n 一起切换）
        act = PhaseMapper._expand_segments_for_handles(g_i, act)
        deact = PhaseMapper._expand_segments_for_handles(g_i, deact)

        # 3) 去重 + 冻结优先
        def _dedupe(seq: list[Any]) -> list[Any]:
            seen, out = set(), []
            for h in seq:
                k = id(h)
                if k not in seen:
                    seen.add(k)
                    out.append(h)
            return out
        act = _dedupe(act)
        deact = _dedupe(deact)
        deact_ids = {id(h) for h in deact}
        act = [h for h in act if id(h) not in deact_ids]

        # 4) 执行（先位置参数，再关键字，再属性法）
        for h in act:
            ok = PhaseMapper._activate(g_i, h, ph_handle)
            if not ok and warn_on_missing:
                name = _ident_str(getattr(h, "Identification", None)) or _ident_str(getattr(h, "Name", None))
                print(f"[PhaseMapper.apply_structures] activate failed for {name or h}.")
        for h in deact:
            ok = PhaseMapper._deactivate(g_i, h, ph_handle)
            if not ok and warn_on_missing:
                name = _ident_str(getattr(h, "Identification", None)) or _ident_str(getattr(h, "Name", None))
                print(f"[PhaseMapper.apply_structures] deactivate failed for {name or h}.")

        # 5) 更新井的流量参数
        if isinstance(phase_or_handle, Phase):
            if phase_or_handle.wells_dict:
                well_handles = [well.plx_id for well in phase_or_handle.wells_dict.keys()]
                values = list(phase_or_handle.wells_dict.values())
                well_handles = PhaseMapper._expand_segments_for_handles(g_i, well_handles)
                for wh, value in zip(well_handles, values):
                        g_i.set(getattr(wh, "Q", None), ph_handle, value)
                        print(f"[PhaseMapper.apply_structures] Update the Q={value} of well {str(wh.Name)} for {phase_or_handle.name}.")
            

    # =========================================================================
    # helper tools: Pick up objects by name from the stage phase
    # =========================================================================

    @staticmethod
    def _iter_candidate_collections(g_i: Any, type_tag: Optional[str] = None) -> List[Any]:
        cols: List[Any] = []
        # 先按族找容器
        rule = _rule_by_tag(type_tag)
        if rule:
            for n in rule["containers"]:
                try:
                    c = getattr(g_i, n, None)
                    if c:
                        _ = iter(c)
                        cols.append(c)
                except Exception:
                    pass
        # 兜底：全局常见容器
        if not cols:
            for n in [
                "Plates","Walls","Beams","EmbeddedBeams","EmbeddedPiles",
                "NodeToNodeAnchors","NodetoNodeAnchors","Anchors","Node_to_node_anchors",
                "Wells","Pipes","SoilVolumes","Soils","Structures","Objects",
            ]:
                try:
                    c = getattr(g_i, n, None)
                    if c:
                        _ = iter(c)
                        cols.append(c)
                except Exception:
                    pass
        return cols


    @staticmethod
    def _find_model_object_by_name(g_i: Any, name: str, *, type_tag: Optional[str] = None) -> Optional[Any]:
        if not name:
            return None

        # 1) 容器优先（你已有按 type_tag 限定的 _iter_candidate_collections，可保留）
        containers = PhaseMapper._iter_candidate_collections(g_i, type_tag=type_tag)

        # 2) 先做精确匹配
        targets = {_canon(name), _canon(name).replace("_", ""), name}
        for col in containers:
            it = _try_iter(col)
            for obj in it:
                ident = _ident_str(getattr(obj, "Identification", None)) or _ident_str(getattr(obj, "Name", None))
                if not ident:
                    continue
                cident = _canon(ident)
                if cident in targets or cident.replace("_", "") in targets:
                    # 对可能的“组”优先返回叶子
                    return (PhaseMapper._collect_children_from_attrs(obj, ("Children","Members","Items","SubObjects")) or [obj])[0]

        # 3) 家族兜底：使用健壮的家族匹配正则
        fam_pats = _family_regexes_for_name(name)  # ← 新的统一族匹配
        for col in containers:
            it = _try_iter(col)
            for obj in it:
                ident = _ident_str(getattr(obj, "Identification", None)) or _ident_str(getattr(obj, "Name", None))
                if not ident:
                    continue
                cident = _canon(ident)
                if any(pat.match(cident) for pat in fam_pats):
                    kids = PhaseMapper._collect_children_from_attrs(obj, ("Children","Members","Items","SubObjects"))
                    return kids[0] if kids else obj

        return None


    @staticmethod
    def _collect_family_in_containers(g_i: Any, containers: Sequence[str], base_name: str, suffix_regex: str) -> list[Any]:
        out: list[Any] = []
        pat = re.compile(rf"^{re.escape(_canon(base_name))}{suffix_regex}$", flags=re.I)
        for cn in containers:
            col = getattr(g_i, cn, None)
            if not col:
                continue
            for obj in _try_iter(col):
                ident = _ident_str(getattr(obj, "Identification", None)) or _ident_str(getattr(obj, "Name", None))
                if ident and pat.match(_canon(ident)):
                    out.append(obj)
        return out

    @staticmethod
    def _expand_handle_name(handle: Any) -> str:
        if handle:
            return str(handle.Name)
        return "There is no handle"

    @staticmethod
    def _expand_segments_for_handle(g_i: Any, handle: Any) -> list[Any]:
        if handle is None:
            return []
        segs = [handle]

        # 1) 展开子对象（若存在）
        segs.extend(PhaseMapper._collect_children_from_attrs(handle, ("Children","Members","SubObjects","Items")))

        # 2) 家族聚合（基于 Identification/Name + 家族正则）
        ident = _ident_str(getattr(handle, "Identification", None)) or _ident_str(getattr(handle, "Name", None))
        if ident:
            fam_pats = _family_regexes_for_name(ident)
            handle_name = PhaseMapper._expand_handle_name(handle)
            # 在常见容器里按族扩展（不限定容器也行；为了性能你可以配合 type_tag 限定）
            for col in PhaseMapper._iter_candidate_collections(g_i, type_tag=_guess_type_tag(handle)):
                for obj in _try_iter(col):
                    oid = _ident_str(getattr(obj, "Identification", None)) or _ident_str(getattr(obj, "Name", None))
                    if oid and any(p.match(_canon(oid)) for p in fam_pats) and handle_name in oid:
                        segs.append(obj)

        # 去重
        uniq, seen = [], set()
        for h in segs:
            k = id(h)
            if k not in seen:
                seen.add(k); uniq.append(h)
        return uniq


    @staticmethod
    def _expand_segments_for_handles(g_i: Any, handles: list[Any]) -> list[Any]:
        out: list[Any] = []
        for h in handles or []:
            out.extend(PhaseMapper._expand_segments_for_handle(g_i, h))
        # 去重
        uniq, seen = [], set()
        for h in out:
            k = id(h)
            if k not in seen and "Feature" in str(h.TypeName):
                seen.add(k)
                uniq.append(h)
        return uniq

    @staticmethod
    def _collect_children_from_attrs(handle: Any, attrs: Sequence[str]) -> list[Any]:
        out: list[Any] = []
        for attr in attrs or ():
            try:
                col = getattr(handle, attr, None)
                if col:
                    try:
                        out.extend(col[:])
                    except Exception:
                        for x in col:
                            out.append(x)
            except Exception:
                continue
        return out

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
        """Extract scalar head (z) from various payloads."""
        if isinstance(water_tbl, (int, float)):
            try:
                return float(water_tbl)
            except Exception:
                return None
        if isinstance(water_tbl, dict):
            return PhaseMapper._extract_head_from_mapping(water_tbl)
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
        for k in ("head", "level", "z", "H", "h"):
            try:
                v = getattr(water_tbl, k, None)
                if v is not None:
                    return float(v)
            except Exception:
                continue
        return None

    @staticmethod
    def _apply_water_table(g_i: Any, phase: Phase, *, warn_on_missing: bool = False) -> None:
        """Best-effort: set scalar head on phase or call solver helper."""
        h = PhaseMapper._ensure_phase_handle(g_i, phase)
        head = PhaseMapper._extract_head(getattr(phase, "water_table", None))
        if head is not None:
            for attr in ("WaterLevel", "PhreaticLevel", "WaterLevelHead", "Head"):
                try:
                    setattr(h, attr, head)
                    return
                except Exception:
                    pass
            for fn_name in ("setwaterlevel", "SetWaterLevel", "set_water_level"):
                fn = getattr(g_i, fn_name, None)
                if callable(fn):
                    try:
                        fn(head, phase=h)
                        return
                    except Exception:
                        pass
            if warn_on_missing:
                print("[PhaseMapper] Could not set scalar head on phase; skipped.")
            return

        # object-level APIs (rare)
        for fn_name in ("setwaterlevelobj", "SetWaterLevelObject", "set_water_level_object"):
            fn = getattr(g_i, fn_name, None)
            if callable(fn):
                try:
                    fn(getattr(phase, "water_table", None), phase=h)
                    return
                except Exception:
                    pass
        if warn_on_missing:
            print("[PhaseMapper] No supported API found to apply water table object; skipped.")

    # #### wells

    @staticmethod
    def _looks_like_handle(obj: Any) -> bool:
        return obj is not None and any(
            hasattr(obj, a) for a in ("Active", "Identification", "IsValid", "Phase", "Material")
        )

    @staticmethod
    def _ensure_object_handle(g_i: Any, obj: Any, *, warn_on_missing: bool = False) -> List[Any]:
        if obj is None:
            return []

        
        # 1) 已有句柄直接用（若是属性句柄，建议在 structuremapper.create 时就重绑为容器对象）
        # 1.1) If the object is a retaining wall, the interfaces of a retaining wall should be processed
        handles = []
        from ..structures import RetainingWall
        def pickup_interface(interface_obj):
            if interface_obj:
                name = str(getattr(interface_obj.plx_id, "Name", ""))
                interface_handle = getattr(g_i, name, None)
                if interface_handle:
                    return interface_handle
            return None

        if isinstance(obj, RetainingWall):
            interfaces = obj.interfaces
            for i in interfaces:
                ih = pickup_interface(i)
                if ih:
                    handles.append(ih)
        

        h = getattr(obj, "plx_id", None)
        if h:
            handles.append(h)
            return handles

        # 2) 按名字 + 族 查找
        type_tag = _guess_type_tag(obj)  # 会从 plx_id.TypeName 或类名 Anchor 推断 'n2n_anchor'
        name = getattr(obj, "name", None)
        if isinstance(name, str) and name:
            h = PhaseMapper._find_model_object_by_name(g_i, name, type_tag=type_tag)
            if h:
                try: setattr(obj, "plx_id", h)
                except Exception: pass
                return [h]

        # 3) 退化：obj 本身像句柄
        if PhaseMapper._looks_like_handle(obj):
            return [obj]
        
        if warn_on_missing:
            print(f"[PhaseMapper] cannot resolve handle for object: {getattr(obj,'name',obj)}")
        return []


    @staticmethod
    def _dedupe_handles(handles: Iterable[Any]) -> list[Any]:
        seen, out = set(), []
        for h in handles or []:
            k = id(h)
            if k not in seen:
                seen.add(k)
                out.append(h)
        return out

    @staticmethod
    def _normalize_toggles(
        g_i: Any,
        activate: Iterable[Any],
        deactivate: Iterable[Any],
        *,
        warn_on_missing: bool = False
    ) -> tuple[list[Any], list[Any]]:
        """
        Normalized activation/freeze list:
        - All are converted to valid handles (with name-based fallback, and if successful, the plx_id is written back)
        - De-duplication
        - Mutual exclusion handling: If the same object is both activated and frozen -> prioritize freezing
        """
        act_handles, deact_handles = [], []

        for obj in (activate or []):
            h = PhaseMapper._ensure_object_handle(g_i, obj, warn_on_missing=warn_on_missing)
            if h is not None:
                act_handles.extend(h)
        for obj in (deactivate or []):
            h = PhaseMapper._ensure_object_handle(g_i, obj, warn_on_missing=warn_on_missing)
            if h is not None:
                deact_handles.extend(h)

        def _unique(seq: list[Any]) -> list[Any]:
            seen, out = set(), []
            for x in seq:
                k = id(x)
                if k not in seen:
                    seen.add(k)
                    out.append(x)
            return out

        act_handles = _unique(act_handles)
        deact_handles = _unique(deact_handles)

        deact_ids = {id(h) for h in deact_handles}
        act_handles = [h for h in act_handles if id(h) not in deact_ids]

        return act_handles, deact_handles

    @staticmethod
    def _set_object_param(g_i: Any, handle: Any, key: str, value: Any, phase_handle: Optional[PhaseHandle]) -> bool:
        """Try to set a parameter on a model object, optionally for a phase."""
        try:
            setattr(handle, key, value)
            return True
        except Exception:
            pass
        for fn in ("setparameter", "SetParameter", "set"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    if phase_handle is not None:
                        f(handle, key, value, phase=phase_handle)
                    else:
                        f(handle, key, value)
                    return True
                except Exception:
                    continue
        return False

    @staticmethod
    def apply_well_overrides_dict(g_i: Any, phase: Phase, overrides: Dict[str, Dict[str, Any]],
                                  *, warn_on_missing: bool = False) -> None:
        """
        Apply well parameter overrides per phase:
            { "Well-1": {"q_well": 0.008, "h_min": -10.0, "well_type": "Extraction"} }
        """
        if not isinstance(overrides, dict):
            return

        phase_handle = PhaseMapper._ensure_phase_handle(g_i, phase)
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

                applied = False
                for key_try in candidates:
                    if PhaseMapper._set_object_param(g_i, handle, key_try, value_present, phase_handle):
                        applied = True
                        break
                if not applied:
                    PhaseMapper._set_object_param(g_i, handle, canonical, value_present, phase_handle)

    @staticmethod
    def apply_well_overrides(g_i: Any, phase: Phase, *, warn_on_missing: bool = False) -> None:
        """Compatibility wrapper expecting an object with `.well_overrides`."""
        mapping = getattr(phase, "well_overrides", None)
        if isinstance(mapping, dict):
            PhaseMapper.apply_well_overrides_dict(g_i, phase, mapping, warn_on_missing=warn_on_missing)

    # #####################################################################
    # High-level "apply phase" (Phase-first)
    # #####################################################################

    @staticmethod
    def apply_phase(g_i: Any, phase: Phase, *, warn_on_missing: bool = False) -> None:
        """
        Apply one phase in four steps, using Phase object:
          1) options/settings
          2) water table (optional)
          3) structure activation/deactivation
          4) well overrides (optional)
        """
        PhaseMapper.goto_stages(g_i)
        PhaseMapper._ensure_phase_handle(g_i, phase)

        PhaseMapper.apply_options(g_i, phase, warn_on_missing=warn_on_missing)
        PhaseMapper._apply_water_table(g_i, phase, warn_on_missing=warn_on_missing)

        # 👉 强烈建议把 warn_on_missing=True 传下去
        PhaseMapper.apply_structures(g_i, phase, warn_on_missing=warn_on_missing)

        PhaseMapper.apply_well_overrides(g_i, phase, warn_on_missing=warn_on_missing)

    # #####################################################################
    # Batch helpers (Phase-first)
    # #####################################################################

    @staticmethod
    def apply_plan(g_i: Any, phases: Sequence[Phase], *, warn_on_missing: bool = False) -> List[Phase]:
        """
        Create and apply a chain of phases in order (each inherits from previous).
        Returns the *Phase objects* (after binding their plx_id).
        """
        if not phases:
            return []

        PhaseMapper.goto_stages(g_i)
        previous_handle = PhaseMapper.get_initial_phase(g_i)

        out: List[Phase] = []
        for ph in phases:
            # 优先读取 ph.inherits；没有则用链上前一个
            base_src = getattr(ph, "inherits", None)
            if base_src is None:
                base_handle = previous_handle
            elif isinstance(base_src, Phase):
                base_handle = getattr(base_src, "plx_id", None)
                if base_handle is None:
                    base_handle = PhaseMapper._ensure_phase_handle(g_i, base_src)
            else:
                base_handle = base_src  # 允许直接给句柄

            # 确保创建，并应用设置/结构/水位/井
            PhaseMapper._ensure_phase_handle(g_i, ph, base=base_handle)
            PhaseMapper.apply_phase(g_i, ph, warn_on_missing=warn_on_missing)

            previous_handle = getattr(ph, "plx_id", previous_handle)
            out.append(ph)
        return out

    @staticmethod
    def update(g_i: Any, phase: Phase, base: Optional[Phase | Any] = None,
               *, warn_on_missing: bool = False) -> Dict[str, Any]:
        """
        Idempotent update (Phase-first):
          - If phase has handle: apply on it.
          - Else create inheriting from `base` (or Initial), then apply.
        """
        report = {"created": False, "applied": False, "handle": None}

        h = getattr(phase, "plx_id", None)
        if h is None:
            PhaseMapper._ensure_phase_handle(g_i, phase, base=base)
            report["created"] = True

        try:
            PhaseMapper.apply_phase(g_i, phase, warn_on_missing=warn_on_missing)
            report["applied"] = True
        finally:
            report["handle"] = getattr(phase, "plx_id", None)
        return report


# ========================================================================================
# Soil picker
# ========================================================================================

    @staticmethod
    def _to_name(obj):
        """Return the PLAXIS object name across API variants (obj.Name or obj.Name.value)."""
        try:
            return str(obj.Name.value)
        except Exception:
            try:
                return str(obj.Name)
            except Exception:
                return ""

    @staticmethod
    def _enter_stages(g_i):
        """Switch to the Staged construction view (safe to call multiple times)."""
        try:
            g_i.gotostages()
        except Exception:
            # Some versions may use a different command; if it fails, keep going.
            pass

    @staticmethod
    def _iter_split_soils_in_stages(g_i, phase=None):
        """
        Read soils AFTER they have been split by retaining structures.
        Must be called from the Staged view; otherwise child names (Soil_k_1, Soil_k_2, ...) are not visible.

        Args:
            g_i: PLAXIS Input object.
            phase: optional phase handle; InitialPhase is used if not provided.

        Returns:
            List of tuples (handle, name) for split soil bodies.
        """
        PhaseMapper._enter_stages(g_i)

        # Choose a phase (InitialPhase preferred, but not strictly required just to read names)
        try:
            ph = phase or g_i.InitialPhase
        except Exception:
            try:
                ph = g_i.Phases[0]
            except Exception:
                ph = None  # Not used further; kept for future extension

        items = []
        try:
            for s in g_i.Soils:
                nm = PhaseMapper._to_name(s)
                if nm:
                    items.append((s, nm))
        except Exception:
            # If Soils is not available for any reason, return empty list
            pass
        return items

    @staticmethod
    def _parse_parent_and_index(soil_name):
        """
        Parse names like:
            'Soil_2_1' -> parent='Soil_2', idx=1
            'Soil_2'   -> parent='Soil_2', idx=None

        Returns:
            (parent_name, index_or_None)
        """
        m = re.match(r'^(Soil_\d+)(?:_(\d+))?$', soil_name)
        if not m:
            return None, None
        parent = m.group(1)
        idx = int(m.group(2)) if m.group(2) is not None else None
        return parent, idx

    @staticmethod
    def collect_split_groups(g_i, phase=None):
        """
        Group split child soils by their parent.

        Returns:
            dict:
            parent_name -> list of tuples (handle, name, idx, volume_or_None)
            Only child pieces with a numeric suffix are included (Soil_k_1, Soil_k_2, ...).

        Note:
            This function does NOT filter on the number of children; the
            filtering (>=2 children) is done in guess_excavation_soils().
        """
        groups = defaultdict(list)
        for h, nm in PhaseMapper._iter_split_soils_in_stages(g_i, phase):
            parent, idx = PhaseMapper._parse_parent_and_index(nm)
            if parent is None or idx is None:
                continue

            # Try to read volume; API attributes may differ by version
            vol = None
            for attr in ("Volume", "volume", "V"):
                try:
                    vol = float(getattr(h, attr))
                    break
                except Exception:
                    pass

            groups[parent].append((h, nm, idx, vol))
        return groups

    @staticmethod
    def get_all_child_soils(g_i, phase=None) -> Dict[Any, str]:
        """
        Return a flat dict of ALL split child soils visible in Staged view:
            { soil_handle -> soil_name }
        Only children with numeric suffix (Soil_k_1, Soil_k_2, ...) are included.
        """
        PhaseMapper._enter_stages(g_i)
        groups = PhaseMapper.collect_split_groups(g_i, phase=phase)
        flat: Dict[Any, str] = {}
        for _, items in groups.items():
            for handle, name, _, _ in items:
                flat[name] = handle
        return flat

    @staticmethod
    def get_all_child_soil_names(g_i, phase=None) -> List[str]:
        """
        Convenience wrapper returning only names (list), preserving insertion order.
        """
        return list(PhaseMapper.get_all_child_soils(g_i, phase=phase).values())

    @staticmethod
    def guess_excavation_soils(g_i, phase=None, prefer_volume=True):
        """
        Pick the excavated (enclosed) soil pieces before creating excavation phases.

        Rules per parent group:
        - Ignore parents that have ONLY ONE child (i.e., only *_1 and no *_2):
            they are NOT considered excavation candidates.
        - If volume is available AND prefer_volume is True: pick the smallest volume.
        - Otherwise: pick the smallest numeric suffix (idx=1).
        """
        PhaseMapper._enter_stages(g_i)
        groups = PhaseMapper.collect_split_groups(g_i, phase)
        results = []
        for parent, items in groups.items():
            # NEW: skip parents with a single child (e.g., only Soil_k_1)
            if len(items) < 2:
                continue
            if prefer_volume and all(v[3] is not None for v in items):
                pick = min(items, key=lambda x: x[3])   # by volume
            else:
                pick = min(items, key=lambda x: x[2])   # by suffix index
            results.append(pick[0])  # handle
        return results

    @staticmethod
    def guess_excavation_soil_names(g_i, phase=None, prefer_volume: bool = True) -> Dict[Any, str]:
        """
        Return a dict mapping { soil_handle -> soil_name } for the candidate
        enclosed (pit) soil clusters, discovered in Staged view.

        Selection per parent group:
          - If volumes are available and prefer_volume=True: choose smallest volume.
          - Else: choose smallest numeric suffix (idx=1).

        Note:
          Python dict preserves insertion order; order follows the handle list returned
          by guess_excavation_soils(...).
        """
        handles: List[Any] = PhaseMapper.guess_excavation_soils(
            g_i, phase=phase, prefer_volume=prefer_volume
        )
        return {h: PhaseMapper._to_name(h) for h in handles}

    # ### optional legacy helper (keep if some code still expects a list of names) ###
    @staticmethod
    def guess_excavation_soil_names_list(g_i, phase=None, prefer_volume: bool = True) -> List[str]:
        """
        Legacy convenience: return only names (list) for logging/inspection.
        """
        m = PhaseMapper.guess_excavation_soil_names(g_i, phase=phase, prefer_volume=prefer_volume)
        return list(m.values())

    @staticmethod
    def apply_deactivate_soilblock(g_i, phase: Phase):
        """ Deactivate the selected soillayers in the specified phase. """
        for soil_block in phase.soil_blocks:
            PhaseMapper._deactivate(g_i, soil_block.plx_id, phase.plx_id)