"""
phasesettings.py — Strongly-typed stage settings (no Optional/|None), mapper-aligned.

What this module provides
-------------------------
1) Enums for calculation, loading, pore calculation, solver and mass matrix types.
2) A strict-typed `StageSettingsBase` (no Optional), with sensible defaults.
3) Concrete stages:
   - PlasticStageSettings
   - ConsolidationStageSettings
   - FullyCoupledStageSettings  (flow–deformation, “fully coupled”)
   - DynamicStageSettings
   - DynamicWithConsolidationStageSettings
   - SafetyStageSettings
4) A tolerant `from_dict` pipeline that accepts strings/aliases and normalizes
   everything to concrete types in the dataclasses (no Optional fields).
5) `to_settings_dict()` of each class emits keys exactly matching PhaseMapper:
   Deform/Loading/Flow/Numerical/Dynamics/Safety + ΣM(aliases) 等。

Key design for "Automatic"/optional fields
------------------------------------------
- Internally we store **fixed types** only.
- For time-step fields (first/min/max_time_step) and other optional knobs we use
  a **sentinel**: `0.0` or `0` means “not specified / Automatic”.
- `to_settings_dict()` 将把 sentinel 转换成 `None`（从而让 PhaseMapper 跳过该键），
  或者直接不导出该键。这保证对 Plaxis 的 “Automatic” 行为兼容，同时满足你对
  “全部参数固定类型”的要求。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence, Type, TypeVar, cast


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class CalcType(Enum):
    Plastic = "Plastic"
    Consolidation = "Consolidation"                # Flow-only solver (transient/steady)
    FlowConsolidation = "Fully coupled flow-deformation"
    Dynamic = "Dynamic"
    DynamicConsolidation = "Dynamic with consolidation"
    Safety = "Safety"


class LoadType(Enum):
    StageConstruction = "Staged construction"
    MinimizePorePressure = "Minimum excess pore pressure"
    ConsolidationDegree = "Degree of consolidation"
    # 其余 UI 选项可按需补充


class PoreCalType(Enum):
    Phreatic = "Phreatic"
    LastStage = "Use pressures from previous phase"
    SteadySeepage = "Steady state groundwater flow"


class SolverType(Enum):
    PICO = "Picos (multicore iterative)"
    Pardiso = "Pardiso (multicore direct)"
    Classic = "Classic (single core iterative)"


class MassMatrixType(Enum):
    Consistent = "Consistent"
    Lumped = "Lumped"


class SafetyMode(Enum):
    IncrementalMultipliers = "IncrementalMultipliers"  # 输出 msf
    TargetSumMsf = "TargetSumMsf"                      # 输出 sum_msf


# -----------------------------------------------------------------------------
# Helpers (normalization + tiny codecs)
# -----------------------------------------------------------------------------
TStage = TypeVar("TStage", bound="StageSettingsBase")

_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}
_AUTO = {"", "automatic", "auto"}

_ALIAS_MAP: Dict[str, str] = {
    # Dynamics / Numerical
    "dynamic_time_interval_s": "dynamic_time_interval",
    "number_of_sub_steps": "number_sub_steps",
    # Flow / Loading
    "pore_calc_type": "pore_cal_type",
    "consolidation_degree": "degree_of_consolidation",
    "min_excess_pore_pressure": "p_stop",
    # ΣM
    "SigmaMstage": "ΣM_stage",
    "SigmaMweight": "ΣM_weight",
    "ΣMweight": "ΣM_weight",
    "weight_sum": "ΣM_weight",
}

def _omit_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys whose values are None (for clean serialization)."""
    return {k: v for k, v in d.items() if v is not None}

def _auto_none(v: float) -> Any:
    """Return None if sentinel (<=0.0), else the value itself."""
    return None if v <= 0.0 else v

def _auto_none_int(v: int) -> Any:
    """Return None if sentinel (<=0), else the value itself."""
    return None if v <= 0 else v

def _enum_from(enum: Type[Enum], v: Any, default: Enum) -> Enum:
    """Parse enum from name/value; fallback to default."""
    if isinstance(v, enum):
        return v
    s = str(v).strip()
    for m in enum:  # by value
        if s == str(m.value):
            return m
    try:            # by name (case-sensitive to keep strict)
        return enum[s]  # type: ignore[index]
    except Exception:
        return default

def _as_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in _TRUE:  return True
    if s in _FALSE: return False
    return default

def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _as_int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return default

def normalize_settings_input(data: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Alias→canonical, coerce strings (“Automatic”→sentinel 0/0.0), keep fixed types."""
    d: Dict[str, Any] = dict(data or {})

    # alias → canonical
    for a, c in _ALIAS_MAP.items():
        if a in d and c not in d:
            d[c] = d[a]

    # automatic sentinel (time steps only)
    for k in ("first_time_step", "min_time_step", "max_time_step"):
        if k in d:
            s = str(d[k]).strip().lower()
            if s in _AUTO:
                d[k] = 0.0

    return dict(d)


# -----------------------------------------------------------------------------
# Base class — fixed types only (no Optional), with mapper-aligned defaults
# -----------------------------------------------------------------------------
@dataclass
class StageSettingsBase:
    """
    Shared knobs across all stages. **No Optional types** — use sentinel:
    - 0.0 (float) or 0 (int) means “Automatic/Not set”.
    """

    # identity / link
    id: str = "Phase_1"
    start_from_phase: str = "Initial phase"
    name: str = ""

    # identity
    calc_type: CalcType = CalcType.Plastic
    load_type: LoadType = LoadType.StageConstruction
    pore_cal_type: PoreCalType = PoreCalType.Phreatic

    # Loading (General timing)
    time_interval: float = 0.0
    estimated_end_time: float = 0.0
    first_step: float = 0.0
    last_step: float = 0.0
    special_option: int = 0

    # ΣM (staged construction multipliers)
    ΣM_stage: float = 0.0      # not always used; Plastic常用
    ΣM_weight: float = 1.0     # Consolidation/FC常见

    # Deform
    ignore_undr_behavior: bool = False
    force_fully_drained: bool = False
    reset_displacemnet: bool = False
    reset_small_strain: bool = False
    reset_state_variable: bool = False
    reset_time: bool = False
    update_mesh: bool = False
    ignore_suction_F: bool = False
    cavitation_cutoff: bool = False
    cavitation_limit: float = 0.0

    # Numerical
    solver: SolverType = SolverType.PICO
    max_cores_use: int = 8
    max_number_of_step_store: int = 25
    use_compression_result: bool = False
    use_default_iter_param: bool = True
    max_steps: int = 100
    time_step_determination: str = "Automatic"  # "Automatic" or "Manual"
    first_time_step: float = 0.0   # 0.0 => Automatic
    min_time_step: float = 0.0     # 0.0 => Automatic
    max_time_step: float = 0.0     # 0.0 => Automatic
    tolerance_error: float = 1e-4
    max_unloading_step: int = 0
    max_load_fraction_per_step: float = 0.0
    over_relaxation_factor: float = 0.0
    max_iterations: int = 50
    desired_min_iterations: int = 0
    desired_max_iterations: int = 0
    Arc_length_control: bool = False
    use_subspace_accelerator: bool = False
    subspace_size: int = 0
    line_search: bool = False
    use_gradual_error_reduction: bool = False
    number_sub_steps: int = 0     # dynamic families

    # Dynamics
    dynamic_time_interval: float = 0.0
    newmark_alpha: float = 0.0
    newmark_beta: float = 0.0
    mass_matrix: MassMatrixType = MassMatrixType.Consistent

    # Flow
    flow_use_default_iter_param: bool = True
    flow_max_steps: int = 1000
    flow_tolerance_error: float = 5.0e-3
    flow_over_relaxation_factor: float = 1.5

    # Safety
    safety_multiplier: SafetyMode = SafetyMode.IncrementalMultipliers
    msf: float = 0.0
    sum_msf: float = 0.0

    # Consolidation-specific extras
    p_stop: float = 0.0                    # Minimum excess pore pressure
    degree_of_consolidation: float = 0.0   # [0,100], when using degree target

    # ---------------- serialization ----------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Enums → value
        for k, v in list(d.items()):
            if isinstance(v, Enum):
                d[k] = v.value
        return _omit_none(d)

    @classmethod
    def from_dict(cls: Type[TStage], data: Dict[str, Any]) -> TStage:
        """Tolerant parsing from dict, but end result uses **fixed types** only."""
        d = normalize_settings_input(data)

        # Enums
        if "calc_type" in d:
            d["calc_type"] = _enum_from(CalcType, d["calc_type"], cls.calc_type)
        if "load_type" in d:
            d["load_type"] = _enum_from(LoadType, d["load_type"], cls.load_type)
        if "pore_cal_type" in d:
            d["pore_cal_type"] = _enum_from(PoreCalType, d["pore_cal_type"], cls.pore_cal_type)
        if "solver" in d:
            d["solver"] = _enum_from(SolverType, d["solver"], cls.solver)
        if "mass_matrix" in d:
            d["mass_matrix"] = _enum_from(MassMatrixType, d["mass_matrix"], cls.mass_matrix)
        if "safety_multiplier" in d:
            d["safety_multiplier"] = _enum_from(SafetyMode, d["safety_multiplier"], cls.safety_multiplier)

        # Booleans
        for b in (
            "ignore_undr_behavior", "force_fully_drained", "reset_displacemnet",
            "reset_small_strain", "reset_state_variable", "reset_time", "update_mesh",
            "ignore_suction_F", "cavitation_cutoff", "Arc_length_control",
            "use_compression_result", "use_default_iter_param", "use_subspace_accelerator",
            "line_search", "use_gradual_error_reduction", "flow_use_default_iter_param",
        ):
            if b in d: d[b] = _as_bool(d[b], getattr(cls, b))  # type: ignore[arg-type]

        # Floats
        f_keys = (
            "time_interval", "estimated_end_time", "first_step", "last_step",
            "cavitation_limit", "tolerance_error", "max_load_fraction_per_step",
            "over_relaxation_factor", "newmark_alpha", "newmark_beta",
            "dynamic_time_interval", "flow_tolerance_error", "flow_over_relaxation_factor",
            "msf", "sum_msf", "p_stop", "degree_of_consolidation",
            "ΣM_stage", "ΣM_weight", "first_time_step", "min_time_step", "max_time_step",
        )
        for k in f_keys:
            if k in d: d[k] = _as_float(d[k], getattr(cls, k))  # type: ignore[arg-type]

        # Ints
        i_keys = (
            "special_option", "max_cores_use", "max_number_of_step_store", "max_steps",
            "max_unloading_step", "max_iterations", "desired_min_iterations", "desired_max_iterations",
            "subspace_size", "number_sub_steps", "flow_max_steps",
        )
        for k in i_keys:
            if k in d: d[k] = _as_int(d[k], getattr(cls, k))  # type: ignore[arg-type]

        # Strings
        for k in ("id", "start_from_phase", "name", "time_step_determination"):
            if k in d: d[k] = str(d[k])

        return cls(**d)  # type: ignore[arg-type]

    # mapper-friendly export; subclasses specialize as needed
    def to_settings_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Concrete stages (fixed-typed, with mapper-aligned exports)
# -----------------------------------------------------------------------------
@dataclass
class PlasticStageSettings(StageSettingsBase):
    calc_type: CalcType = CalcType.Plastic
    load_type: LoadType = LoadType.StageConstruction
    pore_cal_type: PoreCalType = PoreCalType.LastStage

    # Stage-specific defaults
    ΣM_stage: float = 0.5
    ΣM_weight: float = 1.0
    tolerance_error: float = 1.0e-2
    ignore_undr_behavior: bool = False
    reset_displacemnet: bool = True
    reset_small_strain: bool = True
    solver: SolverType = SolverType.PICO

    def to_settings_dict(self) -> Dict[str, Any]:
        return _omit_none({
            # General
            "calc_type": self.calc_type.value,
            "load_type": self.load_type.value,
            "pore_cal_type": (
                "Use pressures from" if self.pore_cal_type is PoreCalType.LastStage else self.pore_cal_type.value
            ),
            "time_interval": self.time_interval,
            "estimated_end_time": self.estimated_end_time,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "special_option": self.special_option,
            # ΣM (+ aliases)
            "ΣM_stage": self.ΣM_stage,
            "ΣM_weight": self.ΣM_weight,
            "SigmaMstage": self.ΣM_stage,
            "SigmaMweight": self.ΣM_weight,
            # Deform
            "ignore_undr_behavior": self.ignore_undr_behavior,
            "reset_displacemnet": self.reset_displacemnet,
            "reset_small_strain": self.reset_small_strain,
            "reset_state_variable": self.reset_state_variable,
            "reset_time": self.reset_time,
            "update_mesh": self.update_mesh,
            "ignore_suction_F": self.ignore_suction_F,
            "cavitation_cutoff": self.cavitation_cutoff,
            "cavitation_limit": self.cavitation_limit,
            # Numerical
            "solver": self.solver.value,
            "max_cores_use": self.max_cores_use,
            "max_number_of_step_store": self.max_number_of_step_store,
            "use_compression_result": self.use_compression_result,
            "use_default_iter_param": self.use_default_iter_param,
            "max_steps": self.max_steps,
            "tolerance_error": self.tolerance_error,
            "max_unloading_step": self.max_unloading_step,
            "max_load_fraction_per_step": self.max_load_fraction_per_step,
            "over_relaxation_factor": self.over_relaxation_factor,
            "max_iterations": self.max_iterations,
            "desired_min_iterations": self.desired_min_iterations,
            "desired_max_iterations": self.desired_max_iterations,
            "Arc_length_control": self.Arc_length_control,
            "use_subspace_accelerator": self.use_subspace_accelerator,
            "subspace_size": self.subspace_size,
            "line_search": self.line_search,
            "use_gradual_error_reduction": self.use_gradual_error_reduction,
        })


@dataclass
class ConsolidationStageSettings(StageSettingsBase):
    calc_type: CalcType = CalcType.Consolidation
    load_type: LoadType = LoadType.StageConstruction
    pore_cal_type: PoreCalType = PoreCalType.LastStage

    force_fully_drained: bool = True
    ignore_suction_F: bool = True

    def to_settings_dict(self) -> Dict[str, Any]:
        out = {
            "calc_type": self.calc_type.value,
            "load_type": self.load_type.value,
            "pore_cal_type": (
                "Use pressures from" if self.pore_cal_type is PoreCalType.LastStage
                else ("Steady state groundwater" if self.pore_cal_type is PoreCalType.SteadySeepage
                      else self.pore_cal_type.value)
            ),
            # ΣM_weight（该族常见）
            "ΣM_weight": self.ΣM_weight,
            "SigmaMweight": self.ΣM_weight,
            # Loading
            "time_interval": self.time_interval,
            "estimated_end_time": self.estimated_end_time,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "special_option": self.special_option,
            # Deform
            "force_fully_drained": self.force_fully_drained,
            "reset_displacemnet": self.reset_displacemnet,
            "reset_small_strain": self.reset_small_strain,
            "reset_state_variable": self.reset_state_variable,
            "reset_time": self.reset_time,
            "update_mesh": self.update_mesh,
            "ignore_suction_F": self.ignore_suction_F,
            "cavitation_cutoff": self.cavitation_cutoff,
            "cavitation_limit": self.cavitation_limit,
            # Numerical
            "solver": self.solver.value,
            "max_cores_use": self.max_cores_use,
            "max_number_of_step_store": self.max_number_of_step_store,
            "use_compression_result": self.use_compression_result,
            "use_default_iter_param": self.use_default_iter_param,
            "max_steps": self.max_steps,
            "time_step_determination": self.time_step_determination,
            "first_time_step": _auto_none(self.first_time_step),
            "min_time_step": _auto_none(self.min_time_step),
            "max_time_step": _auto_none(self.max_time_step),
            "tolerance_error": self.tolerance_error,
            "max_unloading_step": self.max_unloading_step,
            "max_load_fraction_per_step": self.max_load_fraction_per_step,
            "over_relaxation_factor": self.over_relaxation_factor,
            "max_iterations": self.max_iterations,
            "desired_min_iterations": self.desired_min_iterations,
            "desired_max_iterations": self.desired_max_iterations,
            "use_subspace_accelerator": self.use_subspace_accelerator,
            "subspace_size": self.subspace_size,
        }
        # Consolidation extras by load_type
        if self.load_type == LoadType.MinimizePorePressure and self.p_stop > 0.0:
            out["p_stop"] = self.p_stop
        if self.load_type == LoadType.ConsolidationDegree and self.degree_of_consolidation > 0.0:
            out["degree_of_consolidation"] = self.degree_of_consolidation
            out["solidation_degree"] = self.degree_of_consolidation  # alias
        # Flow solver block (only if SteadySeepage)
        if self.pore_cal_type is PoreCalType.SteadySeepage:
            out.update({
                "flow_use_default_iter_param": self.flow_use_default_iter_param,
                "flow_max_steps": self.flow_max_steps,
                "flow_tolerance_error": self.flow_tolerance_error,
                "flow_over_relaxation_factor": self.flow_over_relaxation_factor,
            })
        return _omit_none(out)


@dataclass
class FullyCoupledStageSettings(StageSettingsBase):
    calc_type: CalcType = CalcType.FlowConsolidation
    load_type: LoadType = LoadType.StageConstruction
    solver: SolverType = SolverType.Pardiso

    force_fully_drained: bool = True
    tolerance_error: float = 1.0e-2

    def to_settings_dict(self) -> Dict[str, Any]:
        return _omit_none({
            "calc_type": self.calc_type.value,
            "load_type": self.load_type.value,
            # Loading
            "time_interval": self.time_interval,
            "estimated_end_time": self.estimated_end_time,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "special_option": self.special_option,
            # ΣM_weight (+ alias)
            "ΣM_weight": self.ΣM_weight,
            "SigmaMweight": self.ΣM_weight,
            # Deform
            "force_fully_drained": self.force_fully_drained,
            "reset_displacemnet": self.reset_displacemnet,
            "reset_small_strain": self.reset_small_strain,
            "reset_state_variable": self.reset_state_variable,
            "reset_time": self.reset_time,
            "ignore_suction_F": self.ignore_suction_F,
            "cavitation_cutoff": self.cavitation_cutoff,
            "cavitation_limit": self.cavitation_limit,
            # Numerical
            "solver": self.solver.value,
            "max_cores_use": self.max_cores_use,
            "max_number_of_step_store": self.max_number_of_step_store,
            "use_compression_result": self.use_compression_result,
            "use_default_iter_param": self.use_default_iter_param,
            "max_steps": self.max_steps,
            "time_step_determination": self.time_step_determination,
            "first_time_step": _auto_none(self.first_time_step),
            "min_time_step": _auto_none(self.min_time_step),
            "max_time_step": _auto_none(self.max_time_step),
            "tolerance_error": self.tolerance_error,
            "max_load_fraction_per_step": self.max_load_fraction_per_step,
            "over_relaxation_factor": self.over_relaxation_factor,
            "max_iterations": self.max_iterations,
            "desired_min_iterations": self.desired_min_iterations,
            "desired_max_iterations": self.desired_max_iterations,
            "use_subspace_accelerator": self.use_subspace_accelerator,
            "subspace_size": self.subspace_size,
            # Flow (常用默认)
            "flow_use_default_iter_param": self.flow_use_default_iter_param,
            "flow_max_steps": self.flow_max_steps,
            "flow_tolerance_error": self.flow_tolerance_error,
            "flow_over_relaxation_factor": self.flow_over_relaxation_factor,
        })


@dataclass
class DynamicStageSettings(StageSettingsBase):
    calc_type: CalcType = CalcType.Dynamic
    load_type: LoadType = LoadType.StageConstruction
    pore_cal_type: PoreCalType = PoreCalType.LastStage
    solver: SolverType = SolverType.Pardiso

    reset_state_variable: bool = True
    update_mesh: bool = True
    tolerance_error: float = 1.0e-2
    max_load_fraction_per_step: float = 0.25
    newmark_alpha: float = 0.25
    newmark_beta: float = 0.5

    def to_settings_dict(self) -> Dict[str, Any]:
        return _omit_none({
            "calc_type": self.calc_type.value,
            "load_type": self.load_type.value,
            "pore_cal_type": (
                "Use pressures from" if self.pore_cal_type is PoreCalType.LastStage else self.pore_cal_type.value
            ),
            # Loading
            "dynamic_time_interval": self.dynamic_time_interval,
            "estimated_end_time": self.estimated_end_time,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "special_option": self.special_option,
            # Deform
            "ignore_undr_behavior": self.ignore_undr_behavior,
            "reset_displacemnet": self.reset_displacemnet,
            "reset_small_strain": self.reset_small_strain,
            "reset_state_variable": self.reset_state_variable,
            "reset_time": self.reset_time,
            "update_mesh": self.update_mesh,
            "ignore_suction_F": self.ignore_suction_F,
            "cavitation_cutoff": self.cavitation_cutoff,
            "cavitation_limit": self.cavitation_limit,
            # Numerical
            "solver": self.solver.value,
            "max_cores_use": self.max_cores_use,
            "max_number_of_step_store": self.max_number_of_step_store,
            "use_compression_result": self.use_compression_result,
            "use_default_iter_param": self.use_default_iter_param,
            "max_steps": self.max_steps,
            "time_step_determination": self.time_step_determination,
            "number_sub_steps": _auto_none_int(self.number_sub_steps),
            "tolerance_error": self.tolerance_error,
            "max_unloading_step": self.max_unloading_step,
            "max_load_fraction_per_step": self.max_load_fraction_per_step,
            "over_relaxation_factor": self.over_relaxation_factor,
            "max_iterations": self.max_iterations,
            "desired_min_iterations": self.desired_min_iterations,
            "desired_max_iterations": self.desired_max_iterations,
            "use_subspace_accelerator": self.use_subspace_accelerator,
            "subspace_size": self.subspace_size,
            "line_search": self.line_search,
            "use_gradual_error_reduction": self.use_gradual_error_reduction,
            # Dynamics
            "newmark_alpha": self.newmark_alpha,
            "newmark_beta": self.newmark_beta,
            "mass_matrix": self.mass_matrix.value,
        })


@dataclass
class DynamicWithConsolidationStageSettings(StageSettingsBase):
    calc_type: CalcType = CalcType.DynamicConsolidation
    load_type: LoadType = LoadType.StageConstruction
    pore_cal_type: PoreCalType = PoreCalType.LastStage
    solver: SolverType = SolverType.Pardiso

    max_load_fraction_per_step: float = 0.25
    tolerance_error: float = 1.0e-2
    newmark_alpha: float = 0.25
    newmark_beta: float = 0.5

    def to_settings_dict(self) -> Dict[str, Any]:
        return _omit_none({
            "calc_type": self.calc_type.value,
            "load_type": self.load_type.value,
            "pore_cal_type": (
                "Use pressures from" if self.pore_cal_type is PoreCalType.LastStage else self.pore_cal_type.value
            ),
            # Loading
            "dynamic_time_interval": self.dynamic_time_interval,
            "estimated_end_time": self.estimated_end_time,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "special_option": self.special_option,
            # Deform
            "ignore_undr_behavior": self.ignore_undr_behavior,
            "reset_displacemnet": self.reset_displacemnet,
            "reset_small_strain": self.reset_small_strain,
            "reset_state_variable": self.reset_state_variable,
            "reset_time": self.reset_time,
            "update_mesh": self.update_mesh,
            "ignore_suction_F": self.ignore_suction_F,
            "cavitation_cutoff": self.cavitation_cutoff,
            "cavitation_limit": self.cavitation_limit,
            # Numerical
            "solver": self.solver.value,
            "max_cores_use": self.max_cores_use,
            "max_number_of_step_store": self.max_number_of_step_store,
            "use_compression_result": self.use_compression_result,
            "use_default_iter_param": self.use_default_iter_param,
            "max_steps": self.max_steps,
            "time_step_determination": self.time_step_determination,
            "number_sub_steps": _auto_none_int(self.number_sub_steps),
            "tolerance_error": self.tolerance_error,
            "max_unloading_step": self.max_unloading_step,
            "max_load_fraction_per_step": self.max_load_fraction_per_step,
            "over_relaxation_factor": self.over_relaxation_factor,
            "max_iterations": self.max_iterations,
            "desired_min_iterations": self.desired_min_iterations,
            "desired_max_iterations": self.desired_max_iterations,
            "use_subspace_accelerator": self.use_subspace_accelerator,
            "subspace_size": self.subspace_size,
            "line_search": self.line_search,
            "use_gradual_error_reduction": self.use_gradual_error_reduction,
            # Dynamics
            "newmark_alpha": self.newmark_alpha,
            "newmark_beta": self.newmark_beta,
            "mass_matrix": self.mass_matrix.value,
        })


@dataclass
class SafetyStageSettings(StageSettingsBase):
    calc_type: CalcType = CalcType.Safety
    safety_multiplier: SafetyMode = SafetyMode.IncrementalMultipliers
    pore_cal_type: PoreCalType = PoreCalType.LastStage
    solver: SolverType = SolverType.Pardiso
    tolerance_error: float = 1.0e-2

    def to_settings_dict(self) -> Dict[str, Any]:
        out = {
            "calc_type": self.calc_type.value,
            "safety_multiplier": self.safety_multiplier.value,
            "pore_cal_type": (
                "Use pressures from" if self.pore_cal_type is PoreCalType.LastStage else self.pore_cal_type.value
            ),
            "first_step": self.first_step,
            "last_step": self.last_step,
            "special_option": self.special_option,
            # Deform
            "ignore_undr_behavior": self.ignore_undr_behavior,
            "reset_displacemnet": self.reset_displacemnet,
            "reset_small_strain": self.reset_small_strain,
            "reset_state_variable": self.reset_state_variable,
            "update_mesh": self.update_mesh,
            "ignore_suction_F": self.ignore_suction_F,
            "cavitation_cutoff": self.cavitation_cutoff,
            "cavitation_limit": self.cavitation_limit,
            # Numerical
            "solver": self.solver.value,
            "max_cores_use": self.max_cores_use,
            "max_number_of_step_store": self.max_number_of_step_store,
            "use_compression_result": self.use_compression_result,
            "use_default_iter_param": self.use_default_iter_param,
            "max_steps": self.max_steps,
            "tolerance_error": self.tolerance_error,
            "max_unloading_step": self.max_unloading_step,
            "over_relaxation_factor": self.over_relaxation_factor,
            "max_iterations": self.max_iterations,
            "desired_min_iterations": self.desired_min_iterations,
            "desired_max_iterations": self.desired_max_iterations,
            "Arc_length_control": self.Arc_length_control,
            "use_subspace_accelerator": self.use_subspace_accelerator,
            "subspace_size": self.subspace_size,
            "line_search": self.line_search,
        }
        # Safety mode specific
        if self.safety_multiplier == SafetyMode.IncrementalMultipliers and self.msf > 0.0:
            out["msf"] = self.msf
        if self.safety_multiplier == SafetyMode.TargetSumMsf and self.sum_msf > 0.0:
            out["sum_msf"] = self.sum_msf
        return _omit_none(out)


# -----------------------------------------------------------------------------
# Factory + Plan helpers
# -----------------------------------------------------------------------------
_STAGE_REGISTRY: dict[CalcType, type[StageSettingsBase]] = {
    CalcType.Plastic: PlasticStageSettings,
    CalcType.Consolidation: ConsolidationStageSettings,
    CalcType.FlowConsolidation: FullyCoupledStageSettings,
    CalcType.Dynamic: DynamicStageSettings,
    CalcType.DynamicConsolidation: DynamicWithConsolidationStageSettings,
    CalcType.Safety: SafetyStageSettings,
}


class StageSettingsFactory:
    """Create proper StageSettings subclass from a plain dict (Pylance-safe)."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> StageSettingsBase:
        # 1) 预处理：别名归一化、"Automatic" → 哨兵值
        d = normalize_settings_input(data)

        # 2) 解析 calc_type（接受枚举名或枚举值字符串）
        calc_raw = d.get("calc_type", CalcType.Plastic.value)
        ct = cast(CalcType, _enum_from(CalcType, calc_raw, CalcType.Plastic))

        # 3) Pylance 友好：先拿 Optional，再显式兜底，最终类型固定为 type[StageSettingsBase]
        cls_opt = _STAGE_REGISTRY.get(ct)            # -> type[StageSettingsBase] | None
        if cls_opt is None:
            # 显式 cast，告诉类型检查器这是一个满足签名的构造类（且有 classmethod from_dict）
            cls: type[StageSettingsBase] = cast(type[StageSettingsBase], PlasticStageSettings)
        else:
            cls = cls_opt

        # 4) 子类的 classmethod from_dict 返回具体的 StageSettingsBase 派生实例
        return cls.from_dict(d)

    @staticmethod
    def list_from_dict(items: Sequence[dict[str, Any]] | None) -> List[StageSettingsBase]:
        # 接受 None，避免上游空值时 union 推断不稳定
        if not items:
            return []
        return [StageSettingsFactory.from_dict(x) for x in items]

@dataclass
class StagePlan:
    """A simple ordered holder for stage settings with mapper export."""
    stages: List[StageSettingsBase] = field(default_factory=list)

    def add(self, *s: StageSettingsBase) -> None:
        self.stages.extend(s)

    def to_dict(self) -> Dict[str, Any]:
        return {"stages": [s.to_dict() for s in self.stages]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StagePlan":
        return cls(stages=StageSettingsFactory.list_from_dict(data.get("stages", [])))

    def to_settings_payloads(self) -> List[Dict[str, Any]]:
        """Build mapper-ready list: one payload per stage."""
        return [s.to_settings_dict() for s in self.stages]
