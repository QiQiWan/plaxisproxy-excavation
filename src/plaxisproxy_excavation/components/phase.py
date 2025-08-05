from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
from ..core.plaxisobject import PlaxisObject
from watertable import WaterLevelTable
from ..structures.soilblock import SoilBlock

# ----------------------------------------------------------------------
# 1) StageSettings Enums
# ----------------------------------------------------------------------
class CalcType(Enum):
    """Enumeration for calculation types in Plaxis."""
    Plastic = "Plastic"
    Consolidation = "Consolidation"
    Safety = "Safety"
    Dynamic = "Dynamic"
    FlowConsolidation = "FlowConsolidation"
    DynamicConsolidation = "DynamicConsolidation"

class LoadType(Enum):
    """Enumeration for load types in a phase."""
    StageConstruction = "Stage by stage construction"
    MinimizePorePressure = "MinimizePorePressure"
    ConsolidationDegree = "SolidationDegree"
    TargetMulti = "TargetMulti"
    IncrementMulti = "IncrementMulti"

class SafetyMultiplier(Enum):
    """Enumeration for multipliers in safety analyse"""
    TargetSumMsf = "TargetSumMsf"
    IncrementalMultipliers = "IncrementalMultipliers"

class PoreCalType(Enum):
    """Enumeration for pore pressure calculation types."""
    Diving = "Diving"
    LastStage = "LastStage"
    SteadySeepage = "SteadySeepage"

class ConsoSubtype(Enum):
    """Enumeration for consolidation subtypes."""
    Trabsient = auto()
    Steady = auto()

class SolverType(Enum):
    """Enumeration for available solvers."""
    PICO = "Picos"
    Pardiso = "Pardiso"
    Classical = "Classical"

# ----------------------------------------------------------------------
# 2) StageSettings Dataclass and Factory Methods
# ----------------------------------------------------------------------
@dataclass
class StageSettings(PlaxisObject):
    """
    A dataclass for storing calculation settings.
    This class is intended to be created via factory methods to ensure parameter consistency.
    The `settings` dictionary holds all parameters.
    """
    calc_type: CalcType = field(default=CalcType.Plastic)
    settings: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Performs validation and ensures calc_type is in settings."""
        if 'calc_type' not in self.settings:
            self.settings['calc_type'] = self.calc_type.value

    @classmethod
    def _get_common_settings(cls, **kwargs) -> Dict:
        """
        Helper function to get common calculation settings.
        This reduces redundancy across factory methods.
        """
        return {
            # -- General --
            "reset_displacemnet": kwargs.get("reset_displacemnet", False),
            "reset_small_strain": kwargs.get("reset_small_strain", False),
            "reset_state_variable": kwargs.get("reset_state_variable", False),
            "reset_time": kwargs.get("reset_time", False),
            "p_stop": kwargs.get("p_stop", 1.0),
            "solidation_degree": kwargs.get("solidation_degree", 90.0),
            "pore_cal_type": kwargs.get("pore_cal_type", PoreCalType.LastStage.value),
            "use_default_seepage": kwargs.get("use_default_seepage", True), # PoreCalType.SteadySeepage
            "max_steps": kwargs.get("max_steps", 1000), # PoreCalType.SteadySeepage
            "steadyseepage_tolerance_error": kwargs.get("steadyseepage_tolerance_error", 5e-3), # PoreCalType.SteadySeepage
            "hyper_relax_factor": kwargs.get("hyper_relax_factor", 1.5), # PoreCalType.SteadySeepage
            "time_interval": kwargs.get("time_interval", 1.0), # Withdraw in CalcType.Safety
            # -- Deformation control Parameters --
            "ignore_suction_F": kwargs.get("ignore_suction_F", False),
            "update_mesh": kwargs.get("update_mesh", False),
            "cavitation_cutoff": kwargs.get("cavitation_cutoff", False),
            "cavitation_limit": kwargs.get("cavitation_limit", 100.0),
            # -- Numerical control parameters --
            "solver": kwargs.get("solver", SolverType.PICO.value),
            "max_cores_use": kwargs.get("max_cores_use", 256),
            "max_number_of_step_store": kwargs.get("max_number_of_step_store", 1),
            "use_compression_result": kwargs.get("use_compression_result", False),
            "use_default_iter_param": kwargs.get("use_default_iter_param", True),
            "first_time_step": kwargs.get("first_time_step", 1e-9),
            "min_time_step": kwargs.get("min_time_step", 1e-9),
            "max_time_step": kwargs.get("max_time_step", 1e-9),
            "tolerance_error": kwargs.get("tolerance_error", 1e-2),
            "max_iterations": kwargs.get("max_iterations", 60),
            "over_relaxation_factor": kwargs.get("over_relaxation_factor", 1.2),
            "max_load_fraction_per_step": kwargs.get("max_load_fraction_per_step", 0.5),
            "desired_min_iterations": kwargs.get("desired_min_iterations", 6),
            "desired_max_iterations": kwargs.get("desired_max_iterations", 15),
            "subspace_size": kwargs.get("subspace_size", 3)
        }

    @classmethod
    def create_plastic_settings(cls, **kwargs) -> StageSettings:
        """Factory method for Plastic calculation settings."""
        settings_dict = cls._get_common_settings(**kwargs)
        settings_dict.update({
            "calc_type": CalcType.Plastic.value,
            "load_type": kwargs.get("load_type", LoadType.StageConstruction.value),
            "sum_M_stage": kwargs.get("sum_M_stage", 1.0),
            "sum_M_weight": kwargs.get("sum_M_weight", 1.0),
            "hyper_relax_factor": kwargs.get("hyper_relax_factor", 1.5),
            "time_interval": kwargs.get("time_interval", 1.0),
            "ignore_drainage": kwargs.get("ignore_drainage", False),
            "max_unloading_step": kwargs.get("max_unloading_step", 5),
            "reset_time": kwargs.get("reset_time", False),
            "Arc_length_control": kwargs.get("Arc_length_control", True),
            "line_search": kwargs.get("line_search", False),
            "gradual_error_reduction": kwargs.get("gradual_error_reduction", False)
        })
        return cls(calc_type=CalcType.Plastic, settings=settings_dict)

    @classmethod
    def create_consolidation_settings(cls, **kwargs) -> StageSettings:
        """Factory method for Consolidation calculation settings."""
        settings_dict = cls._get_common_settings(**kwargs)
        settings_dict.update({
            "calc_type": CalcType.Consolidation.value,
            "load_type": kwargs.get("load_type", LoadType.StageConstruction.value),
            "sum_M_weight": kwargs.get("sum_M_weight", 1.0),
            "force_fully_drained": kwargs.get("force_fully_drained", True),
            "reset_time": kwargs.get("reset_time", False),
            "time_step_determination": kwargs.get("time_step_determination", True),
            "over_relaxation_factor": kwargs.get("over_relaxation_factor", 1.2),
            "use_subspace_accelerator": kwargs.get("use_subspace_accelerator", False)
        })
        return cls(calc_type=CalcType.Consolidation, settings=settings_dict)
    
    @classmethod
    def create_flow_consolidation_settings(cls, **kwargs) -> StageSettings:
        """Factory method for FlowConsolidation calculation settings."""
        settings_dict = cls._get_common_settings(**kwargs)
        settings_dict.update({
            "calc_type": CalcType.FlowConsolidation.value,
            "sum_M_weight": kwargs.get("sum_M_weight", 1.0),
            "force_fully_drained": kwargs.get("force_fully_drained", True),
            "time_step_determination": kwargs.get("time_step_determination", True),
            "first_time_step": kwargs.get("first_time_step", 1e-9),
            "use_subspace_accelerator": kwargs.get("use_subspace_accelerator", False)
        })
        return cls(calc_type=CalcType.FlowConsolidation, settings=settings_dict)
    
    @classmethod
    def create_safety_settings(cls, **kwargs) -> StageSettings:
        """Factory method for Safety calculation settings."""
        settings_dict = cls._get_common_settings(**kwargs)
        settings_dict.update({
            "calc_type": CalcType.Safety.value,
            "load_type": kwargs.get("load_type", LoadType.StageConstruction.value),
            "safety_multiplier": kwargs.get("safety_multiplier", SafetyMultiplier.TargetSumMsf.value),
            "sum_msf": kwargs.get("sum_msf", 1.0),
            "msf": kwargs.get("msf", 0.1),
            "ignore_drainage": kwargs.get("ignore_drainage", False),
            "max_unloading_step": kwargs.get("max_unloading_step", 5)
        })
        return cls(calc_type=CalcType.Safety, settings=settings_dict)

    @classmethod
    def create_dynamic_settings(cls, **kwargs) -> StageSettings:
        """Factory method for Dynamic calculation settings."""
        settings_dict = cls._get_common_settings(**kwargs)
        settings_dict.update({
            "calc_type": CalcType.Dynamic.value,
            "ignore_drainage": kwargs.get("ignore_drainage", False),
            "reset_time": kwargs.get("reset_time", False),
            "max_steps": kwargs.get("max_steps", 100),
            "number_sub_step": kwargs.get("number_sub_step", 1),
            "alpha_time_integration": kwargs.get("alpha_time_integration", 0.25),
            "beta_time_integration": kwargs.get("beta_time_integration", 0.5),
            "mass_matrix": kwargs.get("mass_matrix", 0.0)
        })
        return cls(calc_type=CalcType.Dynamic, settings=settings_dict)

    @classmethod
    def create_dynamic_consolidation_settings(cls, **kwargs) -> StageSettings:
        """Factory method for DynamicConsolidation calculation settings."""
        settings_dict = cls._get_common_settings(**kwargs)
        settings_dict.update({
            "calc_type": CalcType.Dynamic.value,
            "ignore_drainage": kwargs.get("ignore_drainage", False),
            "reset_time": kwargs.get("reset_time", False),
            "max_steps": kwargs.get("max_steps", 100),
            "number_sub_step": kwargs.get("number_sub_step", 1),
            "alpha_time_integration": kwargs.get("alpha_time_integration", 0.25),
            "beta_time_integration": kwargs.get("beta_time_integration", 0.5),
            "mass_matrix": kwargs.get("mass_matrix", 0.0)
        })
        return cls(calc_type=CalcType.DynamicConsolidation, settings=settings_dict)


    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-safe serialize：
        """
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "__version__": getattr(self, "_SERIAL_VERSION", 1),
            "calc_type": self.calc_type.value,
            "settings": self.settings 
        }

    # ---------- Deserialize ----------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageSettings":
        """
        Automatically select the corresponding factory method based on calc_type → then write back all parameters.
        """
        calc_type_str = data.get("calc_type", CalcType.Plastic.value)
        settings_dict = data.get("settings", {})

        # --- Route to the appropriate factory ---
        factory_map = {
            CalcType.Plastic.value: cls.create_plastic_settings,
            CalcType.Consolidation.value: cls.create_consolidation_settings,
            CalcType.FlowConsolidation.value: cls.create_flow_consolidation_settings,
            CalcType.Safety.value: cls.create_safety_settings,
            CalcType.Dynamic.value: cls.create_dynamic_settings,
            CalcType.DynamicConsolidation.value: cls.create_dynamic_consolidation_settings
        }

        factory = factory_map.get(calc_type_str, cls.create_plastic_settings)
        obj = factory(**settings_dict)
        obj.settings.update(settings_dict)
        return obj

    def __repr__(self) -> str:
        """Provides a unified and informative string representation."""
        return f"<plx.components.StageSettings calc_type='{self.calc_type.value}'>"


# ----------------------------------------------------------------------
# 3) ConstructionStage
# ----------------------------------------------------------------------
class PhaseType(Enum):
    """Enumeration for construction phase types."""
    INITIAL = "Initial"
    EXCAVATION = "Excavation"
    BACKFILL = "Backfill"
    DEWATERING = "Dewatering"
    CONSOLIDATION = "Consolidation"
    LOAD = "Load"
    OTHER = "Other"

ActionTarget = Union[str, PlaxisObject]

class ConstructionStage(PlaxisObject):
    """
    An object representing a single construction phase or stage.

    Includes base information, calculation settings, and dispatch containers
    for structural and soil actions. This object is designed to be
    serialized and sent to a remote scripting service like Plaxis 3D's API.
    """

    def __init__(self,
                 name: str,
                 comment: str,
                 phase_type: PhaseType = PhaseType.OTHER,
                 duration: float = 0.0,
                 settings: Optional[StageSettings] = None,
                 notes: Optional[str] = None,
                 water_table: Optional[WaterLevelTable] = None):
        """
        Initializes a ConstructionStage instance.

        Args:
            name (str): The name of the construction phase.
            phase_type (PhaseType, optional): The type of the phase. Defaults to PhaseType.OTHER.
            duration (float, optional): The duration of the phase in days. Defaults to 0.0.
            settings (Optional[StageSettings], optional): The calculation settings for this phase.
            notes (Optional[str], optional): Optional notes for the phase.
            water_table (Optional[WaterLevelTable], optional): The water table associated with this phase.
        """
        super().__init__(name, comment)
        self._phase_type = phase_type
        self._duration = duration
        self.settings  = settings or StageSettings()
        self._notes = notes
        self._water_table = water_table

        # Dispatch containers
        self._activate : List[ActionTarget] = []
        self._deactivate: List[ActionTarget] = []
        self._excavate : List[SoilBlock] = []
        self._freeze   : List[SoilBlock] = []
        self._thaw     : List[SoilBlock] = []
        self._backfill : List[SoilBlock] = []

    # --- Properties ---

    @property
    def phase_type(self) -> PhaseType:
        """The type of the phase."""
        return self._phase_type

    @property
    def duration(self) -> float:
        """The duration of the phase in days."""
        return self._duration

    @property
    def water_table(self) -> Optional[WaterLevelTable]:
        """The water table associated with this phase."""
        return self._water_table

    @water_table.setter
    def water_table(self, wt: WaterLevelTable):
        self._water_table = wt

    @property
    def notes(self) -> Optional[str]:
        """Optional notes for the phase."""
        return self._notes

    # --- Structure Scheduling ---
    def activate(self, obj: ActionTarget):
        """Adds a structure to the list of objects to be activated in this phase."""
        self._activate.append(obj)

    def deactivate(self, obj: ActionTarget):
        """Adds a structure to the list of objects to be deactivated in this phase."""
        self._deactivate.append(obj)

    # --- Soil Scheduling ---
    def excavate_block(self, blk: SoilBlock):
        """Adds a soil block to be excavated."""
        self._excavate.append(blk)

    def freeze_block(self, blk: SoilBlock):
        """Adds a soil block to be frozen."""
        self._freeze.append(blk)

    def thaw_block(self, blk: SoilBlock):
        """Adds a soil block to be thawed."""
        self._thaw.append(blk)

    def backfill_block(self, blk: SoilBlock):
        """Adds a soil block to be backfilled."""
        self._backfill.append(blk)

    # --- Summary ---
    def summary(self) -> Dict:
        """Returns a dictionary summary of the phase for logging or serialization."""
        return {
            "name": self._name,
            "type": self._phase_type.value,
            "plx_id": self._plx_id,
            "duration": self._duration,
            "settings": self.settings.to_dict(),
            "activate":   [getattr(x, "name", str(x)) for x in self._activate],
            "deactivate": [getattr(x, "name", str(x)) for x in self._deactivate],
            "excavate":   [b.name for b in self._excavate],
            "freeze":     [b.name for b in self._freeze],
            "thaw":       [b.name for b in self._thaw],
            "backfill":   [b.name for b in self._backfill],
            "water_table": self._water_table.label if self._water_table else None,
            "notes": self._notes
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "__version__": getattr(self, "_SERIAL_VERSION", 1),
            "name": self._name,
            "comment": self._comment,
            "phase_type": self._phase_type.value,
            "duration": self._duration,
            "settings": self.settings.to_dict(),
            "notes": self._notes,
            "water_table": (
                self._water_table.label if self._water_table else None
            ),
            
            "activate":   [getattr(x, "name", str(x)) for x in self._activate],
            "deactivate": [getattr(x, "name", str(x)) for x in self._deactivate],
            "excavate":   [blk.name for blk in self._excavate],
            "freeze":     [blk.name for blk in self._freeze],
            "thaw":       [blk.name for blk in self._thaw],
            "backfill":   [blk.name for blk in self._backfill],
            "plx_id": self._plx_id
        }

    # ----------------- Deserialize -----------------
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        registry: Optional[Dict[str, "SoilBlock"]] = None
    ) -> "ConstructionStage":
        registry = registry or {}

        # ---------- 工具函数 ----------
        def _ensure_soil_block(obj: Union["SoilBlock", "PlaxisObject", str]) -> "SoilBlock":
            """
            If obj is already a SoilBlock → return it as is; otherwise, create a placeholder SoilBlock based on the name.
            The return value is always a SoilBlock, meeting the requirements of the static checker.
            """
            if isinstance(obj, SoilBlock):
                return obj
            # 兜底：将任何其它对象或字符串包成占位 SoilBlock
            return SoilBlock(getattr(obj, "name", str(obj)))

        def _resolve(name_or_id: str) -> ActionTarget:
            """resolve 用于结构件激活/停用，可返回 str 或 PlaxisObject"""
            return registry.get(name_or_id, name_or_id)

        # ---------- settings ----------
        settings_obj = StageSettings.from_dict(data["settings"])

        # ---------- 实例化 ----------
        stage = cls(
            name=data["name"],
            comment=data.get("comment", ""),
            phase_type=PhaseType(data.get("phase_type", PhaseType.OTHER.value)),
            duration=float(data.get("duration", 0.0)),
            settings=settings_obj,
            notes=data.get("notes"),
            water_table=(
                WaterLevelTable(data["water_table"])
                if data.get("water_table") else None
            )
        )

        for act in data.get("activate", []):
            stage.activate(_resolve(act))
        for dea in data.get("deactivate", []):
            stage.deactivate(_resolve(dea))

        # ---------- SoilBlock 容器 ----------
        for blk in data.get("excavate", []):
            target_obj = registry.get(blk) or blk
            stage.excavate_block(_ensure_soil_block(target_obj))

        for blk in data.get("freeze", []):
            target_obj = registry.get(blk) or blk
            stage.freeze_block(_ensure_soil_block(target_obj))

        for blk in data.get("thaw", []):
            target_obj = registry.get(blk) or blk
            stage.thaw_block(_ensure_soil_block(target_obj))

        for blk in data.get("backfill", []):
            target_obj = registry.get(blk) or blk
            stage.backfill_block(_ensure_soil_block(target_obj))

        # ---------- plx_id ----------
        stage._plx_id = data.get("plx_id")
        return stage
    
    def __repr__(self) -> str:
        """Provides a unified and informative string representation."""
        plx_id_info = f"plx_id={self._plx_id}" if self._plx_id else "no_plx_id"
        return (f"<plx.components.ConstructionStage name='{self._name}', "
                f"type='{self._phase_type.value}', "
                f"duration={self._duration}d, "
                f"act={len(self._activate)}, excav={len(self._excavate)} | {plx_id_info}>")