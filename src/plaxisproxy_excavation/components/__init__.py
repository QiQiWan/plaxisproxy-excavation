from .mesh import Mesh
from .phase import Phase

from .phasesettings import (
    StageSettingsBase,
    PlasticStageSettings,
    ConsolidationStageSettings,
    FullyCoupledStageSettings,
    DynamicStageSettings,
    DynamicWithConsolidationStageSettings,
    SafetyStageSettings,
    StageSettingsFactory,
    PoreCalType,
    SolverType,
    CalcType,
    LoadType
)

# Explicitly define public interfaces to avoid exposing internal module details during import.
__all__ = [
    # Core classes
    "Mesh",
    "Phase",
    # Phase setting classes
    "StageSettingsBase",
    "PlasticStageSettings",
    "ConsolidationStageSettings",
    "FullyCoupledStageSettings",
    "DynamicStageSettings",
    "DynamicWithConsolidationStageSettings",
    "SafetyStageSettings",
    "StageSettingsFactory",
]