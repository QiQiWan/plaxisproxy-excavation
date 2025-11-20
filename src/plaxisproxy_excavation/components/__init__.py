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

from .projectinformation import Units, ProjectInformation
from .watertable import WaterLevel, WaterLevelTable

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

    # Water level
    "WaterLevel",
    "WaterLevelTable",

    # Project Information
    "Units",
    "ProjectInformation"
]