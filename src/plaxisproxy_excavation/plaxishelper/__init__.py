"""
Convenience re-exports for plaxishelper submodules.

Importing from plaxisproxy_excavation.plaxishelper gives access to the most
commonly used mapper classes and utilities in this package.
"""

# Geometry / project / boreholes
from .geometrymapper import GeometryMapper
from .projectinfomapper import ProjectInformationMapper
from .boreholemapper import BoreholeSetMapper

# Materials
from .materialmapper import (
    SoilMaterialMapper,
    PlateMaterialMapper,
    BeamMaterialMapper,
    PileMaterialMapper,
    AnchorMaterialMapper,
)

# Structures
from .structuremapper import (
    AnchorMapper,
    BeamMapper,
    EmbeddedPileMapper,
    RetainingWallMapper,
    WellMapper,
    SoilBlockMapper,
)

# Loads
from .loadmapper import LoadMapper, LoadMultiplierMapper

# Water table / mesh / monitors / phases
from .watertablemapper import WaterTableMapper
from .meshmapper import MeshMapper
from .monitormapper import MonitorMapper
from .phasemapper import PhaseMapper

# Runner (session orchestration)
from .plaxisrunner import PlaxisRunner
from .plaxisoutput import PlaxisOutput

__all__ = [
    # Geometry / project / boreholes
    "GeometryMapper",
    "ProjectInformationMapper",
    "BoreholeSetMapper",
    # Materials
    "SoilMaterialMapper",
    "PlateMaterialMapper",
    "BeamMaterialMapper",
    "PileMaterialMapper",
    "AnchorMaterialMapper",
    # Structures
    "AnchorMapper",
    "BeamMapper",
    "EmbeddedPileMapper",
    "RetainingWallMapper",
    "WellMapper",
    "SoilBlockMapper",
    # Loads
    "LoadMapper",
    "LoadMultiplierMapper",
    # Water table / mesh / monitors / phases
    "WaterTableMapper",
    "MeshMapper",
    "MonitorMapper",
    "PhaseMapper",
    # Runner
    "PlaxisRunner",
    # Output
    "PlaxisOutput",
    "ResultDomain",
    "ResultLocation",
    "SoilResult",
    "PlateResult",
    "BeamResult",
    "EmbeddedBeamResult",
    "AnchorResult",
    "WellResult"
]
