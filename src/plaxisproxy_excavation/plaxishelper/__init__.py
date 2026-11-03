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

from .resulttypes import (
    Beam,
    EmbeddedBeam,
    FixedEndAnchor,
    NodeToNodeAnchor,
    Plate,
    RigidBody,
    Soil,
    SurfaceLoad,
    Well,
)

from .phase_properties import (
    PhasePropertyError, set_phase_property, set_phase_properties,
    set_node_to_node_prestress,
)
from .phase_results import PhaseResultError, read_node_to_node_axial_force

__all__ = [
    "PhasePropertyError", "set_phase_property", "set_phase_properties",
    "set_node_to_node_prestress", "PhaseResultError", "read_node_to_node_axial_force",
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
    "Beam",
    "EmbeddedBeam",
    "FixedEndAnchor",
    "NodeToNodeAnchor",
    "Plate",
    "RigidBody",
    "Soil",
    "SurfaceLoad",
    "Well",
]
