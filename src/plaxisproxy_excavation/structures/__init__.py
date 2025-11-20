"""
Convenience re-exports for the top-level package API.

This module exposes the most commonly used classes and enums so users can write:
    from yourpkg import Anchor, PointLoad, SoilBlock, Well

Design notes
############
- Keep imports here lightweight. Heavy submodules should do their own lazy loading.
- When adding new public classes/enums, place them in the appropriate group below
  and remember to append their names to `__all__`.
- Having an explicit `__all__` makes `from yourpkg import *` predictable and helps linters.
"""

# ###########################
# Structural / support system
# ###########################
from .anchor import Anchor, AnchorType
from .beam import Beam
from .embeddedpile import EmbeddedPile
from .retainingwall import RetainingWall, PositiveInterface, NegativeInterface

# ###############
# Ground elements
# ###############
from .soilblock import SoilBlock

# #######################
# Hydrogeology / wells API
# #######################
from .well import WellType, Well

# ####################
# Loads and load enums
# ####################
from .load import (
    # Load configuration / enums
    LoadStage,
    DistributionType,
    SignalType,
    LoadMultiplier,

    # Static loads
    PointLoad,
    LineLoad,
    SurfaceLoad,
    UniformSurfaceLoad,
    XAlignedIncrementSurfaceLoad,
    YAlignedIncrementSurfaceLoad,
    ZAlignedIncrementSurfaceLoad,
    VectorAlignedIncrementSurfaceLoad,
    FreeIncrementSurfaceLoad,
    PerpendicularSurfaceLoad,

    # Dynamic loads (time-dependent)
    DynPointLoad,
    DynLineLoad,
    DynSurfaceLoad,
    DynUniformSurfaceLoad,
    DynXAlignedIncrementSurfaceLoad,
    DynYAlignedIncrementSurfaceLoad,
    DynZAlignedIncrementSurfaceLoad,
    DynVectorAlignedIncrementSurfaceLoad,
    DynFreeIncrementSurfaceLoad,
    DynPerpendicularSurfaceLoad,
)

# #########################
# Explicit public API surface
# #########################
__all__ = (
    # Structural / support system
    "Anchor",
    "AnchorType",
    "Beam",
    "EmbeddedPile",
    "RetainingWall",
    "PositiveInterface",
    "NegativeInterface",

    # Ground elements
    "SoilBlock",

    # Hydrogeology / wells
    "Well",
    "WellType",

    # Load configuration / enums
    "LoadStage",
    "DistributionType",
    "SignalType",
    "LoadMultiplier",

    # Static loads
    "PointLoad",
    "LineLoad",
    "SurfaceLoad",
    "UniformSurfaceLoad",
    "XAlignedIncrementSurfaceLoad",
    "YAlignedIncrementSurfaceLoad",
    "ZAlignedIncrementSurfaceLoad",
    "VectorAlignedIncrementSurfaceLoad",
    "FreeIncrementSurfaceLoad",
    "PerpendicularSurfaceLoad",

    # Dynamic loads
    "DynPointLoad",
    "DynLineLoad",
    "DynSurfaceLoad",
    "DynUniformSurfaceLoad",
    "DynXAlignedIncrementSurfaceLoad",
    "DynYAlignedIncrementSurfaceLoad",
    "DynZAlignedIncrementSurfaceLoad",
    "DynVectorAlignedIncrementSurfaceLoad",
    "DynFreeIncrementSurfaceLoad",
    "DynPerpendicularSurfaceLoad",
)
