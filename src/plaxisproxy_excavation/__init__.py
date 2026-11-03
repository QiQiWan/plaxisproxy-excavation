# plaxisproxy_excavation/__init__.py
# 公开 API

from .excavation import FoundationPit, StructureType, MaterialType
from .geometry import Point, PointSet, Line3D, Polygon3D, Cube, Polyhedron
from .borehole import Borehole, BoreholeSet, BoreholeLayer, SoilLayer
from .components.projectinformation import ProjectInformation
from .components.phase import Phase

# IFC export is optional. Some distributions do not contain this module;
# keep the core excavation API importable in that case.
try:
    from .ifc_exporter import IFCExporter, export_to_ifc
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.ifc_exporter":
        raise

__all__ = [
    # 核心模型
    "FoundationPit",
    "StructureType",
    "MaterialType",
    "ProjectInformation",
    "Phase",
    # 几何
    "Point",
    "PointSet",
    "Line3D",
    "Polygon3D",
    "Cube",
    "Polyhedron",
    # 钻孔
    "Borehole",
    "BoreholeSet",
    "BoreholeLayer",
    "SoilLayer",
]

if "IFCExporter" in globals():
    __all__.extend(["IFCExporter", "export_to_ifc"])
