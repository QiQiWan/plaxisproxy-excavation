# plaxisproxy_excavation/__init__.py
# 公开 API

from .excavation import FoundationPit, StructureType, MaterialType
from .geometry import Point, PointSet, Line3D, Polygon3D, Cube, Polyhedron
from .borehole import Borehole, BoreholeSet, BoreholeLayer, SoilLayer
from .components.projectinformation import ProjectInformation
from .components.phase import Phase

# IFC 导出（需要 ifcopenshell）
from .ifc_exporter import IFCExporter, export_to_ifc

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
    # IFC 导出
    "IFCExporter",
    "export_to_ifc",
]
