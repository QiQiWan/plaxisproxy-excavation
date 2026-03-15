"""
example_ifc_export.py
=====================
演示如何使用 IFCExporter 将 FoundationPit 导出为 IFC 文件。

运行前请确保已安装 ifcopenshell：
    pip install ifcopenshell
"""

from plaxisproxy_excavation.excavation import FoundationPit, StructureType
from plaxisproxy_excavation.components.projectinformation import ProjectInformation
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D
from plaxisproxy_excavation.structures.retainingwall import RetainingWall
from plaxisproxy_excavation.structures.anchor import Anchor
from plaxisproxy_excavation.structures.beam import Beam
from plaxisproxy_excavation.structures.embeddedpile import EmbeddedPile
from plaxisproxy_excavation.materials.platematerial import ElasticPlate
from plaxisproxy_excavation.materials.anchormaterial import ElasticAnchor, AnchorType
from plaxisproxy_excavation.materials.beammaterial import ElasticBeam
from plaxisproxy_excavation.materials.pilematerial import ElasticPile
from plaxisproxy_excavation.borehole import Borehole, BoreholeLayer, SoilLayer, BoreholeSet
from plaxisproxy_excavation.materials.soilmaterial import SoilMaterialFactory, SoilMaterialsType
from plaxisproxy_excavation.ifc_exporter import export_to_ifc


# ─────────────────────────────────────────────────────────────────────────────
# 1. 创建项目信息
# ─────────────────────────────────────────────────────────────────────────────
proj_info = ProjectInformation(
    title="示例基坑工程",
    company="智能建造研究院",
    x_min=0, y_min=0, x_max=30, y_max=20,
)

pit = FoundationPit(project_information=proj_info)
pit.excava_depth = 10.0   # 开挖深度 10m（自动转为负值）


# ─────────────────────────────────────────────────────────────────────────────
# 2. 添加挡土墙（矩形截面 Polygon3D）
# ─────────────────────────────────────────────────────────────────────────────
plate_mat = ElasticPlate(name="C30混凝土板", E=30e6, nu=0.2, d=0.8, gamma=25.0)
pit.add_material("plate_materials", plate_mat)

# 南侧挡土墙：x=0~20m，z=0~-10m
wall_south_pts = PointSet([
    Point(0,  0,  0),
    Point(20, 0,  0),
    Point(20, 0, -10),
    Point(0,  0, -10),
    Point(0,  0,  0),   # 闭合
])
wall_south = RetainingWall(
    name="南侧挡土墙",
    surface=Polygon3D.from_points(wall_south_pts),
    plate_type=plate_mat,
)
pit.add_structure(StructureType.RETAINING_WALLS, wall_south)

# 北侧挡土墙
wall_north_pts = PointSet([
    Point(0,  15,  0),
    Point(20, 15,  0),
    Point(20, 15, -10),
    Point(0,  15, -10),
    Point(0,  15,  0),
])
wall_north = RetainingWall(
    name="北侧挡土墙",
    surface=Polygon3D.from_points(wall_north_pts),
    plate_type=plate_mat,
)
pit.add_structure(StructureType.RETAINING_WALLS, wall_north)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 添加锚杆
# ─────────────────────────────────────────────────────────────────────────────
anchor_mat = ElasticAnchor(name="预应力锚杆", EA=50000.0)
pit.add_material("anchor_materials", anchor_mat)

anchor1 = Anchor(
    name="锚杆-S1",
    anchor_type=anchor_mat,
    p_start=Point(5,  0, -3),
    p_end=Point(5, -8, -3),
)
pit.add_structure(StructureType.ANCHORS, anchor1)

anchor2 = Anchor(
    name="锚杆-S2",
    anchor_type=anchor_mat,
    p_start=Point(15, 0, -3),
    p_end=Point(15, -8, -3),
)
pit.add_structure(StructureType.ANCHORS, anchor2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 添加腰梁（Beam）
# ─────────────────────────────────────────────────────────────────────────────
beam_mat = ElasticBeam(name="H型钢腰梁", E=210e6, nu=0.3, gamma=78.5, diameter=0.3)
pit.add_material("beam_materials", beam_mat)

beam1 = Beam(
    name="腰梁-L1",
    p_start=Point(0,  0, -3),
    p_end=Point(20, 0, -3),
    beam_type=beam_mat,
)
pit.add_structure(StructureType.BEAMS, beam1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 添加嵌入桩
# ─────────────────────────────────────────────────────────────────────────────
pile_mat = ElasticPile(name="钻孔灌注桩", E=30e6, nu=0.2, gamma=25.0, diameter=0.8)
pit.add_material("pile_materials", pile_mat)

pile1 = EmbeddedPile(
    name="桩-P1",
    p_start=Point(10, 7.5,  0),
    p_end=Point(10, 7.5, -15),
    pile_type=pile_mat,
)
pit.add_structure(StructureType.EMBEDDED_PILES, pile1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 添加钻孔地层信息
# ─────────────────────────────────────────────────────────────────────────────
soil_fill = SoilLayer(name="杂填土", top_z=0.0, bottom_z=-2.0)
soil_clay = SoilLayer(name="粉质黏土", top_z=-2.0, bottom_z=-8.0)
soil_sand = SoilLayer(name="中砂", top_z=-8.0, bottom_z=-20.0)

bh1 = Borehole(name="ZK-1", location=Point(5, 5, 0), ground_level=0.0, water_head=-3.0)
bh1.add_layer(BoreholeLayer("ZK1-填土", 0.0, -2.0, soil_fill))
bh1.add_layer(BoreholeLayer("ZK1-黏土", -2.0, -8.0, soil_clay))
bh1.add_layer(BoreholeLayer("ZK1-砂层", -8.0, -20.0, soil_sand))

bh2 = Borehole(name="ZK-2", location=Point(15, 10, 0), ground_level=0.0, water_head=-3.5)
bh2.add_layer(BoreholeLayer("ZK2-填土", 0.0, -1.5, soil_fill))
bh2.add_layer(BoreholeLayer("ZK2-黏土", -1.5, -9.0, soil_clay))
bh2.add_layer(BoreholeLayer("ZK2-砂层", -9.0, -20.0, soil_sand))

pit.borehole_set = BoreholeSet(name="钻孔组", boreholes=[bh1, bh2])


# ─────────────────────────────────────────────────────────────────────────────
# 7. 导出 IFC
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_path = "example_excavation.ifc"
    export_to_ifc(
        pit,
        file_path=output_path,
        author="EatRice",
        organization="智能建造研究院",
    )
    print(f"\n✅ 导出完成: {output_path}")
    print("可用 Autodesk Viewer / BIMvision / FreeCAD 打开查看。")
