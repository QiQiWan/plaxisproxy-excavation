"""
example_ifc_export.py  v2.0
============================
完整基坑工程 IFC 示例：
  - 矩形基坑（20m × 15m × 10m 深）
  - 四面地连墙（厚 0.8m，高 10m）
  - 两层水平支撑（矩形截面 H 型钢，0.4m × 0.4m）
  - 预应力锚杆（南北两侧各 2 根）
  - 2 根嵌入桩（坑内立柱桩）
  - 2 个钻孔（含 3 层地层）
  - 完整元数据（项目名、地址、经纬度）

运行：
    cd code/
    PYTHONPATH=src python3 example_ifc_export.py
"""

from plaxisproxy_excavation.excavation import FoundationPit, StructureType
from plaxisproxy_excavation.components.projectinformation import ProjectInformation
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D
from plaxisproxy_excavation.structures.retainingwall import RetainingWall
from plaxisproxy_excavation.structures.anchor import Anchor
from plaxisproxy_excavation.structures.beam import Beam
from plaxisproxy_excavation.structures.embeddedpile import EmbeddedPile
from plaxisproxy_excavation.materials.platematerial import ElasticPlate
from plaxisproxy_excavation.materials.anchormaterial import ElasticAnchor
from plaxisproxy_excavation.materials.beammaterial import ElasticBeam
from plaxisproxy_excavation.materials.pilematerial import ElasticPile
from plaxisproxy_excavation.borehole import Borehole, BoreholeLayer, SoilLayer, BoreholeSet
from plaxisproxy_excavation.ifc_exporter import export_to_ifc

# ─────────────────────────────────────────────────────────────────────────────
# 基坑几何参数
# ─────────────────────────────────────────────────────────────────────────────
X0, X1 = 0.0, 20.0   # 东西方向（m）
Y0, Y1 = 0.0, 15.0   # 南北方向（m）
Z_TOP  =  0.0         # 地面标高
Z_BOT  = -10.0        # 坑底标高
WALL_T =  0.8         # 地连墙厚度（m）

# ─────────────────────────────────────────────────────────────────────────────
# 1. 项目信息
# ─────────────────────────────────────────────────────────────────────────────
proj_info = ProjectInformation(
    title="示例基坑工程 — 上海某地铁车站",
    company="智能建造研究院",
    x_min=X0, y_min=Y0, x_max=X1 + 20, y_max=Y1 + 20,
)

pit = FoundationPit(project_information=proj_info)
pit.excava_depth = abs(Z_BOT)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 材料定义
# ─────────────────────────────────────────────────────────────────────────────
# 地连墙（C30 混凝土，厚 0.8m）
plate_mat = ElasticPlate(
    name="C30混凝土地连墙",
    E=30e6, nu=0.2, d=WALL_T, gamma=25.0
)

# 水平支撑（H 型钢，截面 400×400mm）
beam_mat = ElasticBeam(
    name="H400×400型钢支撑",
    E=210e6, nu=0.3, gamma=78.5,
    width=0.4, height=0.4,
)

# 预应力锚杆（直径 150mm）
anchor_mat = ElasticAnchor(
    name="预应力锚杆φ150",
    EA=80000.0,
)

# 嵌入桩（钻孔灌注桩，直径 800mm）
pile_mat = ElasticPile(
    name="钻孔灌注桩φ800",
    E=30e6, nu=0.2, gamma=25.0,
    diameter=0.8,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. 四面地连墙（俯视图为矩形，顺时针：南→东→北→西）
# ─────────────────────────────────────────────────────────────────────────────
def make_wall(name, pts_2d, z_top=Z_TOP, z_bot=Z_BOT, mat=plate_mat):
    """
    pts_2d: [(x0,y0), (x1,y1)] 墙的两端平面坐标
    生成竖直矩形面（Polygon3D），顶点顺序：左下→右下→右上→左上→闭合
    """
    (x0, y0), (x1, y1) = pts_2d
    ps = PointSet([
        Point(x0, y0, z_top),
        Point(x1, y1, z_top),
        Point(x1, y1, z_bot),
        Point(x0, y0, z_bot),
        Point(x0, y0, z_top),   # 闭合
    ])
    surface = Polygon3D.from_points(ps)
    return RetainingWall(name=name, surface=surface, plate_type=mat)

# 南墙（y=Y0，沿 X 方向）
wall_s = make_wall("南侧地连墙", [(X0, Y0), (X1, Y0)])
# 北墙（y=Y1，沿 X 方向）
wall_n = make_wall("北侧地连墙", [(X0, Y1), (X1, Y1)])
# 西墙（x=X0，沿 Y 方向）
wall_w = make_wall("西侧地连墙", [(X0, Y0), (X0, Y1)])
# 东墙（x=X1，沿 Y 方向）
wall_e = make_wall("东侧地连墙", [(X1, Y0), (X1, Y1)])

for w in [wall_s, wall_n, wall_w, wall_e]:
    pit.add_structure(StructureType.RETAINING_WALLS, w)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 两层水平支撑（矩形截面，架设在四面墙之间）
# ─────────────────────────────────────────────────────────────────────────────
# 第一层支撑（z = -2m）：东西向 3 根 + 南北向 2 根
Z_STRUT1 = -2.0
Z_STRUT2 = -6.0

def add_struts(pit, z_level, mat, prefix):
    """在 z_level 标高添加一层水平支撑（东西向 + 南北向）。"""
    # 东西向（沿 X，间距 5m）
    for i, y in enumerate([Y0 + 2.5, Y0 + 7.5, Y0 + 12.5]):
        beam = Beam(
            name=f"{prefix}-EW{i+1}",
            p_start=Point(X0, y, z_level),
            p_end=Point(X1, y, z_level),
            beam_type=mat,
        )
        pit.add_structure(StructureType.BEAMS, beam)
    # 南北向（沿 Y，间距 5m）
    for i, x in enumerate([X0 + 5.0, X0 + 10.0, X0 + 15.0]):
        beam = Beam(
            name=f"{prefix}-NS{i+1}",
            p_start=Point(x, Y0, z_level),
            p_end=Point(x, Y1, z_level),
            beam_type=mat,
        )
        pit.add_structure(StructureType.BEAMS, beam)

add_struts(pit, Z_STRUT1, beam_mat, "第一层支撑")
add_struts(pit, Z_STRUT2, beam_mat, "第二层支撑")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 预应力锚杆（南北两侧，斜向坑外，倾角 15°）
# ─────────────────────────────────────────────────────────────────────────────
import math
ANCHOR_ANGLE = math.radians(15)   # 倾角
ANCHOR_LEN   = 12.0               # 锚杆长度（m）
DX_ANCHOR    = ANCHOR_LEN * math.cos(ANCHOR_ANGLE)
DZ_ANCHOR    = -ANCHOR_LEN * math.sin(ANCHOR_ANGLE)

# 南侧锚杆（向南延伸，y 减小）
for i, x in enumerate([5.0, 15.0]):
    anchor = Anchor(
        name=f"南侧锚杆-A{i+1}",
        anchor_type=anchor_mat,
        p_start=Point(x, Y0, -3.0),
        p_end=Point(x, Y0 - DX_ANCHOR, -3.0 + DZ_ANCHOR),
    )
    pit.add_structure(StructureType.ANCHORS, anchor)

# 北侧锚杆（向北延伸，y 增大）
for i, x in enumerate([5.0, 15.0]):
    anchor = Anchor(
        name=f"北侧锚杆-A{i+1}",
        anchor_type=anchor_mat,
        p_start=Point(x, Y1, -3.0),
        p_end=Point(x, Y1 + DX_ANCHOR, -3.0 + DZ_ANCHOR),
    )
    pit.add_structure(StructureType.ANCHORS, anchor)

# ─────────────────────────────────────────────────────────────────────────────
# 6. 嵌入桩（坑内立柱桩，支撑水平支撑）
# ─────────────────────────────────────────────────────────────────────────────
for i, (x, y) in enumerate([(10.0, 5.0), (10.0, 10.0)]):
    pile = EmbeddedPile(
        name=f"立柱桩-P{i+1}",
        p_start=Point(x, y, Z_TOP),
        p_end=Point(x, y, -15.0),
        pile_type=pile_mat,
    )
    pit.add_structure(StructureType.EMBEDDED_PILES, pile)

# ─────────────────────────────────────────────────────────────────────────────
# 7. 钻孔地层信息
# ─────────────────────────────────────────────────────────────────────────────
soil_fill = SoilLayer(name="杂填土",   top_z=0.0,  bottom_z=-2.0)
soil_clay = SoilLayer(name="淤泥质黏土", top_z=-2.0, bottom_z=-8.0)
soil_sand = SoilLayer(name="粉细砂",   top_z=-8.0, bottom_z=-20.0)

bh1 = Borehole(name="ZK-1", location=Point(3, 3, 0), ground_level=0.0, water_head=-2.5)
bh1.add_layer(BoreholeLayer("ZK1-填土", 0.0, -2.0, soil_fill))
bh1.add_layer(BoreholeLayer("ZK1-黏土", -2.0, -8.0, soil_clay))
bh1.add_layer(BoreholeLayer("ZK1-砂层", -8.0, -20.0, soil_sand))

bh2 = Borehole(name="ZK-2", location=Point(17, 12, 0), ground_level=0.0, water_head=-3.0)
bh2.add_layer(BoreholeLayer("ZK2-填土", 0.0, -1.5, soil_fill))
bh2.add_layer(BoreholeLayer("ZK2-黏土", -1.5, -9.0, soil_clay))
bh2.add_layer(BoreholeLayer("ZK2-砂层", -9.0, -20.0, soil_sand))

pit.borehole_set = BoreholeSet(name="勘察钻孔组", boreholes=[bh1, bh2])

# ─────────────────────────────────────────────────────────────────────────────
# 8. 导出 IFC
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_path = "example_excavation_v2.ifc"
    export_to_ifc(
        pit,
        file_path=output_path,
        author="EatRice",
        organization="智能建造研究院",
        project_address="上海市浦东新区XX路1号",
        latitude=31.23,
        longitude=121.47,
        elevation=3.5,
    )

    print("\n📦 模型统计：")
    print(f"  地连墙：4 面（厚 {WALL_T}m，高 {abs(Z_BOT)}m）")
    print(f"  水平支撑：12 根（两层，矩形截面 0.4×0.4m）")
    print(f"  锚杆：4 根（倾角 15°，长 {ANCHOR_LEN}m）")
    print(f"  嵌入桩：2 根（φ800，深 15m）")
    print(f"  钻孔：2 个（含 3 层地层）")
    print(f"\n✅ 输出文件：{output_path}")
    print("   可用 BIMvision / FreeCAD / Autodesk Viewer 打开查看")
