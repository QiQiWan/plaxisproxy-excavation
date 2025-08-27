from __future__ import annotations
from typing import List, Tuple
from plxscripting.server import new_server

# ===== 根据你的项目结构调整以下导入 =====
# 几何/材料/钻孔对象
from src.plaxisproxy_excavation.geometry import Point
from src.plaxisproxy_excavation.materials.soilmaterial import MCMaterial
from src.plaxisproxy_excavation.borehole import SoilLayer, BoreholeLayer, Borehole, BoreholeSet

# 材料 & 钻孔集 Mapper
from src.plaxisproxy_excavation.plaxishelper.materialmapper import SoilMaterialMapper
from src.plaxisproxy_excavation.plaxishelper.boreholemapper import BoreholeSetMapper


def assert_plx_id_set(obj, name: str):
    plx_id = getattr(obj, "plx_id", None)
    assert plx_id is not None, f"{name}.plx_id should be set after creation."


def make_materials(g_i):
    """
    创建 3 个常见土体材料并写回 plx_id。
    可按需替换参数（单位以你的库为准）。
    """
    fill = MCMaterial(name="Fill", E_ref=15e6, c_ref=5e3,  phi=25.0, psi=0.0, gamma=18.0, gamma_sat=20.0)
    sand = MCMaterial(name="Sand", E_ref=35e6, c_ref=1e3,  phi=32.0, psi=2.0, gamma=19.0, gamma_sat=21.0)
    clay = MCMaterial(name="Clay", E_ref=12e6, c_ref=15e3, phi=22.0, psi=0.0, gamma=17.0, gamma_sat=19.0)

    SoilMaterialMapper.create_material(g_i, fill)
    SoilMaterialMapper.create_material(g_i, sand)
    SoilMaterialMapper.create_material(g_i, clay)

    for m, n in [(fill, "Fill"), (sand, "Sand"), (clay, "Clay")]:
        assert_plx_id_set(m, n)

    return fill, sand, clay


def build_borehole_set(fill_mat, sand_mat, clay_mat) -> BoreholeSet:
    """
    定义全局 SoilLayer（只描述“层的类型/材料/名称”），
    然后构建 3 个钻孔并给出各自的 BoreholeLayer(绝对高程)。
    第 3 个孔故意不包含 Fill 层，用于测试“缺层→零厚度”的归一化。
    """
    # 1) 全局土层定义（名称要唯一，后续以名称去重/归并）
    sl_fill = SoilLayer(name="Fill",  material=fill_mat)
    sl_sand = SoilLayer(name="Sand",  material=sand_mat)
    sl_clay = SoilLayer(name="Clay",  material=clay_mat)
    soil_layers = [sl_fill, sl_sand, sl_clay]

    # 2) 钻孔 1
    bh1_layers = [
        BoreholeLayer(name="L1_Fill", top_z=0.0,   bottom_z=-1.5, soil_layer=sl_fill),
        BoreholeLayer(name="L1_Sand", top_z=-1.5, bottom_z=-8.0, soil_layer=sl_sand),
        BoreholeLayer(name="L1_Clay", top_z=-8.0, bottom_z=-12.0, soil_layer=sl_clay),
    ]
    bh1 = Borehole(name="BH_1", location=Point(0, 0, 0), ground_level=0.0, layers=bh1_layers, water_head=-2.0)

    # 3) 钻孔 2
    bh2_layers = [
        BoreholeLayer(name="L2_Fill", top_z=0.0,   bottom_z=-2.0, soil_layer=sl_fill),
        BoreholeLayer(name="L2_Sand", top_z=-2.0, bottom_z=-6.0, soil_layer=sl_sand),
        BoreholeLayer(name="L2_Clay", top_z=-6.0, bottom_z=-10.0, soil_layer=sl_clay),
    ]
    bh2 = Borehole(name="BH_2", location=Point(12, 0, 0), ground_level=0.0, layers=bh2_layers, water_head=-1.5)

    # 4) 钻孔 3（故意缺少 Fill 层）
    bh3_layers = [
        BoreholeLayer(name="L3_Sand", top_z=0.0,  bottom_z=-4.0, soil_layer=sl_sand),
        BoreholeLayer(name="L3_Clay", top_z=-4.0, bottom_z=-9.0, soil_layer=sl_clay),
    ]
    bh3 = Borehole(name="BH_3", location=Point(24, 0, 0), ground_level=0.0, layers=bh3_layers, water_head=-1.0)

    # 5) BoreholeSet
    bhset = BoreholeSet(name="Site BH", boreholes=[bh1, bh2, bh3], comment="Demo borehole set")
    return bhset


def print_summary(summary):
    """
    summary: {layer_name: [(top,bottom)_bh0, (top,bottom)_bh1, ...]}
    """
    print("\n=== Zone Summary (top, bottom) per layer per borehole ===")
    layer_names = list(summary.keys())
    for lname in layer_names:
        pairs = summary[lname]
        cells = ", ".join([f"BH{i}:({t:.3g},{b:.3g})" for i, (t, b) in enumerate(pairs)])
        print(f"  - {lname}: {cells}")


def run_demo():
    # ========= 0) 连接 PLAXIS =========
    passwd = "yS9f$TMP?$uQ@rW3"    # ← 根据你的设置修改
    s_i, g_i = new_server("localhost", 10000, password=passwd)
    s_i.new()  # 新工程

    # ========= 1) 材料 =========
    fill_mat, sand_mat, clay_mat = make_materials(g_i)

    # ========= 2) 构建钻孔集 =========
    bhset = build_borehole_set(fill_mat, sand_mat, clay_mat)

    # ========= 3) 一次性导入钻孔集 =========
    # normalize=True 会把所有孔的层序统一化；缺失层将赋零厚度（top==bottom）
    summary = BoreholeSetMapper.create(g_i, bhset, normalize=True)

    # 简单断言：所有 Borehole 和 SoilLayer 都应拿到 plx_id
    for i, bh in enumerate(bhset.boreholes):
        assert_plx_id_set(bh, f"Borehole[{i}]")
    for sl in bhset.unique_soil_layers:
        assert_plx_id_set(sl, f"SoilLayer[{sl.name}]")

    # ========= 4) 打印一份整洁的 Zone 汇总 =========
    print_summary(summary)
    print("\n[CHECK] Borehole set import -> OK")

    # ========= 5) 如需清理（可按需启用）=========
    # BoreholeSetMapper.delete_all(g_i, bhset)
    # print("[CHECK] Borehole set deletion -> OK")


if __name__ == "__main__":
    run_demo()
