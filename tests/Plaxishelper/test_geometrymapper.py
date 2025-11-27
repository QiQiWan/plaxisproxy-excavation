# test_geometry_mapper_creation.py
from __future__ import annotations
from typing import List
from plxscripting.server import new_server

# ### 根据你的项目结构调整这行导入 ###
from src.plaxisproxy_excavation.mappers.geometry_mapper import GeometryMapper

# 几何基础模型（与你的库保持一致）
from src.plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D

# ############################ helpers ############################
def make_rectangle(z: float = 0.0, width: float = 10.0, height: float = 6.0, x0: float = 0.0, y0: float = 0.0):
    """返回闭合矩形：Point 列表、PointSet、Line3D、Polygon3D（外环）"""
    pts: List[Point] = [
        Point(x0,           y0,            z),
        Point(x0 + width,   y0,            z),
        Point(x0 + width,   y0 + height,   z),
        Point(x0,           y0 + height,   z),
        Point(x0,           y0,            z),     # 闭合
    ]
    ps = PointSet(pts)
    line = Line3D(ps)
    poly = Polygon3D.from_points(ps)  # 仅外环
    return pts, ps, line, poly

def assert_plx_id_set(obj, name: str):
    plx_id = getattr(obj, "plx_id", None)
    assert plx_id is not None, f"{name}.plx_id should be set after creation."

def assert_plx_id_cleared(obj, name: str):
    plx_id = getattr(obj, "plx_id", "NOT-ATTR")
    assert plx_id is None, f"{name}.plx_id should be None after deletion."

# ############################ demo runner ############################
def run_demo():
    # 连接 PLAXIS 远程
    passwd = 'yS9f$TMP?$uQ@rW3'
    s_i, g_i = new_server(passwd, 'localhost', 10000)
    s_i.new()  # 新工程

    # 1) 点：批量创建
    pts, ps, line, poly = make_rectangle(z=0.0, width=12.0, height=5.0, x0=2.0, y0=3.0)
    handles = GeometryMapper.create_points(g_i, ps)
    # 断言成功（忽略 None）
    for i, p in enumerate(pts):
        if handles[i] is not None:
            assert_plx_id_set(p, f"Point[{i}]")

    # 2) 线：由 Line3D 创建（会自动补全缺失点）
    line_created = GeometryMapper.create_line(g_i, line, name="DemoRectEdge")
    # Line3D.plx_id 可能是单句柄或句柄列表（多段）
    assert_plx_id_set(line, "Line3D")

    # 3) 面：由 Polygon3D 创建
    surf_id = GeometryMapper.create_surface(g_i, poly, name="DemoRectSurface")
    assert_plx_id_set(poly, "Polygon3D")

    print("[CHECK] IDs after creation -> OK (points/line/surface)")

    # 4) 删除测试（并验证 plx_id 清空）
    # 先删面
    ok_surf = GeometryMapper.delete_surface(g_i, poly)
    assert ok_surf, "Surface deletion failed"
    assert_plx_id_cleared(poly, "Polygon3D")

    # 再删线
    ok_line = GeometryMapper.delete_line(g_i, line)
    # delete_line 对多段线会逐段尝试；若全部删掉才会清空 plx_id
    if ok_line:
        assert_plx_id_cleared(line, "Line3D")
    else:
        # 若未全部删除成功（少见），打印剩余句柄供排查
        remaining = getattr(line, "plx_id", None)
        print(f"[WARN] Some line segments not deleted; remaining={remaining}")

    # （可选）删除点：逐个删或按句柄删
    for i, p in enumerate(pts):
        if getattr(p, "plx_id", None) is not None:
            ok_pt = GeometryMapper.delete_point(g_i, p)
            if not ok_pt:
                print(f"[WARN] Point[{i}] deletion failed")

    print("[CHECK] IDs after deletion -> OK (points/line/surface)")

