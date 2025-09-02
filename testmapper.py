# -*- coding: utf-8 -*-
"""
Demo: static & dynamic loads with the updated Base+Dynamic model (mapper-aligned)

Rules:
- Dynamic loads keep `base` and only ATTACH multipliers to the SAME PLAXIS handle.
- Geometry is created via GeometryMapper (the LoadMapper is geometry-aware too).
- One static load per geometry object (point/line/surface). A guard enforces this.
- Deletion:
    * static -> delete the PLAXIS handle
    * dynamic -> clear Multiplierx/y/z on its base (handle remains)
"""

from __future__ import annotations
from typing import Dict, Any

from plxscripting.server import new_server

# ---- adjust paths if needed ----
from src.plaxisproxy_excavation.structures.load import (
    DistributionType, SignalType, LoadMultiplierKey,
    LoadMultiplier,
    PointLoad, LineLoad, SurfaceLoad, UniformSurfaceLoad,
)
from src.plaxisproxy_excavation.plaxishelper.loadmapper import LoadMapper, LoadMultiplierMapper
from src.plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D
from src.plaxisproxy_excavation.plaxishelper.geometrymapper import GeometryMapper


# ---------- guard: forbid >1 static load on the same geometry ----------
class _StaticLoadGuard:
    """Keep a registry: geometry_id -> load_name. Raise if duplicated."""
    def __init__(self) -> None:
        self._reg: Dict[Any, str] = {}

    @staticmethod
    def _gid(geom: Any) -> Any:
        # Prefer stable `id` field if your geometry carries one; fallback to Python id()
        return getattr(geom, "id", None) or id(geom)

    def ensure_free(self, geom: Any, load_name: str) -> None:
        k = self._gid(geom)
        if k in self._reg:
            raise RuntimeError(f"Geometry already has a static load: '{self._reg[k]}'; "
                               f"refuse to create '{load_name}' on the same geometry.")
        self._reg[k] = load_name

    def forget(self, geom: Any) -> None:
        k = self._gid(geom)
        self._reg.pop(k, None)


def main():
    # 1) connect
    s_i, g_i = new_server("localhost", 10000, password="yS9f$TMP?$uQ@rW3")
    s_i.new()

    guard = _StaticLoadGuard()

    # 2) geometry (created via GeometryMapper; mapper will also ensure on demand)
    P1, P2, P3, P4 = Point(0,0,0), Point(10,0,0), Point(10,8,0), Point(0,8,0)
    GeometryMapper.create_point(g_i, P1)
    GeometryMapper.create_point(g_i, P2)
    GeometryMapper.create_point(g_i, P3)
    GeometryMapper.create_point(g_i, P4)

    L12 = Line3D(PointSet([P1, P2]))
    L13 = Line3D(PointSet([P1, P3]))
    GeometryMapper.create_line(g_i, L12)
    GeometryMapper.create_line(g_i, L13)

    Poly = Polygon3D.from_points(PointSet([P1, P2, P3, P4]))
    GeometryMapper.create_surface(g_i, Poly)

    # 3) STATIC loads (ONE per geometry)
    # 3.1 Point @P3
    guard.ensure_free(P3, "PL_static")
    pl_static = PointLoad(
        name="PL_static", comment="static point load", point=P3,
        Fx=0.0, Fy=0.0, Fz=-100.0, Mx=0.0, My=0.0, Mz=0.0,
    )
    LoadMapper.create(g_i, pl_static)

    # 3.2 Line uniform on L12
    guard.ensure_free(L12, "LL_static_uniform")
    ll_static_uniform = LineLoad(
        name="LL_static_uniform", comment="static uniform line load", line=L12,
        distribution=DistributionType.UNIFORM, qx=0.0, qy=0.0, qz=-8.0,
    )
    LoadMapper.create(g_i, ll_static_uniform)

    # 3.3 Line linear on L13 (different line geometry -> allowed)
    guard.ensure_free(L13, "LL_static_linear")
    ll_static_linear = LineLoad(
        name="LL_static_linear", comment="static linear line load", line=L13,
        distribution=DistributionType.LINEAR, qx=0.0, qy=0.0, qz=-5.0,
        qx_end=0.0, qy_end=0.0, qz_end=-12.0,
    )
    LoadMapper.create(g_i, ll_static_linear)

    # 3.4 Surface uniform on Poly
    guard.ensure_free(Poly, "SL_static_uniform")
    sl_static_uniform = UniformSurfaceLoad(
        name="SL_static_uniform", comment="static uniform surface load", surface=Poly,
        sigmax=0.0, sigmay=0.0, sigmaz=-20.0,
    )
    LoadMapper.create(g_i, sl_static_uniform)

    # NOTE: If you try to place another static surface load on the same `Poly`, the guard will block it:
    # guard.ensure_free(Poly, "SL_static_perp")  # -> raises RuntimeError

    # 4) multipliers (optional to pre-create; mapper can auto-create on attach)
    mul_h_5Hz = LoadMultiplier(
        name="Mul_H_5Hz", comment="harmonic 5 Hz", signal_type=SignalType.HARMONIC,
        amplitude=1.0, phase=0.0, frequency=5.0,
    )
    LoadMultiplierMapper.create(g_i, mul_h_5Hz)

    mul_table = LoadMultiplier(
        name="Mul_Table_Ramp", comment="table ramp 0→1", signal_type=SignalType.TABLE,
        table_data=[(0.0, 0.0), (1.0, 1.0), (2.0, 1.0)],
    )
    LoadMultiplierMapper.create(g_i, mul_table)

    # 5) DYNAMIC loads (clone via create_dyn, then mapper attaches multipliers on base)
    pl_dyn = pl_static.create_dyn(
        name="PL_dyn", comment="dynamic point (from PL_static)",
        multiplier={LoadMultiplierKey.Fz: mul_h_5Hz},
    )
    LoadMapper.create(g_i, pl_dyn)  # attaches Multiplierz on the PL_static handle

    ll_dyn_uniform = ll_static_uniform.create_dyn(
        name="LL_dyn_uniform", comment="dynamic uniform line",
        multiplier={LoadMultiplierKey.Z: mul_h_5Hz},
    )
    LoadMapper.create(g_i, ll_dyn_uniform)

    ll_dyn_linear = ll_static_linear.create_dyn(
        name="LL_dyn_linear", comment="dynamic linear line",
        multiplier={LoadMultiplierKey.Z: mul_table},
    )
    LoadMapper.create(g_i, ll_dyn_linear)

    sl_dyn_uniform = sl_static_uniform.create_dyn(
        name="SL_dyn_uniform", comment="dynamic uniform surface",
        multiplier={LoadMultiplierKey.Z: mul_table},
    )
    LoadMapper.create(g_i, sl_dyn_uniform)

    print("✅ Static loads created; dynamic multipliers attached to the same base handles.")

    # 6) deletion examples (use the unified mapper API)
    # 6.1 delete a dynamic: clears multipliers on its base; base object remains
    LoadMapper.delete(g_i, sl_dyn_uniform)

    # 6.2 delete a static: removes the PLAXIS handle; also release guard
    LoadMapper.delete(g_i, ll_static_uniform)
    guard.forget(L12)  # free the geometry so you could place another static load later if needed

    print("🧹 Deleted SL_dyn_uniform (multipliers cleared) and LL_static_uniform (handle deleted).")


if __name__ == "__main__":
    main()
