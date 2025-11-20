from math import ceil
from typing import List, Tuple, Iterable, Any, Dict, Optional

try:
    from plaxis_config import HOST, PORT, PASSWORD
except:
    HOST = "localhost"
    PORT = 10000
    PASSWORD = "yS9f$TMP?$uQ@rW3"

# Runner / Builder / Container
from plaxisproxy_excavation.plaxishelper.plaxisrunner import PlaxisRunner
from plaxisproxy_excavation.builder import ExcavationBuilder
from plaxisproxy_excavation.excavation import FoundationPit, StructureType  # <-- NEW: StructureType

# Core components
from plaxisproxy_excavation.components.projectinformation import ProjectInformation, Units
from plaxisproxy_excavation.components.phase import Phase
from plaxisproxy_excavation.components.phasesettings import PlasticStageSettings, LoadType
from plaxisproxy_excavation.components.watertable import WaterLevel, WaterLevelTable

# Boreholes & materials
from plaxisproxy_excavation.borehole import SoilLayer, BoreholeLayer, Borehole, BoreholeSet
from plaxisproxy_excavation.materials.soilmaterial import SoilMaterialFactory, SoilMaterialsType, MCGWType, MCGwSWCC

# Geometry
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D

# Structure materials & structures
from plaxisproxy_excavation.materials.platematerial import ElasticPlate
from plaxisproxy_excavation.materials.beammaterial import ElasticBeam  # used here for horizontal braces
from plaxisproxy_excavation.structures.retainingwall import RetainingWall
from plaxisproxy_excavation.structures.beam import Beam
from plaxisproxy_excavation.structures.well import Well, WellType


# ############################# geometry helpers #############################

def rect_wall_x(x: float, y0: float, y1: float, z_top: float, z_bot: float) -> Polygon3D:
    pts = [
        Point(x, y0, z_top), Point(x, y1, z_top),
        Point(x, y1, z_bot), Point(x, y0, z_bot),
        Point(x, y0, z_top),
    ]
    return Polygon3D.from_points(PointSet(pts))

def rect_wall_y(y: float, x0: float, x1: float, z_top: float, z_bot: float) -> Polygon3D:
    pts = [
        Point(x0, y, z_top), Point(x1, y, z_top),
        Point(x1, y, z_bot), Point(x0, y, z_bot),
        Point(x0, y, z_top),
    ]
    return Polygon3D.from_points(PointSet(pts))

def line_2pts(p0: Tuple[float, float, float], p1: Tuple[float, float, float]) -> Line3D:
    a = Point(*p0); b = Point(*p1)
    return Line3D(PointSet([a, b]))


# ############################# wells helpers #############################
def _poly_area_sign(xy):
    a = 0.0
    for (x1,y1),(x2,y2) in zip(xy, xy[1:]+xy[:1]):
        a += (x1*y2 - x2*y1)
    return 1 if a > 0 else -1 if a < 0 else 0

def _point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
        if ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1):
            inside = not inside
    return inside

def _dist_point_to_segment(px, py, x1, y1, x2, y2):
    vx, vy = x2-x1, y2-y1
    wx, wy = px-x1, py-y1
    c1 = vx*wx + vy*wy
    if c1 <= 0: return ((px-x1)**2 + (py-y1)**2)**0.5
    c2 = vx*vx + vy*vy
    if c2 <= c1: return ((px-x2)**2 + (py-y2)**2)**0.5
    b = c1 / (c2 + 1e-12)
    bx, by = x1 + b*vx, y1 + b*vy
    return ((px-bx)**2 + (py-by)**2)**0.5

def _min_dist_to_edges(pt, poly):
    x, y = pt
    dmin = 1e18
    for i in range(len(poly)):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
        dmin = min(dmin, _dist_point_to_segment(x,y,x1,y1,x2,y2))
    return dmin

def wells_on_polygon_edges(prefix, poly_xy, z_top, z_bot, q_well, spacing, clearance):
    wells = []
    sign = _poly_area_sign(poly_xy)  # 逆时针为 +1
    for i in range(len(poly_xy)):
        x1,y1 = poly_xy[i]
        x2,y2 = poly_xy[(i+1)%len(poly_xy)]
        ex, ey = x2-x1, y2-y1
        elen = max((ex**2+ey**2)**0.5, 1e-9)
        nx, ny =  ey/elen, -ex/elen      # 外法线
        if sign > 0:                     # 逆时针：内法线取反
            nx, ny = -nx, -ny
        nseg = max(1, int(elen/spacing))
        for k in range(nseg+1):
            t = k/float(nseg)
            px = x1 + t*ex + nx*clearance
            py = y1 + t*ey + ny*clearance
            wells.append(Well(
                name=f"{prefix}_E_{len(wells)+1}",
                line=line_2pts((px, py, z_top), (px, py, z_bot)),
                well_type=WellType.Extraction,
                q_well=q_well,
                h_min=z_bot,
            ))
    return wells

def wells_grid_in_polygon(prefix, poly_xy, z_top, z_bot, q_well, dx, dy, margin):
    xs = [p[0] for p in poly_xy]; ys = [p[1] for p in poly_xy]
    xi, xf = min(xs), max(xs); yi, yf = min(ys), max(ys)
    wells = []
    nx = max(1, int((xf-xi)/dx))
    ny = max(1, int((yf-yi)/dy))
    for ix in range(nx+1):
        x = xi + (xf - xi) * ix / max(1, nx)
        for iy in range(ny+1):
            y = yi + (yf - yi) * iy / max(1, ny)
            if not _point_in_polygon((x,y), poly_xy): 
                continue
            if _min_dist_to_edges((x,y), poly_xy) < margin: 
                continue
            wells.append(Well(
                name=f"{prefix}_G_{len(wells)+1}",
                line=line_2pts((x, y, z_top), (x, y, z_bot)),
                well_type=WellType.Extraction,
                q_well=q_well,
                h_min=z_bot,
            ))
    return wells

def get_well_positions(file_name: str, z_top, z_bot, q_well) -> List[Well]:
    import pandas as pd
    wells = []
    data = pd.read_excel(file_name)
    posi_col_name = ["Index", "X (m)", "Y (m)"]
    selected_data = data[posi_col_name]
    for index, row in selected_data.iterrows():
        x, y = row[1] - 115, row[2]- 16.5
        wells.append(Well(
            name=f"G_{index}_{x}_{y}",
            line=line_2pts((x, y, z_top), (x, y, z_bot)),
            well_type=WellType.Extraction,
            q_well=q_well,
            h_min=z_bot,
        ))
    return wells
    

def layout_wells_with_limit(
    prefix: str,
    poly_xy,
    z_top: float,
    z_bot: float,
    q_well: float,
    edge_spacing: float,
    grid_dx: float,
    grid_dy: float,
    clearance: float,
    margin: float,
    max_wells: int = 50,
    dedupe_tol: float = 1e-6,
):
    """
    基于多边形 poly_xy 生成降水井，严格限制总井数 ≤ max_wells。
    """
    spacing_used = max(0.5, float(edge_spacing))
    edge_wells = wells_on_polygon_edges(prefix, poly_xy, z_top, z_bot, q_well, spacing_used, clearance)

    safety = 0
    while len(edge_wells) > max_wells and safety < 8:
        factor = ceil(len(edge_wells) / max_wells)
        spacing_used *= max(2, factor)
        edge_wells = wells_on_polygon_edges(prefix, poly_xy, z_top, z_bot, q_well, spacing_used, clearance)
        safety += 1

    if len(edge_wells) > max_wells:
        step = ceil(len(edge_wells) / max_wells)
        edge_wells = [edge_wells[i] for i in range(0, len(edge_wells), step)][:max_wells]

    remaining = max(0, max_wells - len(edge_wells))
    grid_selected = []
    if remaining > 0:
        grid_all = wells_grid_in_polygon(prefix, poly_xy, z_top, z_bot, q_well, grid_dx, grid_dy, margin)
        if len(grid_all) > remaining:
            step = ceil(len(grid_all) / remaining)
            grid_selected = [grid_all[i] for i in range(0, len(grid_all), step)][:remaining]
        else:
            grid_selected = grid_all

    all_wells = edge_wells + grid_selected
    wells_unique, wells_dupes, _ = dedupe_wells_by_line(all_wells, tol=dedupe_tol, keep="first")

    if len(wells_unique) > max_wells:
        wells_unique = wells_unique[:max_wells]

    stats = {
        "edge_spacing_used": spacing_used,
        "edge_count": len(edge_wells),
        "grid_count": len(grid_selected),
        "unique": len(wells_unique),
        "dupes": len(wells_dupes),
    }
    return wells_unique, stats


# ############################# dedupe (by line) #############################

def _as_xyz(p: Any) -> Tuple[float, float, float]:
    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return float(p.x), float(p.y), float(p.z)
    return float(p[0]), float(p[1]), float(p[2])

def _extract_line_endpoints(line: Any) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    pts = getattr(line, "points", None)
    if pts and len(pts) >= 2:
        return _as_xyz(pts[0]), _as_xyz(pts[1])
    ps = getattr(line, "point_set", None) or getattr(line, "_point_set", None)
    inner = getattr(ps, "points", None) or getattr(ps, "_points", None) if ps else None
    if inner and len(inner) >= 2:
        return _as_xyz(inner[0]), _as_xyz(inner[1])
    start, end = getattr(line, "start", None), getattr(line, "end", None)
    if start is not None and end is not None:
        return _as_xyz(start), _as_xyz(end)
    it = list(iter(line))
    if len(it) >= 2:
        return _as_xyz(it[0]), _as_xyz(it[1])
    raise AttributeError("Cannot extract endpoints from the given line object.")

def _q(v: float, tol: float) -> int:
    return int(round(v / tol))

def _line_key(line: Any, tol: float) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    (x0, y0, z0), (x1, y1, z1) = _extract_line_endpoints(line)
    a = (_q(x0, tol), _q(y0, tol), _q(z0, tol))
    b = (_q(x1, tol), _q(y1, tol), _q(z1, tol))
    return (a, b) if a <= b else (b, a)

def dedupe_wells_by_line(
    wells: Iterable[Any],
    tol: float = 1e-6,
    keep: str = "first",
) -> Tuple[List[Any], List[Any], Dict[Tuple[Tuple[int,int,int], Tuple[int,int,int]], Any]]:
    key_to_master: Dict[Tuple[Tuple[int,int,int], Tuple[int,int,int]], Any] = {}
    unique: List[Any] = []
    dupes: List[Any] = []
    if keep not in ("first", "last"):
        raise ValueError("keep must be 'first' or 'last'")
    for w in wells:
        line = getattr(w, "line", w)
        k = _line_key(line, tol)
        if k in key_to_master:
            if keep == "last":
                prev = key_to_master[k]
                if prev in unique:
                    unique.remove(prev); dupes.append(prev)
                key_to_master[k] = w; unique.append(w)
            else:
                dupes.append(w)
        else:
            key_to_master[k] = w; unique.append(w)
    return unique, dupes, key_to_master


# ############################# soil overrides helper #############################

def mk_soil_overrides(names: Iterable[str],
                      active: Optional[bool] = None,
                      deformable: Optional[bool] = None) -> Dict[str, Dict[str, bool]]:
    overrides: Dict[str, Dict[str, bool]] = {}
    for nm in names:
        opts: Dict[str, bool] = {}
        if active is not None:
            opts["active"] = bool(active)
        if deformable is not None:
            opts["deformable"] = bool(deformable)
        if opts:
            overrides[nm] = opts
    return overrides


# ############################# Set wells ##############################
def make_random_flows(
    wells: Iterable[Any],
    base: float = 120.0,
    jitter: float = 0.15,
    *,
    seed: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    round_to: Optional[float] = None,
) -> Dict[Any, float]:
    """
    Build {well_object: flow} with small random variation around `base`.
    - wells: iterable of existing Well objects
    - base: target flow (e.g., m^3/day)
    - jitter: fraction in [0, 1]; each flow = base * (1 ± jitter), uniform
    - seed: optional RNG seed for reproducibility
    - min_value / max_value: optional clipping of generated flows
    - round_to: optional rounding step (e.g., 1.0 → nearest 1 m^3/day)

    Returns a dict keyed by the Well objects you passed in.
    """
    import random
    if seed is not None:
        random.seed(seed)

    flows: Dict[Any, float] = {}
    for w in wells:
        # uniform factor in [1 - jitter, 1 + jitter]
        f = 1.0 + (2.0 * random.random() - 1.0) * max(0.0, float(jitter))
        q = float(base) * f

        if min_value is not None and q < min_value:
            q = float(min_value)
        if max_value is not None and q > max_value:
            q = float(max_value)
        if round_to and round_to > 0:
            q = round(q / round_to) * round_to

        flows[w] = q

    return flows

# ############################# assemble pit #############################

def assemble_pit(runner: Optional[PlaxisRunner] = None) -> FoundationPit:
    """Rectangular pit with D-walls, two brace levels, wells, and phased actions."""

    # 1) Project information (XY = 50 x 80)
    proj = ProjectInformation(
        title="Rectangular Pit – DWall + Braces + Wells",
        company="Demo",
        dir=".",
        file_name="demo_excavation.p3d",
        comment="Builder/PhaseMapper demo",
        model="3D",
        element="10-noded",
        length_unit=Units.Length.M,
        force_unit=Units.Force.KN,
        stress_unit=Units.Stress.KPA,
        time_unit=Units.Time.DAY,
        gamma_water=9.81,
        x_min=-345, x_max=345, 
        y_min=-50, y_max=50,  
    )

    pit = FoundationPit(project_information=proj)

    # Discover split soils (optional)
    excav_names: List[str] = []
    remain_names: List[str] = []
    if runner is not None:
        try:
            excav_names = runner.get_excavation_soil_names(prefer_volume=True)
            remain_names = runner.get_remaining_soil_names(prefer_volume=True)
            print(f"[MAPPER] excav soils: {excav_names}")
            print(f"[MAPPER] remain soils: {remain_names}")
        except Exception as ex:
            print(f"[MAPPER] soil split discovery failed: {ex}")

    # # 2) Soil materials

    # 统一说明：
    # - 所有刚度/强度均为 kPa（kN/m²）
    # - G0_ref 已按土类推荐值给定（≥ Eur/[2(1+ν)]）
    # - gamma07 命名与 PLAXIS 一致；若你的工厂类用其它命名，请映射到 PLAXIS 的 gamma07

    #region Soils definitions
    # layer_001: 11_杂填土
    layer_001 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="11_杂填土",
        gamma=18.5,
        nu=0.30,
        E=1.875e3,        # 1875 kPa
        E_oed=1.500e3,    # 1500 kPa
        E_ur=5.625e3,     # 5625 kPa
        G0_ref=3.245e3,   # ≈ 1.5 * Eur/[2(1+ν)]
        gamma07=5e-4,
        m=2.0,
        P_ref=100.0,
        c=8.0,            # kPa
        phi=10.0,
        psi=0.0,
    )

    # layer_002: 12_素填土
    layer_002 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="12_素填土",
        gamma=18.0,
        nu=0.30,
        E=3.125e3,
        E_oed=2.500e3,
        E_ur=9.375e3,
        G0_ref=5.409e3,
        gamma07=5e-4,
        m=2.0,
        P_ref=100.0,
        c=10.0,
        phi=12.0,
        psi=0.0,
    )

    # layer_003: 22_砂质粉土
    layer_003 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="22_砂质粉土",
        gamma=19.0,
        nu=0.25,
        E=8.750e3,
        E_oed=7.000e3,
        E_ur=2.625e4,
        G0_ref=3.150e4,
        gamma07=1e-4,
        m=2.0,
        P_ref=100.0,
        c=6.0,
        phi=26.0,
        psi=0.0,
    )

    # layer_004: 32_砂质粉土
    layer_004 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="32_砂质粉土",
        gamma=19.2,
        nu=0.25,
        E=4.375e3,
        E_oed=3.500e3,
        E_ur=1.3125e4,
        G0_ref=1.575e4,
        gamma07=1e-4,
        m=2.0,
        P_ref=100.0,
        c=5.0,
        phi=26.0,
        psi=0.0,
    )

    # layer_005: 35_粉砂夹粉土
    layer_005 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="35_粉砂夹粉土",
        gamma=19.3,
        nu=0.25,
        E=1.500e4,
        E_oed=1.200e4,
        E_ur=4.500e4,
        G0_ref=5.400e4,
        gamma07=1e-4,
        m=5.0,
        P_ref=100.0,
        c=5.0,
        phi=29.0,
        psi=0.0,
    )

    # layer_006: 37_砂质粉土夹淤泥质粉质黏土
    layer_006 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="37_砂质粉土夹淤泥质粉质黏土",
        gamma=18.8,
        nu=0.25,
        E=8.125e3,
        E_oed=6.500e3,
        E_ur=2.4375e4,
        G0_ref=2.925e4,
        gamma07=1e-4,
        m=2.0,
        P_ref=100.0,
        c=6.0,
        phi=24.0,
        psi=0.0,
    )

    # layer_007: 62_淤泥质粉质黏土
    layer_007 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="62_淤泥质粉质黏土",
        gamma=17.0,
        nu=0.30,          # 由 0.35 调整为 0.30
        E=5.000e3,
        E_oed=4.000e3,
        E_ur=1.500e4,
        G0_ref=8.654e3,
        gamma07=5e-4,
        m=1.6,
        P_ref=100.0,
        c=20.0,
        phi=11.0,
        psi=0.0,
    )

    # layer_008: 62t_淤泥质粉质黏土夹粉土
    layer_008 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="62t_淤泥质粉质黏土夹粉土",
        gamma=17.7,
        nu=0.30,
        E=3.750e3,
        E_oed=3.000e3,
        E_ur=1.125e4,
        G0_ref=6.490e3,
        gamma07=5e-4,
        m=1.8,
        P_ref=100.0,
        c=10.0,
        phi=14.0,
        psi=0.0,
    )

    # layer_009: 63_粉砂夹粉土
    layer_009 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="63_粉砂夹粉土",
        gamma=19.1,
        nu=0.25,
        E=1.250e4,
        E_oed=1.000e4,
        E_ur=3.750e4,
        G0_ref=4.500e4,
        gamma07=1e-4,
        m=5.5,
        P_ref=100.0,
        c=5.0,
        phi=28.0,
        psi=0.0,
    )

    # layer_010: 81_黏土
    layer_010 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="81_黏土",
        gamma=17.5,
        nu=0.30,
        E=3.125e3,
        E_oed=2.500e3,
        E_ur=8.750e3,
        G0_ref=5.048e3,
        gamma07=5e-4,
        m=1.4,
        P_ref=100.0,
        c=18.0,
        phi=12.0,
        psi=0.0,
    )

    # layer_011: 102_粉质黏土
    layer_011 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="102_粉质黏土",
        gamma=18.5,
        nu=0.30,          # 由 0.35 调整为 0.30
        E=6.250e3,
        E_oed=5.000e3,
        E_ur=2.0625e4,
        G0_ref=1.1899e4,
        gamma07=5e-4,
        m=3.0,
        P_ref=100.0,
        c=20.0,
        phi=15.0,
        psi=0.0,
    )

    # layer_012: 111_粉质黏土
    layer_012 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="111_粉质黏土",
        gamma=19.3,
        nu=0.30,          # 由 0.35 调整为 0.30
        E=1.375e4,
        E_oed=1.100e4,
        E_ur=4.125e4,
        G0_ref=2.3798e4,
        gamma07=5e-4,
        m=5.5,
        P_ref=100.0,
        c=21.0,
        phi=15.4,
        psi=0.0,
    )

    # layer_013: 121_细砂
    layer_013 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="121_细砂",
        gamma=19.6,
        nu=0.25,
        E=2.1875e4,
        E_oed=1.750e4,
        E_ur=6.5625e4,
        G0_ref=7.875e4,
        gamma07=1e-4,
        m=6.0,
        P_ref=100.0,
        c=4.0,
        phi=32.0,
        psi=2.0,
    )

    # layer_014: 132_粉质黏土
    layer_014 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="132_粉质黏土",
        gamma=19.4,
        nu=0.30,          # 由 0.35 调整为 0.30
        E=1.0625e4,
        E_oed=8.500e3,
        E_ur=3.1875e4,
        G0_ref=1.8389e4,
        gamma07=5e-4,
        m=3.2,
        P_ref=100.0,
        c=21.0,
        phi=15.3,
        psi=0.0,
    )

    # layer_015: 144_圆砾（数值很大，请确认是否合理）
    layer_015 = SoilMaterialFactory.create(
        SoilMaterialsType.HSS,
        name="144_圆砾",
        gamma=20.2,
        gamma_sat=25,
        nu=0.25,
        E=5.625e6,        # 5.625e9 Pa → 5.625e6 kPa
        E_oed=4.500e6,
        E_ur=1.6875e7,
        G0_ref=2.025e7,   # 3 × Eur/[2(1+ν)] = 2.025e7 kPa
        gamma07=1e-4,
        m=15.0,
        P_ref=100.0,
        c=3.0,
        phi=38.0,
        psi=8.0,
    )


    # region Update underground water  (permeability now in m/day)

    # layer_001: 11_杂填土
    layer_001.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.432, ky=0.432, kz=0.432,      # m/day (from 5e-06 m/s)
        Gw_Psiunsat=0.0,
    )

    # layer_002: 12_素填土
    layer_002.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.1728, ky=0.1728, kz=0.1728,   # from 2e-06 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_003: 22_砂质粉土
    layer_003.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.41472, ky=0.3456, kz=0.3456,  # from 4.8e-06, 4.0e-06, 4.0e-06 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_004: 32_砂质粉土
    layer_004.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.44928, ky=0.3456, kz=0.3456,  # from 5.2e-06, 4.0e-06, 4.0e-06 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_005: 35_粉砂夹粉土
    layer_005.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.46656, ky=0.3888, kz=0.3888,  # from 5.4e-06, 4.5e-06, 4.5e-06 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_006: 37_砂质粉土夹淤泥质粉质黏土
    layer_006.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.3456, ky=0.1296, kz=0.1296,   # from 4.0e-06, 1.5e-06, 1.5e-06 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_007: 62_淤泥质粉质黏土
    layer_007.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.00432, ky=4.32e-4, kz=4.32e-4,  # from 5e-08, 5e-09, 5e-09 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_008: 62t_淤泥质粉质黏土夹粉土
    layer_008.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.006912, ky=2.592e-4, kz=2.592e-4,  # from 8e-08, 3e-09, 3e-09 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_009: 63_粉砂夹粉土
    layer_009.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.48384, ky=0.3888, kz=0.3888,  # from 5.6e-06, 4.5e-06, 4.5e-06 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_010: 81_黏土
    layer_010.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=2.592e-4, ky=2.592e-4, kz=2.592e-4,  # from 3e-09 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_011: 102_粉质黏土
    layer_011.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.00432, ky=0.00432, kz=0.00432,  # from 5e-08 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_012: 111_粉质黏土
    layer_012.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.00432, ky=0.00432, kz=0.00432,  # from 5e-08 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_013: 121_细砂
    layer_013.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.864, ky=0.432, kz=0.432,        # from 1e-05, 5e-06, 5e-06 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_014: 132_粉质黏土
    layer_014.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=0.00432, ky=0.00432, kz=0.00432,  # from 5e-08 m/s
        Gw_Psiunsat=0.0,
    )

    # layer_015: 144_圆砾
    layer_015.set_under_ground_water(
        type=MCGWType.Standard, SWCC_method=MCGwSWCC.Van,
        soil_posi=None, soil_fine=None, Gw_defaults=None,
        infiltration=None, default_method=None,
        kx=8.64e3, ky=69.12, kz=69.12,       # from 1.0e-01, 8.0e-04, 8.0e-04 m/s
        Gw_Psiunsat=0.0,
    )

    #endregion

    # -*- coding: utf-8 -*-
    # Use ALL newly created soil materials (layer_001 ~ layer_015) in the model.
    # English comments only.
    for m in (
        layer_001, layer_002, layer_003, layer_004, layer_005,
        layer_006, layer_007, layer_008, layer_009, layer_010,
        layer_011, layer_012, layer_013, layer_014, layer_015
    ):
        pit.add_material("soil_materials", m)

    sl_001 = SoilLayer("11_杂填土",           material=layer_001)
    sl_002 = SoilLayer("12_素填土",           material=layer_002)
    sl_003 = SoilLayer("22_砂质粉土",         material=layer_003)
    sl_004 = SoilLayer("32_砂质粉土",         material=layer_004)
    sl_005 = SoilLayer("35_粉砂夹粉土",       material=layer_005)
    sl_006 = SoilLayer("37_砂质粉土夹淤泥质粉质黏土", material=layer_006)
    sl_007 = SoilLayer("62_淤泥质粉质黏土",   material=layer_007)
    sl_008 = SoilLayer("62t_淤泥质粉质黏土夹粉土", material=layer_008)
    sl_009 = SoilLayer("63_粉砂夹粉土",       material=layer_009)
    sl_010 = SoilLayer("81_黏土",             material=layer_010)
    sl_011 = SoilLayer("102_粉质黏土",         material=layer_011)
    sl_012 = SoilLayer("111_粉质黏土",         material=layer_012)
    sl_013 = SoilLayer("121_细砂",             material=layer_013)
    sl_014 = SoilLayer("132_粉质黏土",         material=layer_014)
    sl_015 = SoilLayer("144_圆砾",             material=layer_015)

    #    ALL 15 soil layers are used across the site. (CDG/MWR remain as rock basement.)
    GW_HEAD_NEAR_SURF = -3.5
    # Canonical horizons to reduce cross-overs
    H0    =  0.0     # ground
    H1    = -3.0     # FILL -> UCLAY
    H2    = -10.0    # UCLAY -> LCLAY
    H3    = -15.0    # LCLAY -> SAND (upper)
    HSPL  = -20.5    # split inside SAND (upper -> lower)
    HGRAV = -29.5    # SAND (lower) -> GRAVEL
    HBOT  = -50.0    # model bottom

    # Short aliases to your SoilLayer objects
    SL11, SL12 = sl_001, sl_002
    SL22, SL32, SL35, SL37, SL63, SL121 = sl_003, sl_004, sl_005, sl_006, sl_009, sl_013
    SL62, SL62t, SL81 = sl_007, sl_008, sl_010
    SL102, SL111, SL132 = sl_011, sl_012, sl_014
    SL144 = sl_015

    # Coordinates: 4 rows (y = 30, 10, -10, -30), 6 cols (x = -260, -156, -52, 52, 156, 260)
    # Row 1 (y = 30)
    BH1  = Borehole("BH1",  Point(-260.0,  30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH1_FILL(11)",      H0,   H1,   SL11),
        BoreholeLayer("BH1_UCLAY(62)",     H1,   H2,   SL62),
        BoreholeLayer("BH1_LCLAY(102)",    H2,   H3,   SL102),
        BoreholeLayer("BH1_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH1_SAND_LOW(22)",  HSPL, HGRAV,SL22),
        BoreholeLayer("BH1_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])
    BH2  = Borehole("BH2",  Point(-156.0,  30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH2_FILL(12)",      H0,   H1,   SL12),
        BoreholeLayer("BH2_UCLAY(62t)",    H1,   H2,   SL62t),
        BoreholeLayer("BH2_LCLAY(111)",    H2,   H3,   SL111),
        BoreholeLayer("BH2_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH2_SAND_LOW(32)",  HSPL, HGRAV,SL32),
        BoreholeLayer("BH2_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])
    BH3  = Borehole("BH3",  Point( -52.0,  30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH3_FILL(11)",      H0,   H1,   SL11),
        BoreholeLayer("BH3_UCLAY(81)",     H1,   H2,   SL81),
        BoreholeLayer("BH3_LCLAY(132)",    H2,   H3,   SL132),
        BoreholeLayer("BH3_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH3_SAND_LOW(22)",  HSPL, HGRAV,SL22),
        BoreholeLayer("BH3_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])
    BH4  = Borehole("BH4",  Point(  52.0,  30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH4_FILL(12)",      H0,   H1,   SL12),
        BoreholeLayer("BH4_UCLAY(62)",     H1,   H2,   SL62),
        BoreholeLayer("BH4_LCLAY(102)",    H2,   H3,   SL102),
        BoreholeLayer("BH4_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH4_SAND_LOW(32)",  HSPL, HGRAV,SL32),
        BoreholeLayer("BH4_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])
    BH5  = Borehole("BH5",  Point( 156.0,  30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH5_FILL(11)",      H0,   H1,   SL11),
        BoreholeLayer("BH5_UCLAY(62t)",    H1,   H2,   SL62t),
        BoreholeLayer("BH5_LCLAY(111)",    H2,   H3,   SL111),
        BoreholeLayer("BH5_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH5_SAND_LOW(22)",  HSPL, HGRAV,SL22),
        BoreholeLayer("BH5_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])
    BH6  = Borehole("BH6",  Point( 260.0,  30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH6_FILL(12)",      H0,   H1,   SL12),
        BoreholeLayer("BH6_UCLAY(81)",     H1,   H2,   SL81),
        BoreholeLayer("BH6_LCLAY(132)",    H2,   H3,   SL132),
        BoreholeLayer("BH6_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH6_SAND_LOW(32)",  HSPL, HGRAV,SL32),
        BoreholeLayer("BH6_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])

    # Row 2 (y = 10)
    BH7  = Borehole("BH7",  Point(-260.0,  10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH7_FILL(11)",      H0,   H1,   SL11),
        BoreholeLayer("BH7_UCLAY(62)",     H1,   H2,   SL62),
        BoreholeLayer("BH7_LCLAY(102)",    H2,   H3,   SL102),
        BoreholeLayer("BH7_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH7_SAND_LOW(22)",  HSPL, HGRAV,SL22),
        BoreholeLayer("BH7_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])
    BH8  = Borehole("BH8",  Point(-156.0,  10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH8_FILL(12)",      H0,   H1,   SL12),
        BoreholeLayer("BH8_UCLAY(62t)",    H1,   H2,   SL62t),
        BoreholeLayer("BH8_LCLAY(111)",    H2,   H3,   SL111),
        BoreholeLayer("BH8_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH8_SAND_LOW(32)",  HSPL, HGRAV,SL32),
        BoreholeLayer("BH8_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])
    BH9  = Borehole("BH9",  Point( -52.0,  10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH9_FILL(11)",      H0,   H1,   SL11),
        BoreholeLayer("BH9_UCLAY(81)",     H1,   H2,   SL81),
        BoreholeLayer("BH9_LCLAY(132)",    H2,   H3,   SL132),
        BoreholeLayer("BH9_SAND_UP(121)",  H3,   HSPL, SL121),
        BoreholeLayer("BH9_SAND_LOW(22)",  HSPL, HGRAV,SL22),
        BoreholeLayer("BH9_GRAVEL(144)",   HGRAV,HBOT, SL144),
    ])
    BH10 = Borehole("BH10", Point(  52.0,  10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH10_FILL(12)",     H0,   H1,   SL12),
        BoreholeLayer("BH10_UCLAY(62)",    H1,   H2,   SL62),
        BoreholeLayer("BH10_LCLAY(102)",   H2,   H3,   SL102),
        BoreholeLayer("BH10_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH10_SAND_LOW(32)", HSPL, HGRAV,SL32),
        BoreholeLayer("BH10_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH11 = Borehole("BH11", Point( 156.0,  10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH11_FILL(11)",     H0,   H1,   SL11),
        BoreholeLayer("BH11_UCLAY(62t)",   H1,   H2,   SL62t),
        BoreholeLayer("BH11_LCLAY(111)",   H2,   H3,   SL111),
        BoreholeLayer("BH11_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH11_SAND_LOW(22)", HSPL, HGRAV,SL22),
        BoreholeLayer("BH11_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH12 = Borehole("BH12", Point( 260.0,  10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH12_FILL(12)",     H0,   H1,   SL12),
        BoreholeLayer("BH12_UCLAY(81)",    H1,   H2,   SL81),
        BoreholeLayer("BH12_LCLAY(132)",   H2,   H3,   SL132),
        BoreholeLayer("BH12_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH12_SAND_LOW(32)", HSPL, HGRAV,SL32),
        BoreholeLayer("BH12_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])

    # Row 3 (y = -10)
    BH13 = Borehole("BH13", Point(-260.0, -10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH13_FILL(11)",     H0,   H1,   SL11),
        BoreholeLayer("BH13_UCLAY(62)",    H1,   H2,   SL62),
        BoreholeLayer("BH13_LCLAY(102)",   H2,   H3,   SL102),
        BoreholeLayer("BH13_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH13_SAND_LOW(22)", HSPL, HGRAV,SL22),
        BoreholeLayer("BH13_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH14 = Borehole("BH14", Point(-156.0, -10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH14_FILL(12)",     H0,   H1,   SL12),
        BoreholeLayer("BH14_UCLAY(62t)",   H1,   H2,   SL62t),
        BoreholeLayer("BH14_LCLAY(111)",   H2,   H3,   SL111),
        BoreholeLayer("BH14_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH14_SAND_LOW(32)", HSPL, HGRAV,SL32),
        BoreholeLayer("BH14_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH15 = Borehole("BH15", Point( -52.0, -10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH15_FILL(11)",     H0,   H1,   SL11),
        BoreholeLayer("BH15_UCLAY(81)",    H1,   H2,   SL81),
        BoreholeLayer("BH15_LCLAY(132)",   H2,   H3,   SL132),
        BoreholeLayer("BH15_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH15_SAND_LOW(22)", HSPL, HGRAV,SL22),
        BoreholeLayer("BH15_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH16 = Borehole("BH16", Point(  52.0, -10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH16_FILL(12)",     H0,   H1,   SL12),
        BoreholeLayer("BH16_UCLAY(62)",    H1,   H2,   SL62),
        BoreholeLayer("BH16_LCLAY(102)",   H2,   H3,   SL102),
        BoreholeLayer("BH16_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH16_SAND_LOW(32)", HSPL, HGRAV,SL32),
        BoreholeLayer("BH16_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH17 = Borehole("BH17", Point( 156.0, -10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH17_FILL(11)",     H0,   H1,   SL11),
        BoreholeLayer("BH17_UCLAY(62t)",   H1,   H2,   SL62t),
        BoreholeLayer("BH17_LCLAY(111)",   H2,   H3,   SL111),
        BoreholeLayer("BH17_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH17_SAND_LOW(22)", HSPL, HGRAV,SL22),
        BoreholeLayer("BH17_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH18 = Borehole("BH18", Point( 260.0, -10.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH18_FILL(12)",     H0,   H1,   SL12),
        BoreholeLayer("BH18_UCLAY(81)",    H1,   H2,   SL81),
        BoreholeLayer("BH18_LCLAY(132)",   H2,   H3,   SL132),
        BoreholeLayer("BH18_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH18_SAND_LOW(32)", HSPL, HGRAV,SL32),
        BoreholeLayer("BH18_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])

    # Row 4 (y = -30)
    BH19 = Borehole("BH19", Point(-260.0, -30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH19_FILL(11)",     H0,   H1,   SL11),
        BoreholeLayer("BH19_UCLAY(62)",    H1,   H2,   SL62),
        BoreholeLayer("BH19_LCLAY(102)",   H2,   H3,   SL102),
        BoreholeLayer("BH19_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH19_SAND_LOW(22)", HSPL, HGRAV,SL22),
        BoreholeLayer("BH19_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH20 = Borehole("BH20", Point(-156.0, -30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH20_FILL(12)",     H0,   H1,   SL12),
        BoreholeLayer("BH20_UCLAY(62t)",   H1,   H2,   SL62t),
        BoreholeLayer("BH20_LCLAY(111)",   H2,   H3,   SL111),
        BoreholeLayer("BH20_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH20_SAND_LOW(32)", HSPL, HGRAV,SL32),
        BoreholeLayer("BH20_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH21 = Borehole("BH21", Point( -52.0, -30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH21_FILL(11)",     H0,   H1,   SL11),
        BoreholeLayer("BH21_UCLAY(81)",    H1,   H2,   SL81),
        BoreholeLayer("BH21_LCLAY(132)",   H2,   H3,   SL132),
        BoreholeLayer("BH21_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH21_SAND_LOW(22)", HSPL, HGRAV,SL22),
        BoreholeLayer("BH21_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH22 = Borehole("BH22", Point(  52.0, -30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH22_FILL(12)",     H0,   H1,   SL12),
        BoreholeLayer("BH22_UCLAY(62)",    H1,   H2,   SL62),
        BoreholeLayer("BH22_LCLAY(102)",   H2,   H3,   SL102),
        BoreholeLayer("BH22_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH22_SAND_LOW(32)", HSPL, HGRAV,SL32),
        BoreholeLayer("BH22_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH23 = Borehole("BH23", Point( 156.0, -30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH23_FILL(11)",     H0,   H1,   SL11),
        BoreholeLayer("BH23_UCLAY(62t)",   H1,   H2,   SL62t),
        BoreholeLayer("BH23_LCLAY(111)",   H2,   H3,   SL111),
        BoreholeLayer("BH23_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH23_SAND_LOW(22)", HSPL, HGRAV,SL22),
        BoreholeLayer("BH23_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])
    BH24 = Borehole("BH24", Point( 260.0, -30.0, 0.0), water_head=GW_HEAD_NEAR_SURF, layers=[
        BoreholeLayer("BH24_FILL(12)",     H0,   H1,   SL12),
        BoreholeLayer("BH24_UCLAY(81)",    H1,   H2,   SL81),
        BoreholeLayer("BH24_LCLAY(132)",   H2,   H3,   SL132),
        BoreholeLayer("BH24_SAND_UP(121)", H3,   HSPL, SL121),
        BoreholeLayer("BH24_SAND_LOW(32)", HSPL, HGRAV,SL32),
        BoreholeLayer("BH24_GRAVEL(144)",  HGRAV,HBOT, SL144),
    ])

    # Collect into a set (explicit list)
    boreholes = [
        BH1, BH2, BH3, BH4, BH5, BH6,
        BH7, BH8, BH9, BH10, BH11, BH12,
        BH13, BH14, BH15, BH16, BH17, BH18,
        BH19, BH20, BH21, BH22, BH23, BH24,
    ]

    pit.borehole_set = BoreholeSet(
        name="BHSet",
        boreholes=boreholes,
        comment="All 15 soil layers are used across the boreholes; CDG & MWR as basement."
    )

    # 4) Pit layout & key elevations
    X0, Y0 = 0.0, 0.0
    W, H   = 20.0, 16.0
    Z_TOP  = 0.0
    Z_EXC_BOTTOM = -9.0
    Z_WALL_BOTTOM = -24.0

    # 5) Structure materials
    dwall_mat  = ElasticPlate(name="DWall_E", E=30e6, nu=0.2, d=0.8,  gamma=25.0)
    brace_mat  = ElasticBeam(name="Brace_E", E=35e6, nu=0.2, gamma=25.0)
    pit.add_material("plate_materials", dwall_mat)
    pit.add_material("beam_materials",  brace_mat)

    # 6) Diaphragm walls
    wall_specs = [
        ("wall_start", "x", -115.0, -14, 14),

        ("wall_b_1", "y", -14, -115, -99),
        ("wall_b_2", "x", -99,  -14,  -12),
        ("wall_b_3", "y", -12,  -99,  -39),
        ("wall_b_4", "x", -39,  -12,  -16.5),
        ("wall_b_5", "y", -16.5, -39, -13),
        ("wall_b_6", "x", -13,  -16.5, -13),
        ("wall_b_7", "y", -13,  -13,   98),
        ("wall_b_8", "x",  98,  -13,  -14.5),
        ("wall_b_9", "y", -14.5, 98,  115),
        ("wall_end", "x", 115, -14.5, 14.5),

        # 顶部镜像段
        ("wall_t_1", "y", 14,   -115, -99),
        ("wall_t_2", "x", -99,    12,   14),
        ("wall_t_3", "y", 12,    -99,  -39),
        ("wall_t_4", "x", -39,    12,  16.5),
        ("wall_t_5", "y", 16.5,  -39,  -13),
        ("wall_t_6", "x", -13,    13,  16.5),
        ("wall_t_7", "y", 13,    -13,   98),
        ("wall_t_8", "x",  98,    13,  14.5),
        ("wall_t_9", "y", 14.5,   98,  115),
    ]

    def mk_surface(ori, a, b, c):
        return rect_wall_x(a, b, c, Z_TOP, Z_WALL_BOTTOM) if ori == "x" \
            else rect_wall_y(a, b, c, Z_TOP, Z_WALL_BOTTOM)

    walls = [
        RetainingWall(name=n, surface=mk_surface(o, a, b, c), plate_type=dwall_mat)
        for (n, o, a, b, c) in wall_specs
    ]
    for w in walls:
        pit.add_structure(StructureType.RETAINING_WALLS.value, w)  # <-- enum value

    pit.excava_depth = Z_EXC_BOTTOM

    # 7) Horizontal braces (3 levels at depths 0, 3.1, 6.1 m below surface)
    braces_L1: List[Beam] = []
    braces_L2: List[Beam] = []
    braces_L3: List[Beam] = []
    buckets = [braces_L1, braces_L2, braces_L3]

    def add_brace(name: str, p0: Tuple[float, float, float], p1: Tuple[float, float, float], bucket: List[Beam]):
        b = Beam(name=name, line=line_2pts(p0, p1), beam_type=brace_mat)
        pit.add_structure(StructureType.BEAMS.value, b)   # <-- enum value
        bucket.append(b)

    anchor_pts_b = [[-110, -14], [-105, -14], [-98.5, -12], [-92.5, -12], [-86.5, -12], [-77.5, -12], [-68.5, -12], [-60.5, -12], [-52.5, -12], [-47, -12], [-38, -16.5], [-33.7, -16.5], [-27, -16.5], [-20, -16.5], [-9, -13], [-2.3, -13], [4.5, -13], [12.5, -13], [21.5, -13], [30.5, -13], [39.5, -13], [48.5, -13], [57.5, -13], [66.5, -13], [75.5, -13], [84.5, -13], [92.5, -13], [98.5, -14.5], [110, -14.5], [105, -14.5], [-110, 14], [-105, 14], [105, 14.5], [110, 14.5]]
    anchor_pts_t = [[-115, -9.2], [-115, -4.39], [-98.5, 12], [-92.5, 12], [-86.5, 12], [-77.5, 12], [-68.5, 12], [-60.5, 12], [-52.5, 12], [-47, 12], [-38, 16.5], [-33.7, 16.5], [-27, 16.5], [-20, 16.5], [-9, 13], [-2.3, 13], [4.5, 13], [12.5, 13], [21.5, 13], [30.5, 13], [39.5, 13], [48.5, 13], [57.5, 13], [66.5, 13], [75.5, 13], [84.5, 13], [92.5, 13], [98.5, 14.5], [115, -9.2], [115, -4.39], [-115, 9.2], [-115, 4.39], [115, 4.39], [115, 9.2]]

    L_depths = (0.0, 3.1, 6.1)  # below ground (m)
    for level, depth in enumerate(L_depths):
        z_coord = -float(depth)  # below ground (negative Z)
        for i in range(len(anchor_pts_b)):
            add_brace(
                f"Brace_{level+1}_{i + 1}",
                (anchor_pts_b[i][0], anchor_pts_b[i][1], z_coord),
                (anchor_pts_t[i][0], anchor_pts_t[i][1], z_coord),
                buckets[level]
            )


    # 8) Wells（坑内环井 + 内部网格井）
    xL, xR = X0 - W * 0.5, X0 + W * 0.5
    yB, yT = Y0 - H * 0.5, Y0 + H * 0.5

    MAX_WELLS     = 50
    Z_WELL_BOTTOM = Z_EXC_BOTTOM - 1.5   # 井控：坑底以下 ~1.5 m
    Q_WELL        = 691.2               # m3/day
    EDGE_SPACING  = 6.0
    GRID_DX = GRID_DY = 6.0
    CLEARANCE     = 1.0
    MARGIN        = 2.0

    # 用开挖矩形作为 footprint（如果你想严格用“围护墙轮廓”，可在 build() 后用 builder.get_wall_footprint_xy()）
    # poly_xy = [(xL, yB), (xR, yB), (xR, yT), (xL, yT)]

    # wells_unique, stats = layout_wells_with_limit(
    #     prefix="W",
    #     poly_xy=poly_xy,
    #     z_top=0.0,
    #     z_bot=Z_WELL_BOTTOM,
    #     q_well=Q_WELL,
    #     edge_spacing=EDGE_SPACING,
    #     grid_dx=GRID_DX,
    #     grid_dy=GRID_DY,
    #     clearance=CLEARANCE,
    #     margin=MARGIN,
    #     max_wells=MAX_WELLS,
    #     dedupe_tol=1e-6,
    # )
    # print(f"[WELLS] edge_spacing_used={stats['edge_spacing_used']:.2f}, "
    #       f"edge={stats['edge_count']}, grid={stats['grid_count']}, "
    #       f"unique_total={stats['unique']}, dupes={stats['dupes']}")
    
    wells_unique = get_well_positions("Wellpoint_Layout_Planner.xlsx", 0, Z_WELL_BOTTOM, Q_WELL)

    for w in wells_unique:
        pit.add_structure(StructureType.WELLS.value, w)  # <-- enum value


    well_phase2 = make_random_flows(wells_unique)
    well_phase3 = make_random_flows(wells_unique)
    well_phase4 = make_random_flows(wells_unique)
    well_phase5 = make_random_flows(wells_unique)
    well_phase6 = make_random_flows(wells_unique)
    well_phase7 = make_random_flows(wells_unique)
    well_phase8 = make_random_flows(wells_unique)

    # 9) Phases
    st_init  = PlasticStageSettings(load_type=LoadType.StageConstruction, time_interval=3)
    st_exc  = PlasticStageSettings(load_type=LoadType.StageConstruction, time_interval=5)
    st_dewat = PlasticStageSettings(load_type=LoadType.StageConstruction, time_interval=3)

    ph0 = Phase(name="P0_Initial",          settings=st_init)
    ph1 = Phase(name="P1_Dewatering",       settings=st_dewat)
    ph2 = Phase(name="P2_Excavate_L1",      settings=st_exc)
    ph3 = Phase(name="P3_Add_braces_L1",    settings=st_init)
    ph4 = Phase(name="P4_Excavate_L2",      settings=st_exc)
    ph5 = Phase(name="P5_Add_braces_L2",    settings=st_init)
    ph6 = Phase(name="P6_Excavate_L3",      settings=st_exc)
    ph7 = Phase(name="P7_Add_braces_L3",    settings=st_init)
    ph8 = Phase(name="P8_Excavate_L4",      settings=st_exc)

    ph1.set_inherits(ph0)
    ph2.set_inherits(ph1)
    ph3.set_inherits(ph2)
    ph4.set_inherits(ph3)
    ph5.set_inherits(ph4)
    ph6.set_inherits(ph5)
    ph7.set_inherits(ph6)
    ph8.set_inherits(ph7)


    # activation lists (structures)
    ph0.activate_structures(walls)
    ph1.activate_structures(wells_unique)
    ph3.activate_structures(braces_L1)
    ph5.activate_structures(braces_L2)
    ph7.activate_structures(braces_L3)

    # update wells' status
    ph2.set_wells_dict(well_phase2)
    ph3.set_wells_dict(well_phase3)
    ph4.set_wells_dict(well_phase4)
    ph5.set_wells_dict(well_phase5)
    ph6.set_wells_dict(well_phase6)
    ph7.set_wells_dict(well_phase7)
    ph8.set_wells_dict(well_phase8)

    # Per-phase water table example
    # x_min, x_max, y_min, y_max = proj.x_min, proj.x_max, proj.y_min, proj.y_max
    # water_pts = [
    #     WaterLevel(x_min, y_min, -6.0), WaterLevel(x_max, y_min, -6.0),
    #     WaterLevel(x_max, y_max, -6.0), WaterLevel(x_min, y_max, -6.0),
    #     WaterLevel(0.0,    0.0,  -6.0),
    # ]
    # ph2.set_water_table(WaterLevelTable(water_pts))

    for p in (ph0, ph1, ph2, ph3, ph4, ph5, ph6, ph7, ph8):
        pit.add_phase(p)

    return pit

# ################################# main #################################

runner = PlaxisRunner(PORT, PASSWORD, HOST)
pit = assemble_pit(runner=runner)
builder = ExcavationBuilder(runner, pit)

# Relink borehole-layer materials
try:
    builder._relink_borehole_layers_to_library()
except Exception:
    pass

print("[BUILD] start …")
builder.build(mesh=True)
print("[BUILD] done.")

print("[BUILD] Pick up soillayer.")
excava_soils = builder.get_all_child_soils_dict()
print("[BUILD] Applied the soillayer status for phases.")

print("[APPLY] apply phases …")
pit.phases[2].add_soils(*[excava_soils["Soil_1_2"], excava_soils["Soil_1_1"]])
pit.phases[4].add_soils(*[excava_soils["Soil_2_2"], excava_soils["Soil_2_3"]])
pit.phases[6].add_soils(*[excava_soils["Soil_4_2"], excava_soils["Soil_3_3"]])
pit.phases[8].add_soils(*[excava_soils["Soil_5_4"], excava_soils["Soil_5_3"]])
builder.apply_pit_soil_block()
print("[APPLY] Updated the status of soillayers")

print(
    "\n[NOTE] Braces use 'ElasticBeam' as anchor-like members here."
    "\n[NOTE] PLAXIS assumes well discharge is evenly distributed along each well; "
    "if wells intersect other objects, consider splitting the well geometry."
)

builder.calculate()

def export_walls_horizontal_displacement_excel_2(builder: ExcavationBuilder, excel_path: str) -> str:
    """
    Export max horizontal displacement per wall per phase to an Excel file
    using builder.get_results (auto ensure calculated + auto bind Output per phase).
    """
    import numpy as np
    import pandas as pd
    from plaxisproxy_excavation.excavation import StructureType
    from plaxisproxy_excavation.plaxishelper.resulttypes import Plate  # 新枚举库

    # 1) 项目阶段
    project_phases = builder.list_project_phases()
    if not project_phases:
        raise RuntimeError("No project phases found. Ensure phases are defined and calculated.")

    # 2) 围护墙对象列表（直接用对象，内部优先用 plx_id）
    pit = builder.excavation_object
    walls = (getattr(pit, "structures", {}) or {}).get(StructureType.RETAINING_WALLS.value, [])
    if not walls:
        raise RuntimeError("No retaining walls found in the pit structures.")

    def _to_array(v) -> np.ndarray:
        """统一转为 float 数组；标量→单元素数组；非数值→空数组。"""
        if isinstance(v, list):
            return np.asarray(v, dtype=float).ravel()
        if isinstance(v, (int, float, np.floating)):
            return np.asarray([float(v)], dtype=float)
        return np.asarray([], dtype=float)

    records = []
    for ph in project_phases:
        ph_name = getattr(ph, "name", str(ph))

        for wall in walls:
            wall_name = getattr(wall, "name", "Wall")

            # Ux（节点优先，失败自动切换应力点；builder 内部已处理计算/绑定阶段/节点→应力点兜底）
            try:
                ux_raw = builder.get_results(structure=wall, leaf=Plate.Ux, phase=ph, smoothing=False)
            except Exception as ex:
                records.append({
                    "Phase": ph_name, "Wall": wall_name, "NodeCount": 0,
                    "Ux_max_mm": np.nan, "Uy_max_mm": np.nan, "Uxy_max_mm": np.nan,
                    "Error": f"Ux query failed: {ex}"
                })
                continue

            # Uy
            try:
                uy_raw = builder.get_results(structure=wall, leaf=Plate.Uy, phase=ph, smoothing=False)
            except Exception as ex:
                records.append({
                    "Phase": ph_name, "Wall": wall_name, "NodeCount": 0,
                    "Ux_max_mm": np.nan, "Uy_max_mm": np.nan, "Uxy_max_mm": np.nan,
                    "Error": f"Uy query failed: {ex}"
                })
                continue

            ux_arr = _to_array(ux_raw)
            uy_arr = _to_array(uy_raw)
            node_count = int(max(len(ux_arr), len(uy_arr)))

            if node_count == 0:
                records.append({
                    "Phase": ph_name, "Wall": wall_name, "NodeCount": 0,
                    "Ux_max_mm": np.nan, "Uy_max_mm": np.nan, "Uxy_max_mm": np.nan,
                    "Error": "No numeric results for Ux/ Uy."
                })
                continue

            uxy = np.sqrt(ux_arr**2 + uy_arr**2)

            records.append({
                "Phase": ph_name,
                "Wall": wall_name,
                "NodeCount": node_count,
                "Ux_max_mm": float(np.nanmax(np.abs(ux_arr)) * 1000.0),
                "Uy_max_mm": float(np.nanmax(np.abs(uy_arr)) * 1000.0),
                "Uxy_max_mm": float(np.nanmax(np.abs(uxy)) * 1000.0),
            })

    # 4) 写 Excel
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df = df.sort_values(["Phase", "Wall"], ignore_index=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Walls_H_Disp_Summary")
    return excel_path


def export_walls_horizontal_displacement_excel(builder: ExcavationBuilder, excel_path: str) -> str:
    """
    For each retaining wall, export ALL point-wise results (X,Y,Z,Ux,Uy[,Uz])
    across ALL phases to a dedicated Excel sheet (one wall per sheet).

    Loop order: PHASES (outer) -> WALLS (inner)
    so that Output view binding is reused within each phase.
    """
    import re
    import numpy as np
    import pandas as pd
    from plaxisproxy_excavation.excavation import StructureType
    from plaxisproxy_excavation.plaxishelper.resulttypes import Plate  # 枚举库

    # 1) 项目阶段
    project_phases = builder.list_project_phases()
    if not project_phases:
        raise RuntimeError("No project phases found. Ensure phases are defined and calculated.")

    # 2) 围护墙对象列表
    pit = builder.excavation_object
    walls = (getattr(pit, "structures", {}) or {}).get(StructureType.RETAINING_WALLS.value, [])
    if not walls:
        raise RuntimeError("No retaining walls found in the pit structures.")

    # 3) 需要提取的结果项（按需容错 Uz 可能不存在）
    leaves = {}
    for key in ("X", "Y", "Z", "Ux", "Uy", "Uz"):
        member = getattr(Plate, key, None)
        if member is not None:
            leaves[key] = member

    def _to_array(v) -> np.ndarray:
        """统一数值化为 float 数组；标量→单元素数组；其它→空数组。"""
        if isinstance(v, list):
            return np.asarray(v, dtype=float).ravel()
        if isinstance(v, (int, float, np.floating)):
            return np.asarray([float(v)], dtype=float)
        return np.asarray([], dtype=float)

    def _pad(arr: np.ndarray, n: int) -> np.ndarray:
        """长度补齐到 n，用 NaN 填充。"""
        out = np.full(n, np.nan, dtype=float)
        m = min(len(arr), n)
        if m:
            out[:m] = arr[:m]
        return out

    def _safe_sheet_name(name: str, used: set) -> str:
        """Excel sheet 名清洗与去重（≤31字，禁: \ / ? * [ ] :）。"""
        s = re.sub(r'[:\\/?*\[\]]', '_', str(name)).strip() or "Sheet"
        s = s[:31]
        base = s
        i = 1
        while s in used:
            suf = f"_{i}"
            s = (base[: (31 - len(suf))] + suf) if len(base) + len(suf) > 31 else (base + suf)
            i += 1
        used.add(s)
        return s

    # 为“每面墙一个 sheet”做累积容器
    wall_rows = {getattr(w, "name", "Wall"): [] for w in walls}

    # 4) 外层相位循环（减少 view 重建）
    for ph in project_phases:
        ph_name = getattr(ph, "name", str(ph))

        # （可选优化：这行可省略，builder.get_results 首次调用会自动绑定；
        #  留着能更显式地在每个 phase 开始时完成一次绑定）
        # builder.create_output_viewer(phase=ph, reuse=True)

        # 内层墙循环
        for wall in walls:
            wall_name = getattr(wall, "name", "Wall")

            # 逐列取值（builder 内部：确保已计算 + 自动绑定阶段 + 节点→应力点兜底）
            data_cols = {}
            for k, leaf in leaves.items():
                try:
                    raw = builder.get_results(structure=wall, leaf=leaf, phase=ph, smoothing=False)
                    data_cols[k] = _to_array(raw)
                except Exception:
                    data_cols[k] = np.asarray([], dtype=float)

            max_len = max((len(a) for a in data_cols.values()), default=0)

            if max_len == 0:
                # 该相位该墙没有可用数据，保留占位行方便排查
                wall_rows[wall_name].append({
                    "Phase": ph_name,
                    "Index": np.nan,
                    **{k: np.nan for k in ("X", "Y", "Z", "Ux", "Uy", "Uz") if k in leaves}
                })
                continue

            padded = {k: _pad(a, max_len) for k, a in data_cols.items()}

            # 逐点记录
            for i in range(max_len):
                rec = {"Phase": ph_name, "Index": i}
                for k in ("X", "Y", "Z", "Ux", "Uy", "Uz"):
                    if k in padded:
                        rec[k] = float(padded[k][i])
                wall_rows[wall_name].append(rec)

    # 5) 写 Excel：每面墙一个 sheet
    used_sheet_names: set = set()
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for wall in walls:
            wall_name = getattr(wall, "name", "Wall")
            rows = wall_rows.get(wall_name, [])
            df = pd.DataFrame.from_records(rows)

            # 列顺序统一（存在的才保留）
            cols = ["Phase", "Index"] + [k for k in ("X", "Y", "Z", "Ux", "Uy", "Uz") if k in leaves]
            if not df.empty:
                df = df.loc[:, [c for c in cols if c in df.columns]]

            sheet = _safe_sheet_name(wall_name, used_sheet_names)
            df.to_excel(writer, index=False, sheet_name=sheet)

    return excel_path

print("[TEST] Exporting horizontal wall displacement (all phases) …")
excel_path = "./walls_horizontal_displacements.xlsx"
saved = export_walls_horizontal_displacement_excel(builder, excel_path)
print(f"[TEST] Excel saved: {saved}")
