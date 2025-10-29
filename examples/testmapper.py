# testmapper.py — Builder + Runner-forwarded soil mapping demo (Phase API aligned)
from math import ceil
from typing import List, Tuple, Iterable, Any, Dict, Optional

from plaxis_config import HOST, PORT, PASSWORD

# Runner / Builder / Container
from plaxisproxy_excavation.plaxishelper.plaxisrunner import PlaxisRunner
from plaxisproxy_excavation.builder.excavation_builder import ExcavationBuilder
from plaxisproxy_excavation.excavation import FoundationPit, StructureType  # <-- NEW: StructureType

# Core components
from plaxisproxy_excavation.components.projectinformation import ProjectInformation, Units
from plaxisproxy_excavation.components.phase import Phase
from plaxisproxy_excavation.components.phasesettings import PlasticStageSettings, LoadType
from plaxisproxy_excavation.components.watertable import WaterLevel, WaterLevelTable

# Boreholes & materials
from plaxisproxy_excavation.borehole import SoilLayer, BoreholeLayer, Borehole, BoreholeSet
from plaxisproxy_excavation.materials.soilmaterial import SoilMaterialFactory, SoilMaterialsType

# Geometry
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D

# Structure materials & structures
from plaxisproxy_excavation.materials.platematerial import ElasticPlate
from plaxisproxy_excavation.materials.beammaterial import ElasticBeam  # used here for horizontal braces
from plaxisproxy_excavation.structures.retainingwall import RetainingWall
from plaxisproxy_excavation.structures.beam import Beam
from plaxisproxy_excavation.structures.well import Well, WellType


# ----------------------------- geometry helpers -----------------------------

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


# ----------------------------- wells helpers -----------------------------
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


# ----------------------------- dedupe (by line) -----------------------------

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


# ----------------------------- soil overrides helper -----------------------------

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


# ----------------------------- assemble pit -----------------------------

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
        x_min=-345, x_max=345,   # 50 in X
        y_min=-50, y_max=50,     # 80 in Y
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

    # 2) Soil materials
    fill = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Fill",
        E_ref=15e6, c_ref=5e3, phi=25.0, psi=0.0, nu=0.30,
        gamma=18.0, gamma_sat=20.0, e_init=0.60
    )
    soft_clay = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Soft_Clay",
        E_ref=10e6, c_ref=18e3, phi=20.0, psi=0.0, nu=0.35,
        gamma=17.5, gamma_sat=19.0, e_init=0.95
    )
    silty_clay = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Silty_Clay",
        E_ref=18e6, c_ref=12e3, phi=23.0, psi=0.0, nu=0.33,
        gamma=18.0, gamma_sat=19.5, e_init=0.85
    )
    fine_sand = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Fine_Sand",
        E_ref=35e6, c_ref=1e3, phi=32.0, psi=2.0, nu=0.30,
        gamma=19.0, gamma_sat=21.0, e_init=0.55
    )
    medium_sand = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Medium_Sand",
        E_ref=60e6, c_ref=1e3, phi=35.0, psi=3.0, nu=0.28,
        gamma=19.5, gamma_sat=21.5, e_init=0.50
    )
    gravelly_sand = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Gravelly_Sand",
        E_ref=90e6, c_ref=1e3, phi=38.0, psi=5.0, nu=0.26,
        gamma=20.0, gamma_sat=22.0, e_init=0.45
    )
    cdg = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Completely_Decomposed_Rock",
        E_ref=120e6, c_ref=25e3, phi=36.0, psi=4.0, nu=0.26,
        gamma=20.5, gamma_sat=22.5, e_init=0.40
    )
    mwr = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Moderately_Weathered_Rock",
        E_ref=250e6, c_ref=40e3, phi=40.0, psi=6.0, nu=0.25,
        gamma=21.0, gamma_sat=22.8, e_init=0.35
    )

    for m in (fill, soft_clay, silty_clay, fine_sand, medium_sand, gravelly_sand, cdg, mwr):
        pit.add_material("soil_materials", m)

    # 2) Canonical SoilLayer objects
    sl_fill = SoilLayer("Fill", material=fill)
    sl_soft_clay = SoilLayer("Soft_Clay", material=soft_clay)
    sl_silty_clay = SoilLayer("Silty_Clay", material=silty_clay)
    sl_fine_sand = SoilLayer("Fine_Sand", material=fine_sand)
    sl_medium_sand = SoilLayer("Medium_Sand", material=medium_sand)
    sl_gravelly_sand = SoilLayer("Gravelly_Sand", material=gravelly_sand)
    sl_cdg = SoilLayer("Completely_Decomposed_Rock", material=cdg)
    sl_mwr = SoilLayer("Moderately_Weathered_Rock", material=mwr)

    # 3) Boreholes & layers — 0 → -50 m
    GW_HEAD_NEAR_SURF = -3.5

    bh1_layers = [
        BoreholeLayer("Fill@BH1",                        0.0,  -3.0,  sl_fill),
        BoreholeLayer("Soft_Clay@BH1",                  -3.0, -10.0,  sl_soft_clay),
        BoreholeLayer("Silty_Clay@BH1",                -10.0, -15.0,  sl_silty_clay),
        BoreholeLayer("Fine_Sand@BH1",                 -15.0, -21.0,  sl_fine_sand),
        BoreholeLayer("Medium_Sand@BH1",               -21.0, -29.0,  sl_medium_sand),
        BoreholeLayer("Gravelly_Sand@BH1",             -29.0, -35.0,  sl_gravelly_sand),
        BoreholeLayer("Completely_Decomposed_Rock@BH1",-35.0, -42.0,  sl_cdg),
        BoreholeLayer("Moderately_Weathered_Rock@BH1", -42.0, -50.0,  sl_mwr),
    ]
    bh2_layers = [
        BoreholeLayer("Fill@BH2",                         0.0,  -2.5,  sl_fill),
        BoreholeLayer("Soft_Clay@BH2",                   -2.5,  -9.0,  sl_soft_clay),
        BoreholeLayer("Silty_Clay@BH2",                  -9.0, -14.5,  sl_silty_clay),
        BoreholeLayer("Fine_Sand@BH2",                  -14.5, -20.5,  sl_fine_sand),
        BoreholeLayer("Medium_Sand@BH2",                -20.5, -29.5,  sl_medium_sand),
        BoreholeLayer("Gravelly_Sand@BH2",              -29.5, -36.0,  sl_gravelly_sand),
        BoreholeLayer("Completely_Decomposed_Rock@BH2", -36.0, -43.0,  sl_cdg),
        BoreholeLayer("Moderately_Weathered_Rock@BH2",  -43.0, -50.0,  sl_mwr),
    ]
    bh7_layers = [
        BoreholeLayer("Fill@BH7",                         0.0,  -2.8, sl_fill),
        BoreholeLayer("Soft_Clay@BH7",                   -2.8,  -9.5, sl_soft_clay),
        BoreholeLayer("Silty_Clay@BH7",                  -9.5, -15.2, sl_silty_clay),
        BoreholeLayer("Fine_Sand@BH7",                  -15.2, -21.5, sl_fine_sand),
        BoreholeLayer("Medium_Sand@BH7",                -21.5, -30.0, sl_medium_sand),
        BoreholeLayer("Gravelly_Sand@BH7",              -30.0, -36.0, sl_gravelly_sand),
        BoreholeLayer("Completely_Decomposed_Rock@BH7", -36.0, -43.0, sl_cdg),
        BoreholeLayer("Moderately_Weathered_Rock@BH7",  -43.0, -50.0, sl_mwr),
    ]
    bh8_layers = [
        BoreholeLayer("Fill@BH8",                         0.0,  -3.2, sl_fill),
        BoreholeLayer("Soft_Clay@BH8",                   -3.2, -10.5, sl_soft_clay),
        BoreholeLayer("Silty_Clay@BH8",                 -10.5, -15.0, sl_silty_clay),
        BoreholeLayer("Fine_Sand@BH8",                  -15.0, -22.0, sl_fine_sand),
        BoreholeLayer("Medium_Sand@BH8",                -22.0, -29.0, sl_medium_sand),
        BoreholeLayer("Gravelly_Sand@BH8",              -29.0, -34.5, sl_gravelly_sand),
        BoreholeLayer("Completely_Decomposed_Rock@BH8", -34.5, -41.5, sl_cdg),
        BoreholeLayer("Moderately_Weathered_Rock@BH8",  -41.5, -50.0, sl_mwr),
    ]
    bh9_layers = [
        BoreholeLayer("Fill@BH9",                         0.0,  -2.7, sl_fill),
        BoreholeLayer("Soft_Clay@BH9",                   -2.7,  -9.2, sl_soft_clay),
        BoreholeLayer("Silty_Clay@BH9",                  -9.2, -14.0, sl_silty_clay),
        BoreholeLayer("Fine_Sand@BH9",                  -14.0, -20.0, sl_fine_sand),
        BoreholeLayer("Medium_Sand@BH9",                -20.0, -28.5, sl_medium_sand),
        BoreholeLayer("Gravelly_Sand@BH9",              -28.5, -35.5, sl_gravelly_sand),
        BoreholeLayer("Completely_Decomposed_Rock@BH9", -35.5, -44.0, sl_cdg),
        BoreholeLayer("Moderately_Weathered_Rock@BH9",  -44.0, -50.0, sl_mwr),
    ]
    bh10_layers = [
        BoreholeLayer("Fill@BH10",                         0.0,  -3.0, sl_fill),
        BoreholeLayer("Soft_Clay@BH10",                   -3.0,  -9.8, sl_soft_clay),
        BoreholeLayer("Silty_Clay@BH10",                  -9.8, -14.8, sl_silty_clay),
        BoreholeLayer("Fine_Sand@BH10",                  -14.8, -21.0, sl_fine_sand),
        BoreholeLayer("Medium_Sand@BH10",                -21.0, -29.2, sl_medium_sand),
        BoreholeLayer("Gravelly_Sand@BH10",              -29.2, -35.2, sl_gravelly_sand),
        BoreholeLayer("Completely_Decomposed_Rock@BH10", -35.2, -42.2, sl_cdg),
        BoreholeLayer("Moderately_Weathered_Rock@BH10",  -42.2, -50.0, sl_mwr),
    ]
    bh11_layers = [
        BoreholeLayer("Fill@BH11",                         0.0,  -2.9, sl_fill),
        BoreholeLayer("Soft_Clay@BH11",                   -2.9,  -9.4, sl_soft_clay),
        BoreholeLayer("Silty_Clay@BH11",                  -9.4, -14.6, sl_silty_clay),
        BoreholeLayer("Fine_Sand@BH11",                  -14.6, -20.8, sl_fine_sand),
        BoreholeLayer("Medium_Sand@BH11",                -20.8, -29.8, sl_medium_sand),
        BoreholeLayer("Gravelly_Sand@BH11",              -29.8, -36.5, sl_gravelly_sand),
        BoreholeLayer("Completely_Decomposed_Rock@BH11", -36.5, -43.5, sl_cdg),
        BoreholeLayer("Moderately_Weathered_Rock@BH11",  -43.5, -50.0, sl_mwr),
    ]
    bh12_layers = [
        BoreholeLayer("Fill@BH12",                         0.0,  -3.1, sl_fill),
        BoreholeLayer("Soft_Clay@BH12",                   -3.1, -10.2, sl_soft_clay),
        BoreholeLayer("Silty_Clay@BH12",                 -10.2, -15.4, sl_silty_clay),
        BoreholeLayer("Fine_Sand@BH12",                  -15.4, -22.4, sl_fine_sand),
        BoreholeLayer("Medium_Sand@BH12",                -22.4, -30.2, sl_medium_sand),
        BoreholeLayer("Gravelly_Sand@BH12",              -30.2, -36.8, sl_gravelly_sand),
        BoreholeLayer("Completely_Decomposed_Rock@BH12", -36.8, -44.2, sl_cdg),
        BoreholeLayer("Moderately_Weathered_Rock@BH12",  -44.2, -50.0, sl_mwr),
    ]

    bh1 = Borehole("BH_1", Point(-10,   0, 0), 0.0, layers=bh1_layers, water_head=GW_HEAD_NEAR_SURF)
    bh2 = Borehole("BH_2", Point( 10,   0, 0), 0.0, layers=bh2_layers, water_head=GW_HEAD_NEAR_SURF)
    bh3 = Borehole("BH_3", Point(-12,  -6, 0), 0.0, layers=bh1_layers, water_head=GW_HEAD_NEAR_SURF)
    bh4 = Borehole("BH_4", Point( 12,   6, 0), 0.0, layers=bh2_layers, water_head=GW_HEAD_NEAR_SURF)
    bh5 = Borehole("BH_5", Point(-12,   6, 0), 0.0, layers=bh2_layers, water_head=GW_HEAD_NEAR_SURF)
    bh6 = Borehole("BH_6", Point( 12,  -6, 0), 0.0, layers=bh1_layers, water_head=GW_HEAD_NEAR_SURF)
    bh7  = Borehole("BH_7",  Point(  0.0,  -8.0, 0.0), 0.0, layers=bh7_layers,  water_head=GW_HEAD_NEAR_SURF)
    bh8  = Borehole("BH_8",  Point(  0.0,   8.0, 0.0), 0.0, layers=bh8_layers,  water_head=GW_HEAD_NEAR_SURF)
    bh9  = Borehole("BH_9",  Point( -8.0,   0.0, 0.0), 0.0, layers=bh9_layers,  water_head=GW_HEAD_NEAR_SURF)
    bh10 = Borehole("BH_10", Point(  8.0,   0.0, 0.0), 0.0, layers=bh10_layers, water_head=GW_HEAD_NEAR_SURF)
    bh11 = Borehole("BH_11", Point( -6.0,   6.0, 0.0), 0.0, layers=bh11_layers, water_head=GW_HEAD_NEAR_SURF)
    bh12 = Borehole("BH_12", Point(  6.0,  -6.0, 0.0), 0.0, layers=bh12_layers, water_head=GW_HEAD_NEAR_SURF)

    pit.borehole_set = BoreholeSet(
        name="BHSet",
        boreholes=[bh1, bh2, bh3, bh4, bh5, bh6, bh7, bh8, bh9, bh10, bh11, bh12],
        comment="50 m stack; GW head ≈ -3.5 m; added BH_7~BH_12 with nuanced layer boundaries"
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
    Q_WELL        = 0.008                # m3/s
    EDGE_SPACING  = 6.0
    GRID_DX = GRID_DY = 6.0
    CLEARANCE     = 1.0
    MARGIN        = 2.0

    # 用开挖矩形作为 footprint（如果你想严格用“围护墙轮廓”，可在 build() 后用 builder.get_wall_footprint_xy()）
    poly_xy = [(xL, yB), (xR, yB), (xR, yT), (xL, yT)]

    wells_unique, stats = layout_wells_with_limit(
        prefix="W",
        poly_xy=poly_xy,
        z_top=0.0,
        z_bot=Z_WELL_BOTTOM,
        q_well=Q_WELL,
        edge_spacing=EDGE_SPACING,
        grid_dx=GRID_DX,
        grid_dy=GRID_DY,
        clearance=CLEARANCE,
        margin=MARGIN,
        max_wells=MAX_WELLS,
        dedupe_tol=1e-6,
    )
    print(f"[WELLS] edge_spacing_used={stats['edge_spacing_used']:.2f}, "
          f"edge={stats['edge_count']}, grid={stats['grid_count']}, "
          f"unique_total={stats['unique']}, dupes={stats['dupes']}")
    for w in wells_unique:
        pit.add_structure(StructureType.WELLS.value, w)  # <-- enum value

    # 9) Phases
    st_init  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=120, time_interval=0.5)
    st_exc1  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=140, time_interval=1.0)
    st_dewat = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=120, time_interval=0.5)
    st_exc2  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=140, time_interval=1.0)

    ph0 = Phase(name="P0_Initial",     settings=st_init)
    ph1 = Phase(name="P2_Dewatering",  settings=st_dewat)
    ph2 = Phase(name="P1_Excavate_L1", settings=st_exc1)
    ph3 = Phase(name="P3_Excavate_L2", settings=st_exc2)

    ph1.set_inherits(ph0)
    ph2.set_inherits(ph1)
    ph3.set_inherits(ph2)

    # Soil overrides before phases are added
    if excav_names:
        ph1.set_soil_overrides(mk_soil_overrides(excav_names, active=False))
    if remain_names:
        ph2.set_soil_overrides(mk_soil_overrides(remain_names, deformable=False))
        ph3.set_soil_overrides(mk_soil_overrides(remain_names, deformable=True))

    # activation lists (structures)
    ph0.activate_structures(walls)
    ph0.activate_structures(braces_L1)
    ph1.activate_structures(braces_L2)
    ph3.activate_structures(braces_L3)

    # Per-phase water table example
    x_min, x_max, y_min, y_max = proj.x_min, proj.x_max, proj.y_min, proj.y_max
    water_pts = [
        WaterLevel(x_min, y_min, -6.0), WaterLevel(x_max, y_min, -6.0),
        WaterLevel(x_max, y_max, -6.0), WaterLevel(x_min, y_max, -6.0),
        WaterLevel(0.0,    0.0,  -6.0),
    ]
    ph2.set_water_table(WaterLevelTable(water_pts))

    for p in (ph0, ph1, ph2, ph3):
        pit.add_phase(p)

    return pit


# --------------------------------- main ---------------------------------

runner = PlaxisRunner(HOST, PORT, PASSWORD)
pit = assemble_pit(runner=runner)
builder = ExcavationBuilder(runner, pit)

# Relink borehole-layer materials
try:
    builder._relink_borehole_layers_to_library()
except Exception:
    pass

print("[BUILD] start …")
builder.build()
print("[BUILD] done.")

print("[BUILD] Pick up soillayer.")
excava_soils = builder.get_all_child_soils_dict()
print("[BUILD] Applied the soillayer status for phases.")

print("[APPLY] apply phases …")
pit.phases[1].add_soils(excava_soils["Soil_1_1"])
pit.phases[3].add_soils(*[excava_soils["Soil_1_1"], excava_soils["Soil_2_1"]])
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


print("[TEST] Exporting horizontal wall displacement (all phases) …")
excel_path = "./walls_horizontal_displacements.xlsx"
saved = export_walls_horizontal_displacement_excel_2(builder, excel_path)
print(f"[TEST] Excel saved: {saved}")
