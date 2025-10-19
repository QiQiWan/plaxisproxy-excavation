# testmapper.py — Builder + Runner-forwarded soil mapping demo (Phase API aligned)
from math import ceil
from typing import List, Tuple, Iterable, Any, Dict, Optional

from config.plaxis_config import HOST, PORT, PASSWORD

# Runner / Builder / Container
from src.plaxisproxy_excavation.plaxishelper.plaxisrunner import PlaxisRunner
from src.excavation_builder import ExcavationBuilder
from src.plaxisproxy_excavation.excavation import FoundationPit

# Core components
from src.plaxisproxy_excavation.components.projectinformation import ProjectInformation, Units
from src.plaxisproxy_excavation.components.phase import Phase
from src.plaxisproxy_excavation.components.phasesettings import PlasticStageSettings, LoadType
from src.plaxisproxy_excavation.components.watertable import WaterLevel, WaterLevelTable

# Boreholes & materials
from src.plaxisproxy_excavation.borehole import SoilLayer, BoreholeLayer, Borehole, BoreholeSet
from src.plaxisproxy_excavation.materials.soilmaterial import SoilMaterialFactory, SoilMaterialsType

# Geometry
from src.plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D

# Structure materials & structures
from src.plaxisproxy_excavation.materials.platematerial import ElasticPlate
from src.plaxisproxy_excavation.materials.beammaterial import ElasticBeam  # used here for horizontal braces
from src.plaxisproxy_excavation.structures.retainingwall import RetainingWall
from src.plaxisproxy_excavation.structures.beam import Beam
from src.plaxisproxy_excavation.structures.well import Well, WellType


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

def ring_wells(prefix: str, x0: float, y0: float, w: float, h: float,
               z_top: float, z_bot: float, spacing: float,
               clearance: float = 0.8) -> List[Well]:
    """Perimeter wells around [x0,x0+w]×[y0,y0+h], inset by 'clearance'."""
    wells: List[Well] = []
    xs = [x0 + clearance, x0 + w - clearance]
    ys = [y0 + clearance, y0 + h - clearance]

    # vertical edges
    ny = max(1, ceil((h - 2 * clearance) / spacing))
    for i in range(ny + 1):
        y = ys[0] + (ys[1] - ys[0]) * i / max(1, ny)
        for x in (xs[0], xs[1]):
            wells.append(
                Well(
                    name=f"{prefix}_P_{len(wells)+1}",
                    line=line_2pts((x, y, z_top), (x, y, z_bot)),
                    well_type=WellType.Extraction,
                    h_min=z_bot,
                )
            )

    # horizontal edges
    nx = max(1, ceil((w - 2 * clearance) / spacing))
    for i in range(nx + 1):
        x = xs[0] + (xs[1] - xs[0]) * i / max(1, nx)
        for y in (ys[0], ys[1]):
            wells.append(
                Well(
                    name=f"{prefix}_P_{len(wells)+1}",
                    line=line_2pts((x, y, z_top), (x, y, z_bot)),
                    well_type=WellType.Extraction,
                    h_min=z_bot,
                )
            )
    return wells

def grid_wells(prefix: str, x0: float, y0: float, w: float, h: float,
               z_top: float, z_bot: float, dx: float, dy: float,
               margin: float = 2.0) -> List[Well]:
    """Interior grid of wells; kept off the walls by 'margin'."""
    wells: List[Well] = []
    xi = x0 + margin; xf = x0 + w - margin
    yi = y0 + margin; yf = y0 + h - margin
    nx = max(1, ceil((xf - xi) / dx))
    ny = max(1, ceil((yf - yi) / dy))
    for ix in range(nx + 1):
        x = xi + (xf - xi) * ix / max(1, nx)
        for iy in range(ny + 1):
            y = yi + (yf - yi) * iy / max(1, ny)
            wells.append(
                Well(
                    name=f"{prefix}_G_{len(wells)+1}",
                    line=line_2pts((x, y, z_top), (x, y, z_bot)),
                    well_type=WellType.Extraction,
                    h_min=z_bot,
                )
            )
    return wells


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
    """
    Build a dict compatible with Phase.set_soil_overrides():
        { "Soil_1_1": {"active": False, "deformable": True}, ... }
    Any parameter left as None will be omitted for that soil.
    """
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
        x_min=-25, x_max=25,   # 50 in X
        y_min=-40, y_max=40,   # 80 in Y
    )

    pit = FoundationPit(project_information=proj)

    # --- Discover split soils in Staged view BEFORE adding phases (via runner minimal API) ---
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
    sand = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Sand",
        E_ref=35e6, c_ref=1e3, phi=32.0, psi=2.0, nu=0.30,
        gamma=19.0, gamma_sat=21.0, e_init=0.55
    )
    clay = SoilMaterialFactory.create(
        SoilMaterialsType.MC, name="Clay",
        E_ref=12e6, c_ref=15e3, phi=22.0, psi=0.0, nu=0.35,
        gamma=17.0, gamma_sat=19.0, e_init=0.90
    )
    for m in (fill, sand, clay):
        pit.add_material("soil_materials", m)

    # 3) Boreholes & layers
    sl_fill, sl_sand, sl_clay = SoilLayer("Fill", material=fill), SoilLayer("Sand", material=sand), SoilLayer("Clay", material=clay)

    bh1_layers = [
        BoreholeLayer("Fill@BH1",  0.0, -1.5, sl_fill),
        BoreholeLayer("Sand@BH1", -1.5, -8.0, sl_sand),
        BoreholeLayer("Clay@BH1", -8.0, -25.0, sl_clay),
    ]
    bh2_layers = [
        BoreholeLayer("Fill@BH2",  0.0, -1.0, sl_fill),
        BoreholeLayer("Sand@BH2", -1.0, -7.0, sl_sand),
        BoreholeLayer("Clay@BH2", -7.0, -25.0, sl_clay),
    ]
    bh1 = Borehole("BH_1", Point(-10, 0, 0), 0.0, layers=bh1_layers, water_head=-1.5)
    bh2 = Borehole("BH_2", Point( 10, 0, 0), 0.0, layers=bh2_layers, water_head=-1.0)
    pit.borehole_set = BoreholeSet(name="BHSet", boreholes=[bh1, bh2], comment="Demo set")

    # 4) Pit layout
    X0, Y0 = 0.0, 0.0
    W, H   = 20.0, 16.0
    Z_TOP  = 0.0
    Z_L1   = -4.0
    Z_L2   = -8.0
    Z_EXC_BOTTOM = -10.0
    Z_WALL_BOTTOM = -16.0
    Z_WELL_BOTTOM = -18.0

    # 5) Structure materials
    dwall_mat  = ElasticPlate(name="DWall_E", E=30e6, nu=0.2, d=0.8,  gamma=25.0)
    brace_mat  = ElasticBeam(name="Brace_E", E=35e6, nu=0.2, gamma=25.0)
    pit.add_material("plate_materials", dwall_mat)
    pit.add_material("beam_materials",  brace_mat)

    # 6) Diaphragm walls
    xL, xR = X0 - W/2, X0 + W/2
    yB, yT = Y0 - H/2, Y0 + H/2
    west_wall  = RetainingWall(name="DWall_W", surface=rect_wall_x(xL, yB, yT, Z_TOP, Z_WALL_BOTTOM), plate_type=dwall_mat)
    east_wall  = RetainingWall(name="DWall_E", surface=rect_wall_x(xR, yB, yT, Z_TOP, Z_WALL_BOTTOM), plate_type=dwall_mat)
    south_wall = RetainingWall(name="DWall_S", surface=rect_wall_y(yB, xL, xR, Z_TOP, Z_WALL_BOTTOM), plate_type=dwall_mat)
    north_wall = RetainingWall(name="DWall_N", surface=rect_wall_y(yT, xL, xR, Z_TOP, Z_WALL_BOTTOM), plate_type=dwall_mat)
    for w in (west_wall, east_wall, south_wall, north_wall):
        pit.add_structure("retaining_walls", w)

    pit.excava_depth = Z_EXC_BOTTOM
    # 7) Horizontal braces (two levels)
    off = 0.8
    braces_L1: List[Beam] = []
    braces_L2: List[Beam] = []

    def add_brace(name: str, p0: Tuple[float, float, float], p1: Tuple[float, float, float], bucket: List[Beam]):
        b = Beam(name=name, line=line_2pts(p0, p1), beam_type=brace_mat)
        pit.add_structure("beams", b)
        bucket.append(b)

    for z, bucket in ((Z_L1, braces_L1), (Z_L2, braces_L2)):
        add_brace(f"Brace_X_{abs(int(z))}", (xL + off, Y0, z), (xR - off, Y0, z), bucket)
        add_brace(f"Brace_Y_{abs(int(z))}", (X0, yB + off, z), (X0, yT - off, z), bucket)
        # optional diagonals
        add_brace(f"Brace_WN_{abs(int(z))}", (xL + off, Y0, z), (X0, yT - off, z), bucket)
        add_brace(f"Brace_WS_{abs(int(z))}", (xL + off, Y0, z), (X0, yB + off, z), bucket)
        add_brace(f"Brace_EN_{abs(int(z))}", (xR - off, Y0, z), (X0, yT - off, z), bucket)
        add_brace(f"Brace_ES_{abs(int(z))}", (xR - off, Y0, z), (X0, yB + off, z), bucket)

    # 8) Wells (perimeter + grid, then dedupe)
    wells: List[Well] = []
    wells += ring_wells("W", xL, yB, W, H, z_top=Z_TOP, z_bot=Z_WELL_BOTTOM, spacing=4.0, clearance=0.8)
    wells += grid_wells("W", xL, yB, W, H, z_top=Z_TOP, z_bot=Z_WELL_BOTTOM, dx=6.0, dy=6.0, margin=2.0)
    wells_unique, wells_dupes, _ = dedupe_wells_by_line(wells, tol=1e-6, keep="first")
    print(f"[DEDUPE] wells total={len(wells)}, unique={len(wells_unique)}, dupes={len(wells_dupes)}")
    for w in wells_unique:
        pit.add_structure("wells", w)

    # 9) Phases — use Phase API methods: set_inherits / activate_structures / set_water_table / set_well_overrides
    st_init  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=120, time_interval=0.5)
    st_exc1  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=140, time_interval=1.0)
    st_dewat = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=120, time_interval=0.5)
    st_exc2  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=140, time_interval=1.0)

    ph0 = Phase(name="P0_Initial",     settings=st_init)
    ph1 = Phase(name="P1_Excavate_L1", settings=st_exc1)
    ph2 = Phase(name="P2_Dewatering",  settings=st_dewat)
    ph3 = Phase(name="P3_Excavate_L2", settings=st_exc2)

    # inherits chain
    ph1.set_inherits(ph0)
    ph2.set_inherits(ph1)
    ph3.set_inherits(ph2)

    # ---------------- Soil activation / freezing BEFORE phases are added ----------------
    # Deactivate the enclosed (pit) pieces at the first excavation stage.
    if excav_names:
        ph1.set_soil_overrides(mk_soil_overrides(excav_names, active=False))
        # Deactivation carries forward via inheritance.

    # During dewatering, freeze the remaining large pieces for flow-only calculation.
    if remain_names:
        ph2.set_soil_overrides(mk_soil_overrides(remain_names, deformable=False))

    # Unfreeze them again for the next excavation stage.
    if remain_names:
        ph3.set_soil_overrides(mk_soil_overrides(remain_names, deformable=True))

    # activation lists (structures)
    ph0.activate_structures(west_wall, east_wall, south_wall, north_wall)
    ph1.activate_structures(*braces_L1)
    ph2.activate_structures(*wells_unique)
    ph3.activate_structures(*braces_L2)

    # per-phase well overrides + water table
    ph2.set_well_overrides({ w.name: {"h_min": -10.0, "q_well": 0.008} for w in wells_unique })

    x_min, x_max, y_min, y_max = proj.x_min, proj.x_max, proj.y_min, proj.y_max
    water_pts = [
        WaterLevel(x_min, y_min, -6.0), WaterLevel(x_max, y_min, -6.0),
        WaterLevel(x_max, y_max, -6.0), WaterLevel(x_min, y_max, -6.0),
        WaterLevel(0.0,    0.0,  -6.0),
    ]
    ph2.set_water_table(WaterLevelTable(water_pts))

    # register phases
    for p in (ph0, ph1, ph2, ph3):
        pit.add_phase(p)

    return pit


# --------------------------------- main ---------------------------------

if __name__ == "__main__":
    # Create runner first so assemble_pit() can query split soils in Stages
    runner = PlaxisRunner(HOST, PORT, PASSWORD)
    pit = assemble_pit(runner=runner)
    builder = ExcavationBuilder(runner, pit)

    # Ensure borehole layer materials reference shared library instances
    try:
        builder._relink_borehole_layers_to_library()
    except Exception:
        pass

    print("[BUILD] start …")
    builder.build()
    print("[BUILD] done.")

    print("[BUILD] Pick up soillayer.")
    excava_soils = builder.get_remaining_soil_names()
    print("[BUILD] Applied the soilayer status for phases.")

    print("[APPLY] apply phases …")
    pit.phases[1].add_soils(excava_soils[0])
    pit.phases[3].add_soils(excava_soils[1])
    builder.apply_pit_soil_block()
    print("[APPLY] Updated the status of soillayers")

    print(
        "\n[NOTE] Braces use 'ElasticBeam' as anchor-like members here."
        "\n[NOTE] PLAXIS assumes well discharge is evenly distributed along each well; "
        "if wells intersect other objects, consider splitting the well geometry."
    )


#########################################################################
# Next steps:
# 1. Create a standard excavation with retaining structures
# 2. Code the resultmapper to pick up the calculate result from the result panel.
# ####################################################################### 