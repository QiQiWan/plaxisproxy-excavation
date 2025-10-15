# testmapper.py  — Updated for new PhaseMapper (English comments)
from math import ceil
from typing import List, Tuple, Iterable, Any, Dict

from config.plaxis_config import HOST, PORT, PASSWORD

# Runner / Builder / Container
from src.plaxisproxy_excavation.plaxishelper.plaxisrunner import PlaxisRunner
from src.excavation_builder import ExcavationBuilder
from src.plaxisproxy_excavation.excavation import FoundationPit

# Core components
from src.plaxisproxy_excavation.components.projectinformation import ProjectInformation, Units
from src.plaxisproxy_excavation.components.phase import Phase
from src.plaxisproxy_excavation.components.phasesettings import PlasticStageSettings, LoadType
# ADD this import near other component imports
from src.plaxisproxy_excavation.components.phase import WaterLevelTable

# Boreholes & materials
from src.plaxisproxy_excavation.borehole import SoilLayer, BoreholeLayer, Borehole, BoreholeSet
from src.plaxisproxy_excavation.materials.soilmaterial import SoilMaterialFactory, SoilMaterialsType

# Geometry
from src.plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D

# Structure materials & structures
from src.plaxisproxy_excavation.materials.platematerial import ElasticPlate
from src.plaxisproxy_excavation.materials.beammaterial import ElasticBeam  # used here as anchor material
from src.plaxisproxy_excavation.structures.retainingwall import RetainingWall
from src.plaxisproxy_excavation.structures.beam import Beam
from src.plaxisproxy_excavation.structures.well import Well, WellType


# ----------------------------- small helpers -----------------------------

def rect_polygon_xy(x0: float, y0: float, z: float, w: float, h: float) -> Polygon3D:
    """Axis-aligned rectangle (z-constant) as a closed polygon (for horizontal faces)."""
    pts = [
        Point(x0,     y0,     z),
        Point(x0+w,   y0,     z),
        Point(x0+w,   y0+h,   z),
        Point(x0,     y0+h,   z),
        Point(x0,     y0,     z),
    ]
    return Polygon3D.from_points(PointSet(pts))

def rect_wall_x(x: float, y0: float, y1: float, z_top: float, z_bot: float) -> Polygon3D:
    """Vertical rectangle wall at fixed x (plane parallel to Z), top flush with ground."""
    pts = [
        Point(x, y0, z_top),
        Point(x, y1, z_top),
        Point(x, y1, z_bot),
        Point(x, y0, z_bot),
        Point(x, y0, z_top),
    ]
    return Polygon3D.from_points(PointSet(pts))

def rect_wall_y(y: float, x0: float, x1: float, z_top: float, z_bot: float) -> Polygon3D:
    """Vertical rectangle wall at fixed y (plane parallel to Z), top flush with ground."""
    pts = [
        Point(x0, y, z_top),
        Point(x1, y, z_top),
        Point(x1, y, z_bot),
        Point(x0, y, z_bot),
        Point(x0, y, z_top),
    ]
    return Polygon3D.from_points(PointSet(pts))

def line_2pts(p0: Tuple[float, float, float], p1: Tuple[float, float, float]) -> Line3D:
    """One straight line from two (x,y,z) points."""
    a = Point(*p0); b = Point(*p1)
    return Line3D(PointSet([a, b]))

def ring_wells(prefix: str, x0: float, y0: float, w: float, h: float,
               z_top: float, z_bot: float, spacing: float,
               clearance: float = 0.7) -> List[Well]:
    """
    Perimeter wells around the rectangle [x0,x0+w]x[y0,y0+h].
    Wells are inset by 'clearance' from the rectangle edges.
    """
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
                    h_min=z_bot,  # min head along the well
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
    """Interior grid of wells inside the rectangle."""
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


# ----------------------------- well line dedupe -----------------------------
# We deduplicate wells by their geometric line (two endpoints), direction-agnostic.
# Endpoints are quantized by 'tol' to avoid floating noise.
def _as_xyz(p: Any) -> Tuple[float, float, float]:
    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return float(p.x), float(p.y), float(p.z)
    try:
        return float(p[0]), float(p[1]), float(p[2])
    except Exception as e:
        raise TypeError(f"Point object not recognized: {p!r}") from e

def _extract_line_endpoints(line: Any) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    pts = getattr(line, "points", None)
    if pts and len(pts) >= 2:
        return _as_xyz(pts[0]), _as_xyz(pts[1])
    ps = getattr(line, "point_set", None) or getattr(line, "_point_set", None)
    if ps is not None:
        inner = getattr(ps, "points", None) or getattr(ps, "_points", None)
        if inner and len(inner) >= 2:
            return _as_xyz(inner[0]), _as_xyz(inner[1])
    start, end = getattr(line, "start", None), getattr(line, "end", None)
    if start is not None and end is not None:
        return _as_xyz(start), _as_xyz(end)
    try:
        it = list(iter(line))
        if len(it) >= 2:
            return _as_xyz(it[0]), _as_xyz(it[1])
    except Exception:
        pass
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
    """
    Remove duplicate wells that share the same geometric line (order-independent).
    Returns (unique, duplicates, key_to_master).
    """
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
                    unique.remove(prev)
                    dupes.append(prev)
                key_to_master[k] = w
                unique.append(w)
            else:
                dupes.append(w)
        else:
            key_to_master[k] = w
            unique.append(w)

    return unique, dupes, key_to_master


# ----------------------------- assemble pit -----------------------------

def assemble_pit() -> FoundationPit:
    """
    Build a rectangular pit with vertical diaphragm walls (parallel to Z),
    top flush with ground, deeper than pit bottom; horizontal braces on
    opposite and adjacent walls using anchor material; perimeter + grid wells (deduped).
    """

    # Project info (50 x 80 bounding box in XY as requested)
    proj = ProjectInformation(
        title="Rectangular Pit – Vertical DWall + Anchor Braces + Wells",
        company="Demo",
        dir=".",
        file_name="demo_excavation.p3d",
        comment="Builder/PhaseMapper integration demo",
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

    # ---- Soil materials ----
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

    # ---- Boreholes & layers (minimal but valid) ----
    sl_fill, sl_sand, sl_clay = SoilLayer("Fill", fill), SoilLayer("Sand", sand), SoilLayer("Clay", clay)
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

    # ---- Pit geometry ----
    X0, Y0 = 0.0, 0.0
    W, H   = 20.0, 16.0
    Z_TOP  = 0.0
    Z_L1   = -4.0       # first brace level
    Z_L2   = -8.0       # second brace level
    Z_EXC_BOTTOM = -10.0
    Z_WALL_BOTTOM = -16.0   # deeper than excavation bottom
    Z_WELL_BOTTOM = -18.0

    # ---- Structure materials ----
    dwall_mat  = ElasticPlate(name="DWall_E", E=30e6, nu=0.2, d=0.8,  gamma=25.0)
    anchor_mat = ElasticBeam(name="Anchor_E", E=35e6, nu=0.2, gamma=25.0)  # used as "anchor" material for braces
    pit.add_material("plate_materials", dwall_mat)
    pit.add_material("beam_materials",  anchor_mat)

    # ---- Vertical diaphragm walls (top flush with ground, bottom deeper than pit) ----
    xL, xR = X0 - W/2, X0 + W/2
    yB, yT = Y0 - H/2, Y0 + H/2
    west_wall  = RetainingWall(name="DWall_W", surface=rect_wall_x(xL, yB, yT, Z_TOP, Z_WALL_BOTTOM), plate_type=dwall_mat)
    east_wall  = RetainingWall(name="DWall_E", surface=rect_wall_x(xR, yB, yT, Z_TOP, Z_WALL_BOTTOM), plate_type=dwall_mat)
    south_wall = RetainingWall(name="DWall_S", surface=rect_wall_y(yB, xL, xR, Z_TOP, Z_WALL_BOTTOM), plate_type=dwall_mat)
    north_wall = RetainingWall(name="DWall_N", surface=rect_wall_y(yT, xL, xR, Z_TOP, Z_WALL_BOTTOM), plate_type=dwall_mat)
    for w in (west_wall, east_wall, south_wall, north_wall):
        pit.add_structure("retaining_walls", w)

    # ---- Horizontal braces (anchors) on opposite and adjacent walls ----
    # Use an inner offset so line endpoints lie inside the pit region (avoid coincident with wall edges).
    off = 0.8

    def add_brace(name: str, p0: Tuple[float, float, float], p1: Tuple[float, float, float]):
        pit.add_structure("beams", Beam(name=name, line=line_2pts(p0, p1), beam_type=anchor_mat))

    for z in (Z_L1, Z_L2):
        # Opposite walls (X and Y directions)
        add_brace(f"Brace_X_{abs(int(z))}",
                  (xL + off, Y0, z), (xR - off, Y0, z))
        add_brace(f"Brace_Y_{abs(int(z))}",
                  (X0, yB + off, z), (X0, yT - off, z))

        # Adjacent walls (diagonals among midpoints of adjacent walls)
        add_brace(f"Brace_WN_{abs(int(z))}",
                  (xL + off, Y0, z), (X0, yT - off, z))  # West -> North
        add_brace(f"Brace_WS_{abs(int(z))}",
                  (xL + off, Y0, z), (X0, yB + off, z))  # West -> South
        add_brace(f"Brace_EN_{abs(int(z))}",
                  (xR - off, Y0, z), (X0, yT - off, z))  # East -> North
        add_brace(f"Brace_ES_{abs(int(z))}",
                  (xR - off, Y0, z), (X0, yB + off, z))  # East -> South

    # ---- Wells: perimeter ring + interior grid (then dedupe) ----
    wells: List[Well] = []
    wells += ring_wells("W", xL, yB, W, H, z_top=Z_TOP, z_bot=Z_WELL_BOTTOM, spacing=4.0, clearance=0.7)
    wells += grid_wells("W", xL, yB, W, H, z_top=Z_TOP, z_bot=Z_WELL_BOTTOM, dx=6.0, dy=6.0, margin=2.0)
    wells_unique, wells_dupes, _ = dedupe_wells_by_line(wells, tol=1e-6, keep="first")
    print(f"[DEDUPE] wells total={len(wells)}, unique={len(wells_unique)}, dupes={len(wells_dupes)}")
    for d in wells_dupes:
        n = getattr(d, "name", None)
        if n: print("  duplicate well:", n)
    for w in wells_unique:
        pit.add_structure("wells", w)

    # ---- Phases ----
    st_init  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=120, time_interval=0.5)
    st_exc1  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=140, time_interval=1.0)
    st_dewat = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=120, time_interval=0.5)
    st_exc2  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=140, time_interval=1.0)

    ph0 = Phase(name="P0_Initial",    settings=st_init)   # walls active by default in builder
    ph1 = Phase(name="P1_Excavate_L1",settings=st_exc1)   # excavate to Z_L1, activate first-level braces
    ph2 = Phase(name="P2_Dewatering", settings=st_dewat)  # start wells + set water level
    ph3 = Phase(name="P3_Excavate_L2",settings=st_exc2)   # excavate to Z_L2, activate second-level braces

    # Optional per-phase settings if your SDK exposes helpers
    try:
        ph2.set_well_overrides({ w.name: {"h_min": -10.0, "q_well": 0.008} for w in wells_unique })
    except Exception:
        pass
    try:
        ph2.set_water_table(WaterLevelTable(head=-6.0))
    except Exception:
        pass

    for p in (ph0, ph1, ph2, ph3):
        pit.add_phase(p)

    return pit


# --------------------------------- main ---------------------------------

if __name__ == "__main__":
    runner = PlaxisRunner(HOST, PORT, PASSWORD)
    pit = assemble_pit()

    builder = ExcavationBuilder(runner, pit)

    # Optional: relink SoilLayer.material to the library (if your data source created copies)
    try:
        builder._relink_borehole_layers_to_library()
    except Exception:
        pass

    print("[BUILD] start …")
    builder.build()  # initial design only (materials/structures/walls/anchors/wells/mesh + phase shells)
    print("[BUILD] done.")

    # Apply all phases (options + structures + water table + well overrides)
    print("[APPLY] apply phases …")
    result = builder.apply_phases(warn_on_missing=True)
    print("[APPLY] result:", result)

    print(
        "\n[NOTE] Braces use 'ElasticBeam' as anchor material. If your SDK has a specific AnchorMaterial, "
        "replace 'ElasticBeam' with that class and update properties accordingly.\n"
        "[NOTE] PLAXIS assumes well discharge is distributed evenly over the well length. "
        "Intersections with other objects may lead to non-physical flow results; "
        "adjust geometry or split wells if needed."
    )
