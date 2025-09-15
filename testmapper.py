"""
test_boreholes_layers_then_phases_structures.py

Pipeline:
1) Materials (soil + structure) -> Boreholes & Layers FIRST
2) Ensure at least one soil volume (geometry safety)
3) Create structures via StructureMappers (wall/plate/beam/anchor/pile/well)
4) Mesh (optional)
5) Go to Stages, fetch InitialPhase as a Phase object
6) For each new Phase: must inherit from a base phase; create via mapper (writes plx_id), then:
   - apply options
   - apply structure activation/deactivation (freeze/activate)

All comments are in English.
"""

from typing import Iterable, List, Dict, Tuple, Any, Optional
from plxscripting.server import new_server

# ---------- domain imports ----------
from src.plaxisproxy_excavation.plaxishelper.boreholemapper import BoreholeSetMapper
from src.plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D
from src.plaxisproxy_excavation.materials.soilmaterial import MCMaterial
from src.plaxisproxy_excavation.borehole import SoilLayer, BoreholeLayer, Borehole, BoreholeSet

# Material mappers
from src.plaxisproxy_excavation.plaxishelper.materialmapper import (
    SoilMaterialMapper, PlateMaterialMapper,
    BeamMaterialMapper, PileMaterialMapper, AnchorMaterialMapper,
)

# Structure mappers
from src.plaxisproxy_excavation.plaxishelper.structuremapper import (
    RetainingWallMapper, BeamMapper, EmbeddedPileMapper,
    AnchorMapper, WellMapper,
)

# Structure domain classes
from src.plaxisproxy_excavation.structures.retainingwall import RetainingWall  # Plate as wall
from src.plaxisproxy_excavation.structures.beam import Beam
from src.plaxisproxy_excavation.structures.anchor import Anchor
from src.plaxisproxy_excavation.structures.embeddedpile import EmbeddedPile
from src.plaxisproxy_excavation.structures.well import Well, WellType

# Structure materials
from src.plaxisproxy_excavation.materials.platematerial import ElasticPlate
from src.plaxisproxy_excavation.materials.beammaterial import ElasticBeam
from src.plaxisproxy_excavation.materials.pilematerial import ElasticPile
from src.plaxisproxy_excavation.materials.anchormaterial import ElasticAnchor

# Phases & settings
from src.plaxisproxy_excavation.components.phase import Phase
from src.plaxisproxy_excavation.components.phasesettings import PlasticStageSettings, LoadType
from src.plaxisproxy_excavation.plaxishelper.phasemapper import PhaseMapper


# ---------------------------- helpers ----------------------------
def assert_plx_id_set(obj: Any, name: str) -> None:
    plx_id = getattr(obj, "plx_id", None)
    assert plx_id is not None, f"{name}.plx_id should be set after creation."

def iter_unique(items: Iterable[object]) -> List[object]:
    """Deduplicate by object identity while keeping order."""
    seen = set()
    out: List[object] = []
    for it in items:
        oid = id(it)
        if oid in seen:
            continue
        seen.add(oid)
        out.append(it)
    return out

def print_summary(summary: Dict[str, List[Tuple[float, float]]]) -> None:
    """Pretty print zone summary: {layer_name: [(top,bottom)_bh0, (top,bottom)_bh1, ...]}"""
    print("\n=== Zone Summary (top, bottom) per layer per borehole ===")
    for lname, pairs in summary.items():
        cells = ", ".join([f"BH{i}:({t:.3g},{b:.3g})" for i, (t, b) in enumerate(pairs)])
        print(f"  - {lname}: {cells}")

def ensure_min_soil_body(g_i, xmin=-30, ymin=-30, zmin=-30, xmax=30, ymax=30, zmax=0) -> bool:
    """Ensure at least one volume exists."""
    try:
        vols = getattr(g_i, "SoilVolumes", None) or getattr(g_i, "Volumes", None)
        surfs = getattr(g_i, "Surfaces", None) or getattr(g_i, "Planes", None)
        for coll in (vols, surfs):
            try:
                if coll and any(True for _ in coll):
                    return True
            except Exception:
                pass
    except Exception:
        pass
    fn = getattr(g_i, "SoilContour", None)
    if callable(fn):
        try:
            fn(xmin, ymin, zmin, xmax, ymax, zmax)
            return True
        except Exception:
            pass
    create = getattr(g_i, "create", None)
    if callable(create):
        try:
            create("soil", xmin, ymin, zmin, xmax, ymax, zmax)
            return True
        except Exception:
            pass
    return False

def optional_mesh(g_i) -> None:
    """Try to mesh if available; ignore failures (build-dependent)."""
    for fn_name in ("gotomesh", "GoToMesh", "to_mesh"):
        fn = getattr(g_i, fn_name, None)
        if callable(fn):
            try:
                fn(); break
            except Exception:
                pass
    for fn_name in ("mesh", "Mesh", "createmesh", "CreateMesh"):
        fn = getattr(g_i, fn_name, None)
        if callable(fn):
            try:
                fn(); break
            except Exception:
                pass

# ---------- geometry helpers ----------
def make_line(p0: Point, p1: Point) -> Line3D:
    """Construct a fresh Line3D from two Points (no shared instances)."""
    return Line3D(PointSet([Point(p0.x, p0.y, p0.z), Point(p1.x, p1.y, p1.z)]))

def make_rect_polygon_xy(x0: float, y0: float, z: float, width: float, height: float) -> Polygon3D:
    """Horizontal rectangle on plane z = const (Polygon3D requires constant z)."""
    pts = [
        Point(x0,           y0,            z),
        Point(x0 + width,   y0,            z),
        Point(x0 + width,   y0 + height,   z),
        Point(x0,           y0 + height,   z),
        Point(x0,           y0,            z),
    ]
    return Polygon3D.from_points(PointSet(pts))


# ---------------------------- materials ----------------------------
def make_soil_materials(g_i):
    fill   = MCMaterial(name="Fill",   E_ref=15e6, c_ref=5e3,  phi=25.0, psi=0.0, gamma=18.0, gamma_sat=20.0)
    sand   = MCMaterial(name="Sand",   E_ref=35e6, c_ref=1e3,  phi=32.0, psi=2.0,  gamma=19.0, gamma_sat=21.0)
    clay   = MCMaterial(name="Clay",   E_ref=12e6, c_ref=15e3, phi=22.0, psi=0.0, gamma=17.0, gamma_sat=19.0)
    gravel = MCMaterial(name="Sand",   E_ref=60e6, c_ref=0.5e3,phi=38.0, psi=5.0,  gamma=20.0, gamma_sat=22.0)
    for m in (fill, sand, clay, gravel):
        SoilMaterialMapper.create_material(g_i, m)
    for m, n in [(fill, "Fill"), (sand, "Sand"), (clay, "Clay"), (gravel, "Gravel(Sand-dup)")]:
        assert_plx_id_set(m, n)
    return fill, sand, clay, gravel

def make_structure_materials(g_i):
    plate_mat  = ElasticPlate(name="Plate_E", E=30e6, nu=0.2, d=0.5, gamma=25.0)
    beam_mat   = ElasticBeam(name="Beam_E",  E=30e6, nu=0.2, gamma=25.0)
    pile_mat   = ElasticPile(name="Pile_E",  E=30e6, nu=0.2, gamma=25.0, diameter=1.0)
    anchor_mat = ElasticAnchor(name="Anchor_E", EA=1.0e6)

    PlateMaterialMapper.create_material(g_i, plate_mat)
    BeamMaterialMapper.create_material(g_i,  beam_mat)
    PileMaterialMapper.create_material(g_i,  pile_mat)
    AnchorMaterialMapper.create_material(g_i, anchor_mat)

    for m, n in [(plate_mat, "PlateMat"), (beam_mat, "BeamMat"), (pile_mat, "PileMat"), (anchor_mat, "AnchorMat")]:
        assert_plx_id_set(m, n)
    return plate_mat, beam_mat, pile_mat, anchor_mat


# ---------------------------- boreholes & layers ----------------------------
def build_borehole_set(fill_mat, sand_mat, clay_mat, gravel_mat) -> Tuple[BoreholeSet, List[SoilLayer]]:
    sl_fill   = SoilLayer(name="Fill",   material=fill_mat)
    sl_sand   = SoilLayer(name="Sand",   material=sand_mat)
    sl_clay   = SoilLayer(name="Clay",   material=clay_mat)
    sl_gravel = SoilLayer(name="Gravel", material=gravel_mat)

    # BH_1
    bh1_layers = [
        BoreholeLayer("Fill@BH1",   0.0,  -1.5, sl_fill),
        BoreholeLayer("Sand@BH1",  -1.5, -8.0,  sl_sand),
        BoreholeLayer("Clay@BH1",  -8.0, -12.0, sl_clay),
    ]
    bh1 = Borehole("BH_1", Point(0, 0, 0), 0.0, layers=bh1_layers, water_head=-2.0)

    # BH_2 (named BH_1 to trigger conflict)
    bh2_layers = [
        BoreholeLayer("Fill@BH2",   0.0,  -2.0, sl_fill),
        BoreholeLayer("Sand@BH2",  -2.0, -6.0,  sl_sand),
        BoreholeLayer("Clay@BH2",  -6.0, -10.0, sl_clay),
    ]
    bh2 = Borehole("BH_1", Point(12, 0, 0), 0.0, layers=bh2_layers, water_head=-1.5)

    # BH_3 (missing Fill)
    bh3_layers = [
        BoreholeLayer("Sand@BH3",   0.0,  -4.0, sl_sand),
        BoreholeLayer("Clay@BH3",  -4.0,  -9.0, sl_clay),
    ]
    bh3 = Borehole("BH_3", Point(24, 0, 0), 0.0, layers=bh3_layers, water_head=-1.0)

    # BH_4 (Gravel instead of Sand/Clay)
    bh4_layers = [
        BoreholeLayer("Fill@BH4",   0.0,  -1.0, sl_fill),
        BoreholeLayer("Gravel@BH4", -1.0, -7.5, sl_gravel),
    ]
    bh4 = Borehole("BH_4", Point(36, 0, 0), 0.0, layers=bh4_layers, water_head=-1.2)

    bhset = BoreholeSet(name="Site BH", boreholes=[bh1, bh2, bh3, bh4], comment="Full demo set")
    return bhset, [sl_fill, sl_sand, sl_clay, sl_gravel]


# ---------------------------- main ----------------------------
if __name__ == "__main__":
    # 0) Connect
    passwd = "yS9f$TMP?$uQ@rW3"
    s_i, g_i = new_server("localhost", 10000, password=passwd)
    s_i.new()

    # 1) MATERIALS (soil)
    fill_mat, sand_mat, clay_mat, gravel_mat = make_soil_materials(g_i)

    # 2) BOREHOLES & LAYERS (FIRST)
    bhset, global_layers = build_borehole_set(fill_mat, sand_mat, clay_mat, gravel_mat)
    bhset.ensure_unique_names()
    summary = BoreholeSetMapper.create(g_i, bhset, normalize=True)

    for i, bh in enumerate(bhset.boreholes):
        assert_plx_id_set(bh, f"Borehole[{i}]")
    uniq_layers = iter_unique(sl for bh in bhset.boreholes for sl in [ly.soil_layer for ly in bh.layers])
    for sl in uniq_layers:
        assert_plx_id_set(sl, f"SoilLayer[{sl.name}]")
    print_summary(summary)
    print("\n[CHECK] Borehole & layers created and imported -> OK")

    # 3) GEOMETRY SAFETY
    created_body = ensure_min_soil_body(g_i)
    print(f"[GEOMETRY] Soil body present? {created_body}")

    # 4) STRUCTURE MATERIALS
    plate_mat, beam_mat, pile_mat, anchor_mat = make_structure_materials(g_i)

    # 5) STRUCTURE GEOMETRY
    line_beam   = make_line(Point(0,  2,  0),   Point(0,  2, -10))
    line_pile   = make_line(Point(4,  0,  0),   Point(4,  0, -15))
    line_anchor = make_line(Point(0, -2, -2),   Point(6, -2,  -4))
    line_waler  = make_line(Point(5.0, -2.0, -4.0), Point(7.0, -2.0, -4.0))

    poly_wall   = make_rect_polygon_xy(x0=-1.0, y0=-3.0, z=-2.0, width=2.0, height=2.0)  # plate @ z=-2
    poly_plate  = make_rect_polygon_xy(x0= 2.5, y0=-1.0, z= 0.0, width=3.0, height=2.0)  # extra plate @ z=0

    # 6) STRUCTURES
    beam1    = Beam(name="B1",            line=line_beam,   beam_type=beam_mat)
    waler    = Beam(name="Waler_L2",      line=line_waler,  beam_type=beam_mat)
    pile1    = EmbeddedPile(name="P1",    line=line_pile,   pile_type=pile_mat)
    anchor1  = Anchor(name="A1",          line=line_anchor, anchor_type=anchor_mat)
    wall1    = RetainingWall(name="WALL",   surface=poly_wall,  plate_type=plate_mat)
    plate1   = RetainingWall(name="PLATE",  surface=poly_plate, plate_type=plate_mat)

    # 7) CREATE STRUCTURES VIA MAPPERS (anchor depends on plate/waler -> create them first)
    RetainingWallMapper.create(g_i, wall1)
    BeamMapper.create(g_i, waler)
    BeamMapper.create(g_i, beam1)
    EmbeddedPileMapper.create(g_i, pile1)
    RetainingWallMapper.create(g_i, plate1)
    AnchorMapper.create(g_i, anchor1)

    WellMapper.create(g_i, Well(name="W1",
                                line=make_line(Point(10, 0, 0), Point(10, 0, -8)),
                                well_type=WellType.Extraction, h_min=-5.0))

    for s, n in [(beam1, "Beam B1"), (waler, "Waler_L2"), (pile1, "EmbeddedPile P1"),
                 (anchor1, "Anchor A1"), (wall1, "Plate WALL@z=-2"), (plate1, "Plate PLATE@z=0")]:
        assert_plx_id_set(s, n)
    print("[CHECK] IDs after structure creation -> OK")

    # 8) (OPTIONAL) MESH AFTER STRUCTURES EXIST
    optional_mesh(g_i)

    # 9) GO TO STAGES, FETCH INITIAL PHASE OBJECT (with bound plx_id)
    PhaseMapper.goto_stages(g_i)
    initial_phase = PhaseMapper.get_initial_phase(g_i)
    assert getattr(initial_phase, "plx_id", None) is not None, "Initial phase handle not found."

    # 10) SETTINGS
    st_init = PlasticStageSettings(load_type=LoadType.StageConstruction,
                                   max_steps=100, time_interval=0.5, over_relaxation_factor=1.05, ΣM_weight=1.0)
    st_exc1 = PlasticStageSettings(load_type=LoadType.StageConstruction,
                                   max_steps=150, time_interval=1.0, over_relaxation_factor=1.10, ΣM_stage=0.7, ΣM_weight=1.0)
    st_exc2 = PlasticStageSettings(load_type=LoadType.StageConstruction,
                                   max_steps=160, time_interval=1.0, over_relaxation_factor=1.10, ΣM_stage=0.8, ΣM_weight=1.0)
    st_dewater = PlasticStageSettings(load_type=LoadType.StageConstruction,
                                      max_steps=120, time_interval=0.5, over_relaxation_factor=1.05, ΣM_weight=1.0)
    st_backfill = PlasticStageSettings(load_type=LoadType.StageConstruction,
                                       max_steps=120, time_interval=1.0, over_relaxation_factor=1.05, ΣM_stage=0.5, ΣM_weight=1.0)

    # 11) PHASE OBJECTS (each MUST inherit from a base phase)
    phase0 = Phase(
        name="Phase0_InitialSupports",
        comment="Activate base supports the anchor depends on.",
        settings=st_init,
        activate=[wall1, waler, plate1, pile1],
        deactivate=[],
        inherits=initial_phase,  # <- MUST inherit
    )
    phase1 = Phase(
        name="Phase1_Excavation_1",
        comment="Excavate to L1; bring Beam B1 and Anchor A1 online.",
        settings=st_exc1,
        activate=[beam1, anchor1],
        deactivate=[],
        inherits=phase0,  # <- chain
    )
    phase2 = Phase(
        name="Phase2_Excavation_2",
        comment="Excavate to L2; keep temporary members.",
        settings=st_exc2,
        activate=[],
        deactivate=[],
        inherits=phase1,
    )
    phase3 = Phase(
        name="Phase3_Dewatering",
        comment="Start well W1.",
        settings=st_dewater,
        activate=[],
        deactivate=[],
        inherits=phase2,
    )
    phase4 = Phase(
        name="Phase4_Backfill_RemoveTemps",
        comment="Backfill; remove temporary supports.",
        settings=st_backfill,
        activate=[],
        deactivate=[anchor1, beam1],
        inherits=phase3,
    )

    # 12) CREATE & APPLY: for each Phase -> create_for_phase (writes plx_id) -> apply_phase
    ph_handles: List[Any] = []
    for ph in [phase0, phase1, phase2, phase3, phase4]:
        h = PhaseMapper.create(g_i, phase_obj=ph)                 # inherits from ph.inherits
        ph_handles.append(h)
        PhaseMapper.apply_phase(g_i, h, ph, warn_on_missing=True) # 1) options  2) activate/deactivate  3) water

        print(f"[{ph.name}] created handle bound; activate={len(ph.activate)} deactivate={len(ph.deactivate)}")

    # 13) Example: later option update on Phase2
    phase2.settings.max_iterations = 80
    PhaseMapper.apply_options(ph_handles[2], phase2.settings_payload(), warn_on_missing=True)
    print("[UPDATE] Phase2 max_iterations -> 80")

    # 14) Example: late structure change at Phase2 (explicit freeze)
    PhaseMapper.apply_structures(g_i, ph_handles[2], activate=[], deactivate=[beam1])
    print("[STRUCT] Phase2 additionally deactivated Beam B1")

    print("\n[PIPELINE] Boreholes & layers -> geometry -> structures -> mesh -> stages -> inherited phases: COMPLETE")