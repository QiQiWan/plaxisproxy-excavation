"""
Rewritten pipeline with FoundationPit + ExcavationBuilder

Flow:
1) Assemble FoundationPit: materials -> boreholes/layers -> structures (incl. wells) -> phases
2) builder.build(): initial design only (project/boreholes/materials/structures/loads/mesh + phase shells)
3) (optional) ensure soil volume & mesh
4) builder.apply_phases(): create+apply phases (options, activation/deactivation, water table, well overrides)
All comments are in English.
"""

from typing import List, Dict, Any, Tuple

# --- plaxis runner + builder & pit (adjust import paths to your project) ---
from config.plaxis_config import HOST, PORT, PASSWORD
from src.plaxisproxy_excavation.plaxishelper.plaxisrunner import PlaxisRunner
from src.excavation_builder import ExcavationBuilder                    # <- where your class lives
from src.plaxisproxy_excavation.excavation import FoundationPit

# --- geometry & domain types ---
from src.plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D
from src.plaxisproxy_excavation.materials.soilmaterial import MCMaterial
from src.plaxisproxy_excavation.borehole import SoilLayer, BoreholeLayer, Borehole, BoreholeSet

# structure domain classes
from src.plaxisproxy_excavation.structures.retainingwall import RetainingWall
from src.plaxisproxy_excavation.structures.beam import Beam
from src.plaxisproxy_excavation.structures.anchor import Anchor
from src.plaxisproxy_excavation.structures.embeddedpile import EmbeddedPile
from src.plaxisproxy_excavation.structures.well import Well, WellType

# structure materials
from src.plaxisproxy_excavation.materials.platematerial import ElasticPlate
from src.plaxisproxy_excavation.materials.beammaterial import ElasticBeam
from src.plaxisproxy_excavation.materials.pilematerial import ElasticPile
from src.plaxisproxy_excavation.materials.anchormaterial import ElasticAnchor

# phases & settings
from src.plaxisproxy_excavation.components.phase import Phase
from src.plaxisproxy_excavation.components.phasesettings import PlasticStageSettings, LoadType


# ---------------------------- helpers ----------------------------
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

def ensure_min_soil_body(g_i, xmin=-30, ymin=-30, zmin=-30, xmax=30, ymax=30, zmax=0) -> bool:
    """Ensure at least one soil volume/surface exists before meshing."""
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


# ---------------------------- assembling the pit ----------------------------
def assemble_foundation_pit() -> FoundationPit:
    """Create a FoundationPit holding all inputs needed by the builder."""
    # 0) Pit metadata
    footprint = make_rect_polygon_xy(x0=-10.0, y0=-6.0, z=0.0, width=20.0, height=12.0)
    pit = FoundationPit(
        name="DemoPit_BHFirst",
        footprint=footprint,
        depth=12.0,
    )

    # 1) Materials (soil + structures)
    fill   = MCMaterial(name="Fill",   E_ref=15e6, c_ref=5e3,  phi=25.0, psi=0.0, gamma=18.0, gamma_sat=20.0)
    sand   = MCMaterial(name="Sand",   E_ref=35e6, c_ref=1e3,  phi=32.0, psi=2.0,  gamma=19.0, gamma_sat=21.0)
    clay   = MCMaterial(name="Clay",   E_ref=12e6, c_ref=15e3, phi=22.0, psi=0.0, gamma=17.0, gamma_sat=19.0)
    gravel = MCMaterial(name="Gravel", E_ref=60e6, c_ref=0.5e3,phi=38.0, psi=5.0,  gamma=20.0, gamma_sat=22.0)

    plate_mat  = ElasticPlate(name="Plate_E", E=30e6, nu=0.2, d=0.5, gamma=25.0)
    beam_mat   = ElasticBeam(name="Beam_E",  E=30e6, nu=0.2, gamma=25.0)
    pile_mat   = ElasticPile(name="Pile_E",  E=30e6, nu=0.2, gamma=25.0, diameter=1.0)
    anchor_mat = ElasticAnchor(name="Anchor_E", EA=1.0e6)

    # The builder accepts a dict-of-lists (category keys are arbitrary)
    pit.materials = {
        "soils":   [fill, sand, clay, gravel],
        "plates":  [plate_mat],
        "beams":   [beam_mat],
        "piles":   [pile_mat],
        "anchors": [anchor_mat],
    }

    # 2) Boreholes & Layers (first)
    sl_fill   = SoilLayer(name="Fill",   material=fill)
    sl_sand   = SoilLayer(name="Sand",   material=sand)
    sl_clay   = SoilLayer(name="Clay",   material=clay)
    sl_gravel = SoilLayer(name="Gravel", material=gravel)

    bh1_layers = [BoreholeLayer("Fill@BH1", 0.0, -1.5, sl_fill),
                  BoreholeLayer("Sand@BH1",-1.5, -8.0, sl_sand),
                  BoreholeLayer("Clay@BH1",-8.0, -12.0, sl_clay)]
    bh1 = Borehole("BH_1", Point(0, 0, 0), 0.0, layers=bh1_layers, water_head=-2.0)

    bh2_layers = [BoreholeLayer("Fill@BH2", 0.0, -2.0, sl_fill),
                  BoreholeLayer("Sand@BH2",-2.0, -6.0, sl_sand),
                  BoreholeLayer("Clay@BH2",-6.0, -10.0, sl_clay)]
    bh2 = Borehole("BH_2", Point(12, 0, 0), 0.0, layers=bh2_layers, water_head=-1.5)

    bh3_layers = [BoreholeLayer("Sand@BH3", 0.0, -4.0, sl_sand),
                  BoreholeLayer("Clay@BH3",-4.0, -9.0, sl_clay)]
    bh3 = Borehole("BH_3", Point(24, 0, 0), 0.0, layers=bh3_layers, water_head=-1.0)

    bh4_layers = [BoreholeLayer("Fill@BH4", 0.0, -1.0, sl_fill),
                  BoreholeLayer("Gravel@BH4",-1.0, -7.5, sl_gravel)]
    bh4 = Borehole("BH_4", Point(36, 0, 0), 0.0, layers=bh4_layers, water_head=-1.2)

    pit.borehole_set = BoreholeSet(name="Site BH", boreholes=[bh1, bh2, bh3, bh4], comment="Full demo set")

    # 3) Structures (geometry)
    line_beam   = make_line(Point(0,  2,  0),   Point(0,  2, -10))
    line_pile   = make_line(Point(4,  0,  0),   Point(4,  0, -15))
    line_anchor = make_line(Point(0, -2, -2),   Point(6, -2,  -4))
    line_waler  = make_line(Point(5.0, -2.0, -4.0), Point(7.0, -2.0, -4.0))

    poly_wall   = make_rect_polygon_xy(x0=-1.0, y0=-3.0, z=-2.0, width=2.0, height=2.0)  # plate @ z=-2
    poly_plate  = make_rect_polygon_xy(x0= 2.5, y0=-1.0, z= 0.0, width=3.0, height=2.0)  # plate @ z=0

    beam1    = Beam(name="B1",            line=line_beam,   beam_type=beam_mat)
    waler    = Beam(name="Waler_L2",      line=line_waler,  beam_type=beam_mat)
    pile1    = EmbeddedPile(name="P1",    line=line_pile,   pile_type=pile_mat)
    anchor1  = Anchor(name="A1",          line=line_anchor, anchor_type=anchor_mat)
    wall1    = RetainingWall(name="WALL",   surface=poly_wall,  plate_type=plate_mat)
    plate1   = RetainingWall(name="PLATE",  surface=poly_plate, plate_type=plate_mat)

    well1    = Well(name="W1",
                    line=make_line(Point(10, 0, 0), Point(10, 0, -8)),
                    well_type=WellType.Extraction, h_min=-5.0)

    pit.structures = {
        "retaining_walls": [wall1, plate1],
        "beams":           [beam1, waler],
        "anchors":         [anchor1],
        "embedded_piles":  [pile1],
        "wells":           [well1],   # wells are created at structure stage; per-phase params applied later
        "soil_blocks":     [],        # optional
    }

    # 4) Phases & settings (each must inherit from a base phase)
    st_init     = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=100,
                                       time_interval=0.5,  over_relaxation_factor=1.05, ΣM_weight=1.0)
    st_exc1     = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=150,
                                       time_interval=1.0,  over_relaxation_factor=1.10, ΣM_stage=0.7, ΣM_weight=1.0)
    st_exc2     = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=160,
                                       time_interval=1.0,  over_relaxation_factor=1.10, ΣM_stage=0.8, ΣM_weight=1.0)
    st_dewater  = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=120,
                                       time_interval=0.5,  over_relaxation_factor=1.05, ΣM_weight=1.0)
    st_backfill = PlasticStageSettings(load_type=LoadType.StageConstruction, max_steps=120,
                                       time_interval=1.0,  over_relaxation_factor=1.05, ΣM_stage=0.5, ΣM_weight=1.0)

    # Initial inherits from "InitialPhase" (the builder will fetch it internally)
    phase0 = Phase(
        name="Phase0_InitialSupports",
        comment="Activate base supports the anchor depends on.",
        settings=st_init,
        activate=[wall1, waler, plate1, pile1],
        deactivate=[],
        inherits=None,   # builder.apply_phases() will replace None with InitialPhase handle for the first one
    )
    phase1 = Phase(
        name="Phase1_Excavation_1",
        comment="Excavate to L1; bring Beam B1 and Anchor A1 online.",
        settings=st_exc1,
        activate=[beam1, anchor1],
        deactivate=[],
        inherits=phase0,
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
        activate=[],     # structures already created; we just change well params in this phase
        deactivate=[],
        inherits=phase2,
        # per-phase well parameter overrides (applied in apply_phases via PhaseMapper)
        well_overrides={"W1": {"q_well": 800.0, "h_min": -6.0, "well_type": "Extraction"}},
    )
    phase4 = Phase(
        name="Phase4_Backfill_RemoveTemps",
        comment="Backfill; remove temporary supports.",
        settings=st_backfill,
        activate=[],
        deactivate=[anchor1, beam1],
        inherits=phase3,
    )

    pit.phases = [phase0, phase1, phase2, phase3, phase4]
    return pit


# ---------------------------- main ----------------------------
if __name__ == "__main__":
    # Assemble the pit object (data-only)
    pit = assemble_foundation_pit()

    # Create builder + connect through runner (builder.build will also connect if needed)
    runner = PlaxisRunner(PORT, PASSWORD, HOST)  # or let ExcavationBuilder construct internally
    builder = ExcavationBuilder(runner, pit)

    # 1) Build ONLY the initial design (no per-phase updates yet)
    summary_build = builder.build()
    print("[BUILD] initial design summary:", summary_build)

    # 2) (Optional) ensure at least one soil body exists then mesh (when not provided by builder)
    try:
        created_body = ensure_min_soil_body(builder.App.g_i)
        print(f"[GEOMETRY] Soil body present? {created_body}")
        optional_mesh(builder.App.g_i)
    except Exception as e:
        print(f"[GEOMETRY] Optional mesh step skipped: {e}")

    # 3) Apply phases (this will handle options, activations, WATER TABLE (if set), and WELL OVERRIDES)
    summary_phases = builder.apply_phases(warn_on_missing=True)
    print("[PHASES] apply summary:", summary_phases)

    print("\n[PIPELINE] Boreholes & layers -> geometry -> structures -> mesh -> stages -> inherited phases: COMPLETE")
