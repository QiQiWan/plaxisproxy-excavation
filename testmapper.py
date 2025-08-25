from src.plaxisproxy_excavation.geometry import *
from src.plaxisproxy_excavation.materials.beammaterial import *
from src.plaxisproxy_excavation.materials.soilmaterial import *
from src.plaxisproxy_excavation.plaxishelper.materialmapper import *
from src.plaxisproxy_excavation.materials.anchormaterial import *
from plxscripting.server import new_server

passwd = 'yS9f$TMP?$uQ@rW3'
s_i, g_i = new_server('localhost', 10000, password=passwd)
s_i.new()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
created_piles: list[tuple[str, object]] = []
created_anchors: list[tuple[str, object]] = []

def _make_pile(g_i, mat):
    """Create pile via mapper, collect and log handle."""
    plx = PileMaterialMapper.create_material(g_i, mat)
    created_piles.append((mat.name, plx))
    return plx

# -----------------------------------------------------------------------------
# region Create Materials
# -----------------------------------------------------------------------------

# -------- Soil Materials --------
mc = MCMaterial("MC", SoilMaterialsType.MC, "", 22, 5e3, 0.3, 25, 0.4)
soil_mat_mc = SoilMaterialMapper.create_material(g_i, mc)


mcc = MCCMaterial("MCC", SoilMaterialsType.MCC, "", 22, 5e3, 0.3, 25, 0.4)
soil_mat_mcc = SoilMaterialMapper.create_material(g_i, mcc)

hss = HSSMaterial("HSS", SoilMaterialsType.HSS, "", 22, 5e3, 0.3, 25, 0.4)
soil_mat_hss = SoilMaterialMapper.create_material(g_i, hss)

# -------- Plate Materials --------
ep = ElasticPlate(
    name="Slab_EL_ISO",
    type=PlateType.Elastic,
    comment="isotropic elastic plate",
    gamma=24.0,
    E=30e6,
    d=1.0,
    nu=0.2,
    preventpunch=True,
    isotropic=True
)
plx_plate_1 = PlateMaterialMapper.create_material(g_i, ep)

ep_aniso = ElasticPlate(
    name="Slab_EL_ANISO",
    type=PlateType.Elastic,
    comment="orthotropic elastic plate",
    gamma=24.0,
    E=30e6, nu=0.2,
    d=1.0,
    preventpunch=True,
    isotropic=False,
    E2=20e6, G12=12e6, G13=10e6, G23=9e6
)
plx_plate_2 = PlateMaterialMapper.create_material(g_i, ep_aniso)

epp = ElastoplasticPlate(
    name="Slab_EP",
    type=PlateType.Elastoplastic,
    comment="elasto-plastic plate",
    gamma=24.0,
    E=30e6, nu=0.2,
    d=1.0,
    preventpunch=True, isotropic=False,
    sigma_y_11=350e3, W_11=0.05,
    sigma_y_22=300e3, W_22=0.04
)
plx_plate_3 = PlateMaterialMapper.create_material(g_i, epp)

# -------- Beam Materials --------
beam_el_cyl = ElasticBeam(
    name="Beam_EL_Cylinder",
    type=BeamType.Elastic,
    comment="Elastic beam - solid circular section",
    gamma=25.0,
    E=30e6, nu=0.20,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.Cylinder,
    diameter=0.60,
    RayleighAlpha=0.0, RayleighBeta=0.0,
)
plx_beam_el_cyl = BeamMaterialMapper.create_material(g_i, beam_el_cyl)

beam_el_rect = ElasticBeam(
    name="Beam_EL_Rect",
    type=BeamType.Elastic,
    comment="Elastic beam - rectangular section (b=0.4, h=0.6 m)",
    gamma=25.0,
    E=32e6, nu=0.22,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.Rectangle,
    width=0.40,
    height=0.60,
)
plx_beam_el_rect = BeamMaterialMapper.create_material(g_i, beam_el_rect)


beam_el_tube = ElasticBeam(
    name="Beam_EL_Tube",
    type=BeamType.Elastic,
    comment="Elastic beam - circular tube (Ro=0.5, Ri=0.4 m)",
    gamma=24.0,
    E=28e6, nu=0.25,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.CircularArcBeam,
    diameter=1.00,
    thickness=0.1,
)
plx_beam_el_tube = BeamMaterialMapper.create_material(g_i, beam_el_tube)


beam_el_custom = ElasticBeam(
    name="Beam_EL_Custom",
    type=BeamType.Elastic,
    comment="Elastic beam - custom section",
    gamma=24.0,
    E=30e6, nu=0.20,
    cross_section=CrossSectionType.Custom,
    predefined_section=None,
    A=0.36, Iy=0.012, Iz=0.009, W=0.020,
)
plx_beam_el_custom = BeamMaterialMapper.create_material(g_i, beam_el_custom)


beam_ep_rect = ElastoplasticBeam(
    name="Beam_EP_Rect",
    type=BeamType.Elastoplastic,
    comment="Elasto-plastic beam - rectangular with yield",
    gamma=25.0,
    E=30e6, nu=0.20,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.Rectangle,
    width=0.50, height=0.80,
    sigma_y=350e3,
    yield_dir=1,
)
plx_beam_ep_rect = BeamMaterialMapper.create_material(g_i, beam_ep_rect)


beam_ep_cyl = ElastoplasticBeam(
    name="Beam_EP_Cylinder",
    type=BeamType.Elastoplastic,
    comment="Elasto-plastic beam - solid circular",
    gamma=25.0,
    E=31e6, nu=0.22,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.Cylinder,
    diameter=0.50,
    sigma_y=300e3,
    yield_dir="local-2",
)
plx_beam_ep_cyl = BeamMaterialMapper.create_material(g_i, beam_ep_cyl)


beam_ep_custom = ElastoplasticBeam(
    name="Beam_EP_Custom",
    type=BeamType.Elastoplastic,
    comment="Elasto-plastic beam - custom section with yield",
    gamma=24.0,
    E=31e6, nu=0.20,
    cross_section=CrossSectionType.Custom,
    predefined_section=None,
    A=0.42, Iy=0.018, Iz=0.014, W=0.028,
    sigma_y=300e3,
    yield_dir=2,
    RayleighAlpha=0.0, RayleighBeta=0.02,
)
plx_beam_ep_custom = BeamMaterialMapper.create_material(g_i, beam_ep_custom)


# -------- Pile Materials --------
pile_el_cyl = ElasticPile(
    name="Pile_EL_Cylinder",
    type=BeamType.Elastic,
    comment="Elastic pile - solid circular section",
    gamma=25.0,
    E=30e6, nu=0.25,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.Cylinder,
    diameter=1.20,
    lateral_type=LateralResistanceType.Linear,
    T_skin_start_max=120.0,
    T_skin_end_max=150.0,
    F_max=800.0,
    RayleighAlpha=0.0, RayleighBeta=0.015,
)
plx_pile_el_cyl = _make_pile(g_i, pile_el_cyl)

pile_el_square = ElasticPile(
    name="Pile_EL_Square",
    type=BeamType.Elastic,
    comment="Elastic pile - square section (b=0.8 m)",
    gamma=24.0,
    E=28e6, nu=0.23,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.Square,
    width=0.80,
    lateral_type=LateralResistanceType.MultiLinear,
    friction_curve=[
        (0.0,   0.0),
        (0.005, 80.0),
        (0.010, 150.0),
        (0.020, 200.0),
    ],
    F_max=1200.0,
)
plx_pile_el_square = _make_pile(g_i, pile_el_square)

pile_el_tube = ElasticPile(
    name="Pile_EL_Tube",
    type=BeamType.Elastic,
    comment="Elastic pile - circular tube (Do=1.5 m, t=0.08 m)",
    gamma=26.0,
    E=32e6, nu=0.22,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.CircularArcBeam,
    diameter=1.50,
    thickness=0.08,
    lateral_type=LateralResistanceType.LayerDependent,
    F_max=1500.0,
)
plx_pile_el_tube = _make_pile(g_i, pile_el_tube)

pile_el_custom = ElasticPile(
    name="Pile_EL_Custom",
    type=BeamType.Elastic,
    comment="Elastic pile - custom section",
    gamma=25.0,
    E=30e6, nu=0.20,
    cross_section=CrossSectionType.Custom,
    A=0.45, Iy=0.020, Iz=0.015, W=0.070,
    lateral_type=LateralResistanceType.Linear,
    T_skin_start_max=140.0,
    T_skin_end_max=180.0,
    F_max=900.0,
    RayleighAlpha=0.0, RayleighBeta=0.01,
)
plx_pile_el_custom = _make_pile(g_i, pile_el_custom)

pile_ep_square = ElastoplasticPile(
    name="Pile_EP_Square",
    type=BeamType.Elastoplastic,
    comment="Elasto-plastic pile - square with yield",
    gamma=25.0,
    E=31e6, nu=0.25,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.Square,
    width=1.00,
    sigma_y=360e3,
    yield_dir=1,
    lateral_type=LateralResistanceType.MultiLinear,
    friction_curve=[
        (0.0,   0.0),
        (0.010, 120.0),
        (0.020, 200.0),
        (0.030, 260.0),
    ],
    F_max=1400.0,
)
plx_pile_ep_square = _make_pile(g_i, pile_ep_square)

pile_ep_cyl = ElastoplasticPile(
    name="Pile_EP_Cylinder",
    type=BeamType.Elastoplastic,
    comment="Elasto-plastic pile - solid circular with yield",
    gamma=26.0,
    E=33e6, nu=0.26,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.Cylinder,
    diameter=1.00,
    sigma_y=320e3,
    yield_dir="local-2",
    lateral_type=LateralResistanceType.Linear,
    T_skin_start_max=120.0,
    T_skin_end_max=180.0,
    F_max=1100.0,
    RayleighAlpha=0.0, RayleighBeta=0.02,
)
plx_pile_ep_cyl = _make_pile(g_i, pile_ep_cyl)

pile_ep_tube = ElastoplasticPile(
    name="Pile_EP_Tube",
    type=BeamType.Elastoplastic,
    comment="Elasto-plastic pile - circular tube with yield",
    gamma=26.0,
    E=32e6, nu=0.24,
    cross_section=CrossSectionType.PreDefine,
    predefined_section=PreDefineSection.CircularArcBeam,
    diameter=1.40, thickness=0.06,
    sigma_y=355e3, yield_dir=2,
    lateral_type=LateralResistanceType.LayerDependent,
    F_max=1600.0,
)
plx_pile_ep_tube = _make_pile(g_i, pile_ep_tube)

pile_ep_custom = ElastoplasticPile(
    name="Pile_EP_Custom",
    type=BeamType.Elastoplastic,
    comment="Elasto-plastic pile - custom section with yield",
    gamma=24.0,
    E=31e6, nu=0.22,
    cross_section=CrossSectionType.Custom,
    A=0.50, Iy=0.026, Iz=0.021, W=0.085,
    sigma_y=340e3, yield_dir=1,
    lateral_type=LateralResistanceType.MultiLinear,
    friction_curve=[
        (0.0,   0.0),
        (0.008, 140.0),
        (0.016, 220.0),
        (0.025, 300.0),
    ],
    F_max=1800.0,
    RayleighAlpha=0.0, RayleighBeta=0.012,
)
plx_pile_ep_custom = _make_pile(g_i, pile_ep_custom)


# -------- Anchor Materials --------
anc_el = ElasticAnchor(
    name="Anchor_EL_1",
    type=AnchorType.Elastic,
    comment="Elastic anchor (EA=1500 kN)",
    EA=1500.0,
)
plx_anc_el = AnchorMaterialMapper.create_material(g_i, anc_el)
created_anchors.append((anc_el.name, plx_anc_el))


anc_el2 = ElasticAnchor(
    name="Anchor_EL_2",
    type=AnchorType.Elastic,
    comment="Elastic anchor (EA=5000 kN)",
    EA=5000.0,
)
plx_anc_el2 = AnchorMaterialMapper.create_material(g_i, anc_el2)
created_anchors.append((anc_el2.name, plx_anc_el2))


anc_ep = ElastoplasticAnchor(
    name="Anchor_EP_1",
    type=AnchorType.Elastoplastic,
    comment="Elastoplastic anchor with capacities",
    EA=3000.0,
    F_max_tens=1200.0,
    F_max_comp=800.0,
)
plx_anc_ep = AnchorMaterialMapper.create_material(g_i, anc_ep)
created_anchors.append((anc_ep.name, plx_anc_ep))


anc_ep_tension_only = ElastoplasticAnchor(
    name="Anchor_EP_TensionOnly",
    type=AnchorType.Elastoplastic,
    comment="Tension-only capacity; no compressive limit",
    EA=2500.0,
    F_max_tens=900.0,
    F_max_comp=None,
)
plx_anc_ep2 = AnchorMaterialMapper.create_material(g_i, anc_ep_tension_only)
created_anchors.append((anc_ep_tension_only.name, plx_anc_ep2))


anc_epr = ElastoPlasticResidualAnchor(
    name="Anchor_EP_Residual",
    type=AnchorType.ElastoPlasticResidual,
    comment="EP anchor with residual strengths",
    EA=4000.0,
    F_max_tens=1500.0,
    F_max_comp=1000.0,
    F_res_tens=300.0,
    F_res_comp=200.0,
)
plx_anc_epr = AnchorMaterialMapper.create_material(g_i, anc_epr)
created_anchors.append((anc_epr.name, plx_anc_epr))


# -----------------------------------------------------------------------------
# endregion
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# region Delete Materials
# -----------------------------------------------------------------------------

# -------- Soil Materials --------
print("Deleting soil materials...")
SoilMaterialMapper.delete_material(g_i, soil_mat_mc)   # last assigned MCC/HSS etc.
SoilMaterialMapper.delete_material(g_i, soil_mat_mcc)   # last assigned MCC/HSS etc.
SoilMaterialMapper.delete_material(g_i, soil_mat_hss)   # last assigned MCC/HSS etc.
# 如果你有多个 soil 对象 (mc, mcc, hss)，需要逐一 delete
# SoilMaterialMapper.delete_material(g_i, soil_mat_mc)
# SoilMaterialMapper.delete_material(g_i, soil_mat)

# -------- Plate Materials --------
print("Deleting plate materials...")
PlateMaterialMapper.delete_material(g_i, ep)
PlateMaterialMapper.delete_material(g_i, ep_aniso)
PlateMaterialMapper.delete_material(g_i, epp)

# -------- Beam Materials --------
print("Deleting beam materials...")
BeamMaterialMapper.delete_material(g_i, beam_el_cyl)
BeamMaterialMapper.delete_material(g_i, beam_el_rect)
BeamMaterialMapper.delete_material(g_i, beam_el_tube)
BeamMaterialMapper.delete_material(g_i, beam_el_custom)
BeamMaterialMapper.delete_material(g_i, beam_ep_rect)
BeamMaterialMapper.delete_material(g_i, beam_ep_cyl)
BeamMaterialMapper.delete_material(g_i, beam_ep_custom)

# -------- Pile Materials --------
print("Deleting pile materials...")
PileMaterialMapper.delete_material(g_i, pile_el_cyl)
PileMaterialMapper.delete_material(g_i, pile_el_square)
PileMaterialMapper.delete_material(g_i, pile_el_tube)
PileMaterialMapper.delete_material(g_i, pile_el_custom)
PileMaterialMapper.delete_material(g_i, pile_ep_square)
PileMaterialMapper.delete_material(g_i, pile_ep_cyl)
PileMaterialMapper.delete_material(g_i, pile_ep_tube)
PileMaterialMapper.delete_material(g_i, pile_ep_custom)

# -------- Anchor Materials --------
print("Deleting anchor materials...")
AnchorMaterialMapper.delete_material(g_i, anc_el)
AnchorMaterialMapper.delete_material(g_i, anc_el2)
AnchorMaterialMapper.delete_material(g_i, anc_ep)
AnchorMaterialMapper.delete_material(g_i, anc_ep_tension_only)
AnchorMaterialMapper.delete_material(g_i, anc_epr)

print("All materials deleted (Plaxis handles released, local objects preserved).")

# -----------------------------------------------------------------------------
# endregion
# -----------------------------------------------------------------------------
