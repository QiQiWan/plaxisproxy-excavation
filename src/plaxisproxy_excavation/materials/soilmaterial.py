# soilmaterials.py
from .basematerial import BaseMaterial
from enum import Enum, auto
from typing import Optional

# ---------------------------------------------------------------------------
# Enums (strings must match mapper.model_name_map and PLAXIS enumerations)
# ---------------------------------------------------------------------------

class SoilMaterialsType(Enum):
    MC  = "Mohr-Coulomb"       # Mohr-Coulomb
    MCC = "Modified Cam-clay"  # NOTE: exact spelling, used by the mapper
    HSS = "HS small"           # Hardening Soil small-strain (HS small)

class MCGWType(Enum):
    Standard = auto()
    Hypres   = auto()
    USDA     = auto()
    Staring  = auto()

class MCGwSWCC(Enum):
    Van         = auto()
    Approx_Van  = auto()

class RayleighInputMethod(Enum):
    Direct          = "Direct"
    SDOFEquivalent  = "SDOFEquivalent"

# ---------------------------------------------------------------------------
# Base class and concrete soil material classes
# ---------------------------------------------------------------------------

class BaseSoilMaterial(BaseMaterial):
    """
    Base class for PLAXIS soil materials (fields shared by all soil models).

    Read by the mapper:
    - name               -> MaterialName / Identification
    - type               -> SoilModel (via SoilMaterialsType)
    - gamma (unsat)      -> gammaUnsat
    - gamma_sat          -> gammaSat
    - E                  -> (model-specific meaning, e.g., E50ref for HSS)
    - nu                 -> nu
    - e_init             -> eInit (initial void ratio)
    - n_init             -> derived as e/(1+e) if e is provided

    NOTE:
    MCC-specific interface inputs (v_ur, c_inter, phi_in, psi_inter, _R_inter)
    are NOT stored here anymore. They belong to MCCMaterial only.
    """

    def __init__(
        self,
        name,
        type,
        comment,
        gamma,
        E,
        nu,
        gamma_sat,
        e_init,
        **kwargs
    ) -> None:
        # BaseMaterial is assumed to store name, type, comment, gamma, E, nu
        super().__init__(name, type, comment, gamma, E, nu)

        # Primary shared fields consumed by the mapper
        self._gamma_sat = gamma_sat
        self._e_init    = e_init
        self._n_init    = e_init / (1.0 + e_init) if e_init is not None else None

        # Optional feature flags handled by the mapper
        self._set_pore_stress           = False
        self._set_ug_water              = False
        self._set_additional_interface  = False
        self._set_additional_initial    = False

        # Rayleigh damping (mapper supports Direct method pass-through)
        self._input_method: RayleighInputMethod = RayleighInputMethod.Direct
        self._alpha: float = 0.0
        self._beta:  float = 0.0

    # ---------------- Optional groups consumed by the mapper -----------------

    def set_super_pore_stress_parameters(self, pore_stress_type=0, vu=0, value=None):
        """
        Configure super pore-stress options (forwarded by the mapper).

        pore_stress_type = 0 -> Non-drainage:
            vu = 0 -> 'Directly': value -> v_u
            vu = 1 -> 'Skempton B': value -> Skempton B

        pore_stress_type = 1 -> Biot Effective Stress:
            value -> sigma_Biot (naming follows your UI/DB convention)
        """
        self._set_pore_stress   = True
        self._pore_stress_type  = pore_stress_type
        self._vu                = vu
        self._water_value       = value

    def set_under_ground_water(
        self,
        type: MCGWType,
        SWCC_method: MCGwSWCC,
        soil_posi, soil_fine, Gw_defaults,
        infiltration, default_method,
        kx, ky, kz, Gw_Psiunsat
    ):
        """Configure underground water & SWCC options (forwarded by the mapper)."""
        self._set_ug_water   = True
        self._Gw_type        = type
        self._SWCC_method    = SWCC_method
        self._soil_posi      = soil_posi
        self._soil_fine      = soil_fine
        self._Gw_defaults    = Gw_defaults
        self._infiltration   = infiltration
        self._default_method = default_method
        self._kx, self._ky, self._kz = kx, ky, kz
        self._Gw_Psiunsat    = Gw_Psiunsat

    def set_additional_interface_parameters(
        self,
        stiffness_define, strengthen_define,
        k_n, k_s, R_inter,
        gap_closure, cross_permeability,
        drainage_conduct1, drainage_conduct2
    ):
        """
        Configure additional interface parameters (forwarded by the mapper).
        The mapper can still override Rinter based on model-specific inputs
        (for MCC: c_inter/phi_in or explicit _R_inter).
        """
        self._set_additional_interface = True
        self._stiffness_define  = stiffness_define
        self._strengthen_define = strengthen_define
        self._k_n, self._k_s    = k_n, k_s
        self._R_inter           = R_inter
        self._gap_closure       = gap_closure
        self._cross_permeability = cross_permeability
        self._drainage_conduct1 = drainage_conduct1
        self._drainage_conduct2 = drainage_conduct2

    def set_additional_initial_parameters(self, K_0_define, K_0_x=0.5, K_0_y=0.5):
        """Configure additional K0 parameters (forwarded by the mapper)."""
        self._set_additional_initial = True
        self._K_0_define = K_0_define
        self._K_0_x      = K_0_x
        self._K_0_y      = K_0_y

    # -------------------------- Representation & props -----------------------

    def __repr__(self) -> str:
        return "<plx.materials.soilbase>"

    @property
    def gamma_sat(self): return self._gamma_sat

    @property
    def e_init(self): return self._e_init

    @property
    def n_init(self): return self._n_init

    @property
    def pore_stress_type(self): return getattr(self, "_pore_stress_type", None)

    @property
    def vu(self): return getattr(self, "_vu", None)

    @property
    def water_value(self): return getattr(self, "_water_value", None)

    @property
    def set_ug_water(self): return self._set_ug_water

    @property
    def Gw_type(self): return getattr(self, "_Gw_type", None)

    @property
    def SWCC_method(self): return getattr(self, "_SWCC_method", None)

    @property
    def soil_posi(self): return getattr(self, "_soil_posi", None)

    @property
    def soil_fine(self): return getattr(self, "_soil_fine", None)

    @property
    def Gw_defaults(self): return getattr(self, "_Gw_defaults", None)

    @property
    def infiltration(self): return getattr(self, "_infiltration", None)

    @property
    def default_method(self): return getattr(self, "_default_method", None)

    @property
    def kx(self): return getattr(self, "_kx", None)

    @property
    def ky(self): return getattr(self, "_ky", None)

    @property
    def kz(self): return getattr(self, "_kz", None)

    @property
    def Gw_Psiunsat(self): return getattr(self, "_Gw_Psiunsat", None)

    @property
    def stiffness_define(self): return getattr(self, "_stiffness_define", None)

    @property
    def strengthen_define(self): return getattr(self, "_strengthen_define", None)

    @property
    def k_n(self): return getattr(self, "_k_n", None)

    @property
    def k_s(self): return getattr(self, "_k_s", None)

    @property
    def R_inter(self): return getattr(self, "_R_inter", None)

    @property
    def gap_closure(self): return getattr(self, "_gap_closure", None)

    @property
    def cross_permeability(self): return getattr(self, "_cross_permeability", None)

    @property
    def drainage_conduct1(self): return getattr(self, "_drainage_conduct1", None)

    @property
    def drainage_conduct2(self): return getattr(self, "_drainage_conduct2", None)

    @property
    def K_0_define(self): return getattr(self, "_K_0_define", None)

    @property
    def K_0_x(self): return getattr(self, "_K_0_x", None)

    @property
    def K_0_y(self): return getattr(self, "_K_0_y", None)


class MCMaterial(BaseSoilMaterial):
    """
    Mohr-Coulomb soil material.
    The mapper expects fields: E_ref, c_ref, phi, psi, tensile_strength.
    """
    def __init__(self, 
                 name: str = "MediumDenseSand", type: SoilMaterialsType = SoilMaterialsType.MC, 
                 comment: str = "", gamma: float = 18.0, E: float = 4e4, nu: float = 0.3, 
                 gamma_sat: float = 20.0, e_init: float = 0.65, E_ref: float = 4e4, 
                 c_ref: float = 1.0, phi: float = 32.0, psi: float = 2.0, 
                 tensile_strength: float = 1.0, **kwargs) -> None:
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init, **kwargs)
        self._E_ref            = E_ref
        self._c_ref            = c_ref
        self._phi              = phi
        self._psi              = psi
        self._tensile_strength = tensile_strength

    def __repr__(self) -> str:
        return "<plx.materials.MCSoil>"

    @property
    def E_ref(self): return self._E_ref

    @property
    def c_ref(self): return self._c_ref

    @property
    def phi(self): return self._phi

    @property
    def psi(self): return self._psi

    @property
    def tensile_strength(self): return self._tensile_strength

class MCCMaterial(BaseSoilMaterial):
    """
    Modified Cam-Clay soil material.

    Mapper will read core parameters:
    - lam -> lambda
    - kar -> kappa
    - M_CSL -> M

    MCC-specific interface inputs stored ONLY here:
    - v_ur: unloading-reloading Poisson's ratio (alias for nu when nu is None)
            (dimensionless), default = None
    - c_inter: interface cohesion [kPa], default = None
    - phi_in: interface friction angle [deg], default = None
    - psi_inter: interface dilatancy angle [deg], default = None
    - _R_inter: explicit Rinter override (dimensionless), default = None

    The mapper converts (c_inter, phi_in) to InterfaceStrength="Manual" and
    Rinter (using a conservative ratio rule), or uses _R_inter when provided.
    """
    def __init__(
        self,
        name,
        type,
        comment,
        gamma,
        E,
        nu,
        gamma_sat,
        e_init,
        lam: float = 0.2,
        kar: float = 0.03,
        M_CSL: float = 1.2,
        v_ur: float = 0.15,
        c_inter: float = 30e3,
        phi_in: float = 30.0,
        psi_inter: float = 15.0,
        _R_inter: Optional[float] = None,
        **kwargs
    ) -> None:
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init, **kwargs)

        # Core MCC parameters
        self._lam   = lam
        self._kar   = kar
        self._M_CSL = M_CSL

        # MCC-specific interface parameters with defaults
        self.v_ur: float      = v_ur
        self.c_inter: float   = c_inter
        self.phi_in: float    = phi_in
        self.psi_inter: float = psi_inter
        if _R_inter:
            self._R_inter: float  = _R_inter
        else:
            from math import tan, pi
            _psi_inter = psi_inter / 180 * pi
            _phi_in = phi_in / 180 * pi
            self._R_inter = tan(_psi_inter) / tan(_phi_in)

    def __repr__(self) -> str:
        return "<plx.materials.MCCSoil>"

    # region properties
    @property
    def lam(self):
        """Compression index (λ)."""
        return self._lam

    @property
    def kar(self):
        """Swelling/recompression index (κ)."""
        return self._kar

    @property
    def M_CSL(self):
        """Critical state line slope (M) in p-q space."""
        return self._M_CSL

    @property
    def R_inter(self):
        """Explicit Rinter override (dimensionless)."""
        return self._R_inter
    # endregion


class HSSMaterial(BaseSoilMaterial):
    """
    HS small (Hardening Soil with small-strain stiffness).
    Mapper expects: E (-> E50ref), E_oed (-> Eoedref), E_ur (-> Eurref), m, P_ref,
                    G0, gamma_07, c, phi, psi.
    """
    def __init__(
        self, name: str = "SoftClay_HSS", type: SoilMaterialsType = SoilMaterialsType.HSS, 
        comment: str = "", gamma: float = 16.0, E: float = 8e3, nu: float = 0.35, gamma_sat: float = 18.0, 
        e_init: float = 0.9, E_oed: float = 5600.0, E_ur: float = 24000.0, m: float = 1.0, 
        P_ref: float = 100.0, G0: float = 180000.0, gamma_07: float = 0.0015, c: float = 15.0, 
        phi: float = 22.0, psi: float = 0.0, **kwargs
    ) -> None:
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init, **kwargs)
        self._E_oed    = E_oed
        self._E_ur     = E_ur
        self._m        = m
        self._P_ref    = P_ref
        self._G0       = G0
        self._gamma_07 = gamma_07
        self._c        = c
        self._phi      = phi
        self._psi      = psi

    def __repr__(self) -> str:
        return "<plx.materials.HSSSoil>"

    @property
    def E_oed(self): return self._E_oed

    @property
    def E_ur(self): return self._E_ur

    @property
    def m(self): return self._m

    @property
    def P_ref(self): return self._P_ref

    @property
    def G0(self): return self._G0

    @property
    def gamma_07(self): return self._gamma_07

    @property
    def c(self): return self._c

    @property
    def phi(self): return self._phi

    @property
    def psi(self): return self._psi


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class SoilMaterialFactory:
    """
    Factory for constructing specific soil material instances.

    NOTE:
    - For MCC, interface inputs (`v_ur`, `c_inter`, `phi_in`, `psi_inter`, `_R_inter`)
      are accepted here and stored in MCCMaterial only (not in Base).
    """

    @staticmethod
    def create(material_type: SoilMaterialsType, **kwargs) -> BaseSoilMaterial:
        if material_type == SoilMaterialsType.MC:
            try:
                return MCMaterial(
                    name=kwargs["name"],
                    type=material_type,
                    comment=kwargs.get("comment", ""),
                    gamma=kwargs["gamma"],
                    E=kwargs["E"],
                    nu=kwargs["nu"],
                    gamma_sat=kwargs["gamma_sat"],
                    e_init=kwargs["e_init"],
                    E_ref=kwargs["E_ref"],
                    c_ref=kwargs["c_ref"],
                    phi=kwargs["phi"],
                    psi=kwargs["psi"],
                    tensile_strength=kwargs["tensile_strength"],
                )
            except KeyError as e:
                raise TypeError(f"Missing required argument for MCMaterial: {e}")

        elif material_type == SoilMaterialsType.MCC:
            try:
                return MCCMaterial(
                    name=kwargs["name"],
                    type=material_type,
                    comment=kwargs.get("comment", ""),
                    gamma=kwargs["gamma"],
                    E=kwargs["E"],
                    nu=kwargs["nu"],
                    gamma_sat=kwargs["gamma_sat"],
                    e_init=kwargs["e_init"],
                    lam=kwargs["lam"],
                    kar=kwargs["kar"],
                    M_CSL=kwargs["M_CSL"],
                    # MCC-only extras with defaults (all optional)
                    v_ur=kwargs.get("v_ur", kwargs["nu"]),
                    c_inter=kwargs.get("c_inter", 30e3),
                    phi_in=kwargs.get("phi_in", 30.0),
                    psi_inter=kwargs.get("psi_inter", 15.0),
                    _R_inter=kwargs.get("_R_inter", 0.46),
                )
            except KeyError as e:
                raise TypeError(f"Missing required argument for MCCMaterial: {e}")

        elif material_type == SoilMaterialsType.HSS:
            try:
                return HSSMaterial(
                    name=kwargs["name"],
                    type=material_type,
                    comment=kwargs.get("comment", ""),
                    gamma=kwargs["gamma"],
                    E=kwargs["E"],
                    nu=kwargs["nu"],
                    gamma_sat=kwargs["gamma_sat"],
                    e_init=kwargs["e_init"],
                    E_oed=kwargs["E_oed"],
                    E_ur=kwargs["E_ur"],
                    m=kwargs["m"],
                    P_ref=kwargs["P_ref"],
                    G0=kwargs["G0"],
                    gamma_07=kwargs["gamma_07"],
                    c=kwargs["c"],
                    phi=kwargs["phi"],
                    psi=kwargs["psi"],
                )
            except KeyError as e:
                raise TypeError(f"Missing required argument for HSSMaterial: {e}")

        else:
            raise ValueError(f"Unknown soil material type: {material_type}")
