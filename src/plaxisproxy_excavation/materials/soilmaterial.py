from .basematerial import BaseMaterial
from enum import Enum, auto

class SoilMaterialsType(Enum):
    MC = auto()  # Mohr Coulomb
    MCC = auto()  # Modified Cambridge Clay
    HSS = auto()  # Hardening Small Strain

class MCGWType(Enum):
    Standard = auto()
    Hypres = auto()
    USDA = auto()
    Staring = auto()

class MCGwSWCC(Enum):
    Van = auto()
    Approx_Van = auto()

class RayleighInputMethod(Enum):
    Direct = "Direct"
    SDOFEquivalent = "SDOFEquivalent"

class BaseSoilMaterial(BaseMaterial):
    def __init__(self, name, type, comment, gamma, E, nu, gamma_sat, e_init, **kwargs) -> None:
        super().__init__(name, type, comment, gamma, E, nu)
        """
        Args:
            name (str): The name of the material.
            material_type (SoilMaterialsType): The PLAXIS soil model type.
            gamma_unsat (float): Unsaturated unit weight [kN/m³].
            gamma_sat (float): Saturated unit weight [kN/m³].
            k_x (float): Permeability in the x-direction [m/day].
            k_y (float): Permeability in the y-direction [m/day].
        """
        self._gamma_sat = gamma_sat
        self._e_init = e_init
        self._n_init = e_init / (1 + e_init)

        self._set_pore_stress = False
        self._set_ug_water = False
        self._set_additional_interface = False
        self._set_additional_initial = False
        
        # Rayleigh damping
        self._input_method: RayleighInputMethod = RayleighInputMethod.Direct
        self._alpha: float = 0.0
        self._beta: float = 0.0

    # Set the super pore stress parameters
    def set_super_pore_stress_parameters(self, pore_stress_type=0, vu=0, value=None):
        """
            pore_stress_type (float): as the software. Defaults to 0.
            vu (float): as the software. Defaults to 0.
            value (float): as the software. Defaults to None.
            
            pore_stress_type = 0 -> non-drainage:
                vu = 0 -> Directly: value -> v_u
                vu = 1 -> Based on Skempton B: value -> Skempton B

            pore_stress_type = 1 -> Boit Effective Stress:
                value -> simga_Boit
        """
        self._set_pore_stress = True
        self._pore_stress_type = pore_stress_type
        self._vu = vu
        self._water_value = value

    # Set additional underground water parameters
    def set_under_ground_water(self, type: MCGWType, SWCC_method: MCGwSWCC, soil_posi, soil_fine, Gw_defaults, 
                               infiltration, default_method, kx, ky, kz, Gw_Psiunsat):
        self._set_ug_water = True
        self._Gw_type = type
        self._SWCC_method = SWCC_method
        self._soil_posi = soil_posi
        self._soil_fine = soil_fine
        self._Gw_defaults = Gw_defaults
        self._infiltration = infiltration
        self._default_method = default_method
        self._kx = kx
        self._ky = ky
        self._kz = kz
        self._Gw_Psiunsat = Gw_Psiunsat

    # Set additional interface parameters
    def set_additional_interface_parameters(self, stiffness_define, strengthen_define, k_n, k_s, R_inter, gap_closure,
                                            cross_permeability, drainage_conduct1, drainage_conduct2):
        self._set_additional_interface = True
        self._stiffness_define = stiffness_define
        self._strengthen_define = strengthen_define
        self._k_n = k_n
        self._k_s = k_s
        self._R_inter = R_inter
        self._gap_closure = gap_closure
        self._cross_permeability = cross_permeability
        self._drainage_conduct1 = drainage_conduct1
        self._drainage_conduct2 = drainage_conduct2

    # Set additional initial parameters
    def set_additional_initial_parameters(self, K_0_define, K_0_x=0.5, K_0_y=0.5):
        self._set_additional_initial = True
        self._K_0_define = K_0_define
        self._K_0_x = K_0_x
        self._K_0_y = K_0_y

    def __repr__(self) -> str:
        return "<plx.materials.soilbase>"
    
    # region property
    @property
    def gamma_sat(self):
        return self._gamma_sat

    @property
    def e_init(self):
        return self._e_init

    @property
    def n_init(self):
        return self._n_init

    @property
    def pore_stress_type(self):
        return self._pore_stress_type

    @property
    def vu(self):
        return self._vu

    @property
    def water_value(self):
        return self._water_value

    @property
    def set_ug_water(self):
        return self._set_ug_water

    @property
    def Gw_type(self):
        """Get the groundwater type."""
        return self._Gw_type

    @property
    def SWCC_method(self):
        """Get the soil water characteristic curve method."""
        return self._SWCC_method

    @property
    def soil_posi(self):
        """Get the soil position information."""
        return self._soil_posi

    @property
    def soil_fine(self):
        """Get the soil fine content."""
        return self._soil_fine

    @property
    def Gw_defaults(self):
        """Get the groundwater defaults."""
        return self._Gw_defaults

    @property
    def infiltration(self):
        """Get the infiltration parameters."""
        return self._infiltration

    @property
    def default_method(self):
        """Get the default method."""
        return self._default_method

    @property
    def kx(self):
        """Get the hydraulic conductivity in x-direction."""
        return self._kx

    @property
    def ky(self):
        """Get the hydraulic conductivity in y-direction."""
        return self._ky

    @property
    def kz(self):
        """Get the hydraulic conductivity in z-direction."""
        return self._kz

    @property
    def Gw_Psiunsat(self):
        """Get the unsaturated soil water potential."""
        return self._Gw_Psiunsat

    @property
    def stiffness_define(self):
        """Get the stiffness definition parameters."""
        return self._stiffness_define

    @property
    def strengthen_define(self):
        """Get the strength definition parameters."""
        return self._strengthen_define

    @property
    def k_n(self):
        """Get the normal stiffness coefficient."""
        return self._k_n

    @property
    def k_s(self):
        """Get the shear stiffness coefficient."""
        return self._k_s

    @property
    def R_inter(self):
        """Get the interface resistance parameter."""
        return self._R_inter

    @property
    def gap_closure(self):
        """Get the gap closure behavior definition."""
        return self._gap_closure

    @property
    def cross_permeability(self):
        """Get the cross-permeability coefficient."""
        return self._cross_permeability

    @property
    def drainage_conduct1(self):
        """Get the primary drainage conductivity."""
        return self._drainage_conduct1

    @property
    def drainage_conduct2(self):
        """Get the secondary drainage conductivity."""
        return self._drainage_conduct2

    @property
    def K_0_define(self):
        """Get the definition method for the at-rest earth pressure coefficient (K₀)."""
        return self._K_0_define

    @property
    def K_0_x(self):
        """Get the at-rest earth pressure coefficient in the x-direction (K₀ₓ)."""
        return self._K_0_x

    @property
    def K_0_y(self):
        """Get the at-rest earth pressure coefficient in the y-direction (K₀ᵧ)."""
        return self._K_0_y
    # endregion

class MCMaterial(BaseSoilMaterial):
    def __init__(self, name, type, comment, gamma, E, nu, gamma_sat, e_init,  # general
                 E_ref, c_ref, phi, psi, tensile_strength) -> None:  # Mechanics 
        """
        Args:
            name (str): as the software
            type (SoilMaterialsType): as the software
            comment (str): as the software
            gamma (float): as the software
            E (float): E_ref, Reference Young's modulus [kN/m²].
            gamma_sat (float): as the software
            e_init (float): as the software
            c_ref (float): Reference cohesion [kN/m²].
            phi (float): Friction angle [°].
            psi (float): Dilatancy angle [°].
            tensile_strength (float): as the software
        """
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init)
        
        self._E_ref = E_ref 
        self._c_ref = c_ref
        self._phi = phi
        self._psi = psi
        self._tensile_strength = tensile_strength

    # region property
    def __repr__(self) -> str:
        return f"<plx.materials.MCSoil>"
    
    @property
    def E_ref(self):
        return self._E_ref
    
    @property
    def c_ref(self):
        return self._c_ref
    
    @property
    def phi(self):
        return self._phi
    
    @property
    def psi(self):
        return self._psi
    
    @property
    def tensile_strength(self):
        return self._tensile_strength
    # endregion

class MCCMaterial(BaseSoilMaterial):
    def __init__(self, name, type, comment, gamma, E, nu, gamma_sat, e_init, lam, kar, M_CSL) -> None:
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init)
        """
        Args:
            kappa (float): Swelling index (dimensionless).
            lambda_val (float): Compression index (dimensionless).
            M (float): Critical state line slope (dimensionless).
        """
        self._lam = lam
        self._kar = kar
        self._M_CSL = M_CSL

    def __repr__(self) -> str:
        return f"<plx.materials.MCCSoil>"

    @property
    def lam(self):
        """Get the lambda parameter (λ) for MCC constitutive model."""
        return self._lam

    @property
    def kar(self):
        """Get the kappa parameter (κ) for soil swelling/recompression index."""
        return self._kar

    @property
    def M_CSL(self):
        """Get the critical state line slope (M) in p-q space."""
        return self._M_CSL

class HSSMaterial(BaseSoilMaterial):
    def __init__(self, name, type, comment, gamma, E, nu, gamma_sat, e_init, E_oed, E_ur, m, P_ref, 
                 G0, gamma_07, c, phi, psi) -> None:
        """
        Args:
            E (float): E50_ref, Reference secant stiffness in triaxial test [kN/m²].
            Eoed_ref (float): Reference tangent stiffness for oedometer loading [kN/m²].
            Eur_ref (float): Reference unloading/reloading stiffness [kN/m²].
            m (float): Power for stress-level dependency (dimensionless).
            p_ref (float): Reference pressure for stiffnesses [kN/m²].
            K0_nc (float): K0 for normal consolidation (dimensionless).
            Rf (float): Failure ratio (dimensionless).
        """
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init)

        self._E_oed = E_oed
        self._E_ur = E_ur
        self._m = m
        self._P_ref = P_ref
        self._G0 = G0
        self._gamma_07 = gamma_07
        self._c = c
        self._phi = phi
        self._psi = psi

    # region property
    def __repr__(self) -> str:
        return "<plx.materials.HSSSoil>"
    
    @property
    def E_oed(self):
        """Get the oedometric modulus (Eₒₑₓ) for constrained compression."""
        return self._E_oed

    @property
    def E_ur(self):
        """Get the unloading-reloading modulus (Eᵤᵣ) for elastic behavior."""
        return self._E_ur

    @property
    def m(self):
        """Get the exponent (m) for stress-dependent stiffness (e.g., power law)."""
        return self._m

    @property
    def P_ref(self):
        """Get the reference stress (P_ref) for stiffness normalization [kPa or MPa]."""
        return self._P_ref

    @property
    def G0(self):
        """Get the small-strain shear modulus (G₀) at very low strains."""
        return self._G0

    @property
    def gamma_07(self):
        """Get the shear strain (γ₀.₇) at which G = 0.7G₀ (nonlinear threshold)."""
        return self._gamma_07

    @property
    def c(self):
        """Get the cohesion (c) for Mohr-Coulomb strength criterion [kPa]."""
        return self._c

    @property
    def phi(self):
        """Get the friction angle (φ) for Mohr-Coulomb criterion [degrees]."""
        return self._phi

    @property
    def psi(self):
        """Get the dilation angle (ψ) for plastic flow rule [degrees]."""
        return self._psi
    # endregion

# =============================================================================
#  Factory Class for Soil Material Creation
# =============================================================================

class SoilMaterialFactory:
    """
    A factory class to simplify the creation of different soil material objects.

    This class decouples the client code from the concrete implementation of
    soil material classes. Based on the provided material type, it calls the
    appropriate constructor and passes the required parameters.
    """

    @staticmethod
    def create(material_type: SoilMaterialsType, **kwargs) -> BaseSoilMaterial:
        """
        Creates a soil material instance based on the specified type.

        Args:
            material_type (SoilMaterialsType): The enum member specifying which
                                               soil model to create (e.g., MC, HSS).
            **kwargs: A dictionary of keyword arguments required by the constructor
                      of the specific soil material class.

        Raises:
            ValueError: If an unknown material_type is provided.
            TypeError: If the required arguments for a specific model are missing
                       from kwargs.

        Returns:
            BaseSoilMaterial: An instance of the requested soil material class
                               (e.g., MCMaterial, HSSMaterial).
        """
        if material_type == SoilMaterialsType.MC:
            # For Mohr-Coulomb, we expect specific parameters
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
            # For Modified Cambridge Clay
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
                )
            except KeyError as e:
                raise TypeError(f"Missing required argument for MCCMaterial: {e}")

        elif material_type == SoilMaterialsType.HSS:
            # For Hardening Soil with Small-Strain Stiffness
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
