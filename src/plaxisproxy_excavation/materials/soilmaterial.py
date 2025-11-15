from __future__ import annotations

from enum import Enum, auto
from typing import Any, Dict, Optional

from .basematerial import BaseMaterial


# ---------------------------------------------------------------------------
# Enums (strings should match mapper.model_name_map / PLAXIS names)
# ---------------------------------------------------------------------------

class SoilMaterialsType(Enum):
    MC  = "Mohr-Coulomb"       # Mohr-Coulomb
    MCC = "Modified Cam-clay"  # exact spelling used by the mapper
    HSS = "HS small"           # Hardening Soil small-strain (HS small)


class MCGWType(Enum):
    Standard = "Standard"
    Hypres   = "Hypres"
    USDA     = "USDA"
    Staring  = "Staring"


class MCGwSWCC(Enum):
    Van        = "Van Genuchten"
    Approx_Van = "Approx. Van Genuchten"


class RayleighInputMethod(Enum):
    Direct         = "Direct"
    SDOFEquivalent = "SDOFEquivalent"


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
    - E                  -> model-specific meaning (e.g., E50ref for HSS)
    - nu                 -> nu
    - e_init             -> eInit (initial void ratio)
    - n_init             -> derived as e/(1+e) if e provided

    MCC-specific interface inputs (v_ur, c_inter, phi_in, psi_inter, _R_inter)
    are NOT stored here; they live in MCCMaterial only.
    """

    def __init__(
        self,
        name: str,
        type: SoilMaterialsType,  # noqa: A003 (intentional: mapper expects .type)
        comment: str,
        gamma: float,
        E: float,
        nu: float,
        gamma_sat: float,
        e_init: float,
        **_: Any,
    ) -> None:
        # BaseMaterial is assumed to store name, type, comment, gamma, E, nu
        super().__init__(name, type, comment, gamma, E, nu)

        # Primary shared fields consumed by the mapper
        self._gamma_sat: float = float(gamma_sat)
        self._e_init: float = float(e_init)
        self._n_init: Optional[float] = (
            self._e_init / (1.0 + self._e_init) if self._e_init is not None else None
        )

        # Optional feature flags handled by the mapper (initialize to safe defaults)
        self._set_pore_stress: bool = False
        self._set_ug_water: bool = False
        self._set_additional_interface: bool = False
        self._set_additional_initial: bool = False

        # Rayleigh damping (mapper supports Direct method pass-through)
        self._input_method: RayleighInputMethod = RayleighInputMethod.Direct
        self._alpha: float = 0.0
        self._beta: float = 0.0

        # Lazy-initialized option members (silences Pylance unknown attribute)
        self._pore_stress_type: Optional[int] = None
        self._vu: Optional[int] = None
        self._water_value: Optional[float] = None

        self._Gw_type: Optional[MCGWType] = None
        self._SWCC_method: Optional[MCGwSWCC] = None
        self._soil_posi: Any = None
        self._soil_fine: Any = None
        self._Gw_defaults: Any = None
        self._infiltration: Any = None
        self._default_method: Any = None
        self._kx: Optional[float] = None
        self._ky: Optional[float] = None
        self._kz: Optional[float] = None
        self._Gw_Psiunsat: Optional[float] = None

        self._stiffness_define: Any = None
        self._strengthen_define: Any = None
        self._k_n: Optional[float] = None
        self._k_s: Optional[float] = None
        self._R_inter: float = 0.67
        self._gap_closure: Any = None
        self._cross_permeability: Any = None
        self._drainage_conduct1: Any = None
        self._drainage_conduct2: Any = None

        self._K_0_define: Any = None
        self._K_0_x: Optional[float] = None
        self._K_0_y: Optional[float] = None

    # ---------------- Optional groups consumed by the mapper -----------------

    def set_super_pore_stress_parameters(
        self, pore_stress_type: int = 0, vu: int = 0, value: Optional[float] = None
    ) -> None:
        """
        Configure super pore-stress options (forwarded by the mapper).

        pore_stress_type = 0 -> Non-drainage:
            vu = 0 -> 'Directly': value -> v_u
            vu = 1 -> 'Skempton B': value -> Skempton B

        pore_stress_type = 1 -> Biot Effective Stress:
            value -> sigma_Biot (naming follows your UI/DB convention)
        """
        self._set_pore_stress = True
        self._pore_stress_type = pore_stress_type
        self._vu = vu
        self._water_value = value

    def set_under_ground_water(
        self,
        type: MCGWType,
        SWCC_method: MCGwSWCC,
        soil_posi: Any,
        soil_fine: Any,
        Gw_defaults: Any,
        infiltration: Any,
        default_method: Any,
        kx: float,
        ky: float,
        kz: float,
        Gw_Psiunsat: float,
    ) -> None:
        """Configure underground water & SWCC options (forwarded by the mapper)."""
        self._set_ug_water = True
        self._Gw_type = type
        self._SWCC_method = SWCC_method
        self._soil_posi = soil_posi
        self._soil_fine = soil_fine
        self._Gw_defaults = Gw_defaults
        self._infiltration = infiltration
        self._default_method = default_method
        self._kx, self._ky, self._kz = float(kx), float(ky), float(kz)
        self._Gw_Psiunsat = float(Gw_Psiunsat)

    def set_additional_interface_parameters(
        self,
        stiffness_define: Any,
        strengthen_define: Any,
        k_n: float,
        k_s: float,
        R_inter: float,
        gap_closure: Any,
        cross_permeability: Any,
        drainage_conduct1: Any,
        drainage_conduct2: Any,
    ) -> None:
        """
        Configure additional interface parameters (forwarded by the mapper).
        The mapper can still override Rinter based on model-specific inputs
        (for MCC: c_inter/phi_in or explicit _R_inter).
        """
        self._set_additional_interface = True
        self._stiffness_define = stiffness_define
        self._strengthen_define = strengthen_define
        self._k_n, self._k_s = float(k_n), float(k_s)
        self._R_inter = float(R_inter)
        self._gap_closure = gap_closure
        self._cross_permeability = cross_permeability
        self._drainage_conduct1 = drainage_conduct1
        self._drainage_conduct2 = drainage_conduct2

    def set_additional_initial_parameters(
        self, K_0_define: Any, K_0_x: float = 0.5, K_0_y: float = 0.5
    ) -> None:
        """Configure additional K0 parameters (forwarded by the mapper)."""
        self._set_additional_initial = True
        self._K_0_define = K_0_define
        self._K_0_x = float(K_0_x)
        self._K_0_y = float(K_0_y)

    # -------------------------- Representation & props -----------------------

    def __repr__(self) -> str:  # pragma: no cover - friendly string
        return f"<plx.materials.soilbase name={self.name!r} model={self.type.value!r}>"

    @property
    def gamma_sat(self) -> float: return self._gamma_sat

    @property
    def e_init(self) -> float: return self._e_init

    @property
    def n_init(self) -> Optional[float]: return self._n_init

    @property
    def pore_stress_type(self) -> Optional[int]: return self._pore_stress_type

    @property
    def vu(self) -> Optional[int]: return self._vu

    @property
    def water_value(self) -> Optional[float]: return self._water_value

    @property
    def set_ug_water(self) -> bool: return self._set_ug_water

    @property
    def Gw_type(self) -> Optional[MCGWType]: return self._Gw_type

    @property
    def SWCC_method(self) -> Optional[MCGwSWCC]: return self._SWCC_method

    @property
    def soil_posi(self) -> Any: return self._soil_posi

    @property
    def soil_fine(self) -> Any: return self._soil_fine

    @property
    def Gw_defaults(self) -> Any: return self._Gw_defaults

    @property
    def infiltration(self) -> Any: return self._infiltration

    @property
    def default_method(self) -> Any: return self._default_method

    @property
    def kx(self) -> Optional[float]: return self._kx

    @property
    def ky(self) -> Optional[float]: return self._ky

    @property
    def kz(self) -> Optional[float]: return self._kz

    @property
    def Gw_Psiunsat(self) -> Optional[float]: return self._Gw_Psiunsat

    @property
    def stiffness_define(self) -> Any: return self._stiffness_define

    @property
    def strengthen_define(self) -> Any: return self._strengthen_define

    @property
    def k_n(self) -> Optional[float]: return self._k_n

    @property
    def k_s(self) -> Optional[float]: return self._k_s

    @property
    def R_inter(self) -> Optional[float]: return self._R_inter

    @property
    def gap_closure(self) -> Any: return self._gap_closure

    @property
    def cross_permeability(self) -> Any: return self._cross_permeability

    @property
    def drainage_conduct1(self) -> Any: return self._drainage_conduct1

    @property
    def drainage_conduct2(self) -> Any: return self._drainage_conduct2

    @property
    def K_0_define(self) -> Any: return self._K_0_define

    @property
    def K_0_x(self) -> Optional[float]: return self._K_0_x

    @property
    def K_0_y(self) -> Optional[float]: return self._K_0_y

# Numerical guard for tiny denominators
_EPS_DENOM: float = 1e-12

def _clamp01(x: float) -> float:
    """Clamp a float to [0, 1]."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

class MCMaterial(BaseSoilMaterial):
    """
    Mohr-Coulomb soil material.
    The mapper typically reads: E_ref, c_ref, phi, psi, tensile_strength.
    """

    def __init__(
        self,
        name: str = "MediumDenseSand",
        type: SoilMaterialsType = SoilMaterialsType.MC,  # noqa: A003
        comment: str = "",
        gamma: float = 18.0,
        E: float = 4.0e4,
        nu: float = 0.30,
        gamma_sat: float = 20.0,
        e_init: float = 0.65,
        E_ref: float = 4.0e4,
        c_ref: float = 1.0,
        phi: float = 32.0,
        psi: float = 2.0,
        tensile_strength: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init, **kwargs)
        self._E_ref: float = float(E_ref)
        self._c_ref: float = float(c_ref)
        self._phi: float = float(phi)
        self._psi: float = float(psi)
        self._tensile_strength: float = float(tensile_strength)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<plx.materials.MC name={self.name!r}>"

    @property
    def E_ref(self) -> float: return self._E_ref

    @property
    def c_ref(self) -> float: return self._c_ref

    @property
    def phi(self) -> float: return self._phi

    @property
    def psi(self) -> float: return self._psi

    @property
    def tensile_strength(self) -> float: return self._tensile_strength


class MCCMaterial(BaseSoilMaterial):
    """
    Modified Cam-Clay soil material.

    Mapper reads core parameters:
    - lam -> lambda
    - kar -> kappa
    - M_CSL -> M

    MCC-specific interface inputs stored ONLY here:
    - v_ur: unloading-reloading Poisson's ratio (dimensionless)
    - c_inter: interface cohesion [kPa]
    - phi_in: interface friction angle [deg]
    - psi_inter: interface dilatancy angle [deg]
    - _R_inter: explicit Rinter override (dimensionless)

    If _R_inter is not provided, a conservative ratio is inferred from (phi_in, psi_inter).
    """

    def __init__(
        self,
        name: str,
        type: SoilMaterialsType,  # noqa: A003
        comment: str,
        gamma: float,
        E: float,
        nu: float,
        gamma_sat: float,
        e_init: float,
        lam: float = 0.2,
        kar: float = 0.03,
        M_CSL: float = 1.2,
        v_ur: float = 0.15,
        c_inter: float = 30e3,
        phi_in: float = 30.0,
        psi_inter: float = 15.0,
        _R_inter: float = 0.67,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init, **kwargs)

        # Core MCC parameters
        self._lam: float = float(lam)
        self._kar: float = float(kar)
        self._M_CSL: float = float(M_CSL)

        # MCC-specific interface parameters with defaults
        self.v_ur: float = float(v_ur)
        self.c_inter: float = float(c_inter)
        self.phi_in: float = float(phi_in)
        self.psi_inter: float = float(psi_inter)

        # R_inter (protect against division-by-zero and clamp to [0, 1.0])
        if _R_inter is not None:
            self._R_inter = float(_R_inter)
        else:
            self._R_inter = self._compute_R_inter(self.phi_in, self.psi_inter, fallback=1.0)

    @staticmethod
    def _compute_R_inter(phi_in_deg: float, psi_in_deg: float, *, fallback: float = 1.0) -> float:
        """
        Compute interface reduction factor R_inter = tan(psi) / tan(phi), clamped to [0, 1].

        Robustness:
          - If phi_in is ~0° (or tan(phi) is non-finite / too small), return `fallback`.
          - Result is clamped to [0, 1] to avoid unphysical values.
        """
        import math
        phi_rad =  math.radians(float(phi_in_deg))
        psi_rad = math.radians(float(psi_in_deg))
        denom = math.tan(phi_rad)
        numer = math.tan(psi_rad)

        if (not math.isfinite(denom)) or abs(denom) < _EPS_DENOM:
            return float(fallback)

        return _clamp01(numer / denom)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<plx.materials.MCC name={self.name!r}>"

    @property
    def lam(self) -> float:
        """Compression index (λ)."""
        return self._lam

    @property
    def kar(self) -> float:
        """Swelling/recompression index (κ)."""
        return self._kar

    @property
    def M_CSL(self) -> float:
        """Critical state line slope (M) in p-q space."""
        return self._M_CSL

    @property
    def R_inter(self) -> float:
        """Interface reduction factor (dimensionless)."""
        return self._R_inter


class HSSMaterial(BaseSoilMaterial):
    """
    HS small (Hardening Soil with small-strain stiffness).
    Mapper expects: E (-> E50ref), E_oed (-> Eoedref), E_ur (-> Eurref), m, P_ref,
                    G0, gamma_07, c, phi, psi.
    """

    def __init__(
        self,
        name: str = "SoftClay_HSS",
        type: SoilMaterialsType = SoilMaterialsType.HSS,  # noqa: A003
        comment: str = "",
        gamma: float = 16.0,
        E: float = 8.0e3,
        nu: float = 0.35,
        gamma_sat: float = 18.0,
        e_init: float = 0.9,
        E_oed: float = 5600.0,
        E_ur: float = 24000.0,
        m: float = 1.0,
        P_ref: float = 100.0,
        G0: float = 180000.0,
        gamma_07: float = 0.0015,
        c: float = 15.0,
        phi: float = 22.0,
        psi: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, type, comment, gamma, E, nu, gamma_sat, e_init, **kwargs)
        self._E_oed: float = float(E_oed)
        self._E_ur: float = float(E_ur)
        self._m: float = float(m)
        self._P_ref: float = float(P_ref)
        self._G0: float = float(G0)
        self._gamma_07: float = float(gamma_07)
        self._c: float = float(c)
        self._phi: float = float(phi)
        self._psi: float = float(psi)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<plx.materials.HSS name={self.name!r}>"

    @property
    def E_oed(self) -> float: return self._E_oed

    @property
    def E_ur(self) -> float: return self._E_ur

    @property
    def m(self) -> float: return self._m

    @property
    def P_ref(self) -> float: return self._P_ref

    @property
    def G0(self) -> float: return self._G0

    @property
    def gamma_07(self) -> float: return self._gamma_07

    @property
    def c(self) -> float: return self._c

    @property
    def phi(self) -> float: return self._phi

    @property
    def psi(self) -> float: return self._psi


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class SoilMaterialFactory:
    """Factory for soil materials with alias-aware, validated argument parsing."""

    # ------------------------- helpers (private) -------------------------

    _SENTINEL: object = object()

    @staticmethod
    def _coalesce(
        mapping: Dict[str, Any],
        *names: str,
        default: Any = _SENTINEL,
        allow_none: bool = False,
    ) -> Any:
        """Return the first present (and not-None unless allowed) value among aliases."""
        for n in names:
            if n in mapping:
                v = mapping[n]
                if allow_none or v is not None:
                    return v
        if default is not SoilMaterialFactory._SENTINEL:
            return default
        raise KeyError(f"Expected one of {names}")

    @staticmethod
    def _num(
        mapping: Dict[str, Any],
        *aliases: str,
        required: bool = False,
        gt: Optional[float] = None,
        ge: Optional[float] = None,
        lt: Optional[float] = None,
        le: Optional[float] = None,
        default: float = 0,
        name: Optional[str] = None,
    ) -> float:
        """
        Pick first available alias and validate as float with optional bounds.
        """
        try:
            val = SoilMaterialFactory._coalesce(mapping, *aliases, default=default)
        except KeyError:
            if required:
                n = name or (aliases[0] if aliases else "value")
                raise TypeError(f"Missing required argument: '{n}'")
            return default

        if val is None:
            if required:
                n = name or (aliases[0] if aliases else "value")
                raise TypeError(f"Missing required argument: '{n}'")
            return 0

        try:
            num = float(val)
        except (TypeError, ValueError):
            n = name or (aliases[0] if aliases else "value")
            raise TypeError(f"Parameter '{n}' must be numeric, got: {val!r}")

        if gt is not None and not (num > gt):
            n = name or (aliases[0] if aliases else "value")
            raise ValueError(f"Parameter '{n}' must be > {gt}, got {num}")
        if ge is not None and not (num >= ge):
            n = name or (aliases[0] if aliases else "value")
            raise ValueError(f"Parameter '{n}' must be >= {ge}, got {num}")
        if lt is not None and not (num < lt):
            n = name or (aliases[0] if aliases else "value")
            raise ValueError(f"Parameter '{n}' must be < {lt}, got {num}")
        if le is not None and not (num <= le):
            n = name or (aliases[0] if aliases else "value")
            raise ValueError(f"Parameter '{n}' must be <= {le}, got {num}")
        return num

    @staticmethod
    def _str(
        mapping: Dict[str, Any],
        *aliases: str,
        required: bool = False,
        default: str = "",
        name: Optional[str] = None,
    ) -> str:
        """Pick first available alias and cast to str."""
        try:
            val = SoilMaterialFactory._coalesce(mapping, *aliases, default=default)
        except KeyError:
            if required:
                n = name or (aliases[0] if aliases else "value")
                raise TypeError(f"Missing required argument: '{n}'")
            return default

        if val is None:
            if required:
                n = name or (aliases[0] if aliases else "value")
                raise TypeError(f"Missing required argument: '{n}'")
            return default
        return str(val)

    # --------------------------- main factory ---------------------------

    @staticmethod
    def create(material_type: SoilMaterialsType, **kwargs: Any) -> BaseSoilMaterial:
        """
        Create a soil material of the requested type.

        Safe behavior:
        - Accept multiple aliases (E/E_ref/YoungsModulus/E50, nu/poisson, etc.).
        - Validate numbers and ranges.
        - Supply sensible defaults for optional parameters.
        - Avoid passing unknown kwargs to constructors.
        """

        # Common fields across models
        name = SoilMaterialFactory._str(kwargs, "name", required=True)
        comment = SoilMaterialFactory._str(kwargs, "comment", default="")
        gamma = SoilMaterialFactory._num(
            kwargs, "gamma", "gamma_unsat", "unit_weight",
            required=False, gt=0.0, default=18.0, name="gamma"
        )
        gamma_sat = SoilMaterialFactory._num(
            kwargs, "gamma_sat", "unit_weight_sat",
            required=False, gt=0.0, default=20.0, name="gamma_sat"
        )
        nu = SoilMaterialFactory._num(
            kwargs, "nu", "poisson", "poisson_ratio",
            required=False, ge=0.0, lt=0.5, default=0.30, name="nu"
        )
        e_init = SoilMaterialFactory._num(
            kwargs, "e_init", "e0", "void_ratio_init",
            required=False, ge=0.0, default=0.60, name="e_init"
        )
        # Base Young's modulus (many models need it)
        E_base = SoilMaterialFactory._num(
            kwargs, "E", "E_ref", "YoungsModulus", "E50",
            required=True, gt=0.0, name="E"
        )

        if material_type == SoilMaterialsType.MC:
            # MC model
            c_ref = SoilMaterialFactory._num(
                kwargs, "c_ref", "c", "cohesion",
                required=False, ge=0.0, default=0.0, name="c_ref"
            )
            phi = SoilMaterialFactory._num(
                kwargs, "phi", "phi_deg",
                required=True, ge=0.0, name="phi"
            )
            psi = SoilMaterialFactory._num(
                kwargs, "psi", "dilatancy", "psi_deg",
                required=False, default=0.0, name="psi"
            )
            ft = SoilMaterialFactory._num(
                kwargs, "tensile_strength", "ft", "sigma_t", "sig_t",
                required=False, ge=0.0, default=0.0, name="tensile_strength"
            )

            mat = MCMaterial(
                name=name,
                type=material_type,  # noqa: A003
                comment=comment,
                gamma=gamma if gamma is not None else 18.0,
                E=E_base if E_base is not None else 1.0,
                nu=nu if nu is not None else 0.30,
                gamma_sat=gamma_sat if gamma_sat is not None else 20.0,
                e_init=e_init if e_init is not None else 0.60,
                c_ref=c_ref if c_ref is not None else 0.0,
                phi=phi if phi is not None else 30.0,
                psi=psi if psi is not None else 0.0,
                tensile_strength=ft if ft is not None else 0.0,
            )

            # Optional: if MCMaterial exposes E_ref, set it to the best-available value
            E_ref_val = SoilMaterialFactory._num(
                kwargs, "E_ref", "E", default=E_base, gt=0.0, name="E_ref"
            )
            if E_ref_val is not None and hasattr(mat, "E_ref"):
                try:
                    setattr(mat, "_E_ref", float(E_ref_val))
                except Exception:
                    pass
            return mat

        if material_type == SoilMaterialsType.MCC:
            # Modified Cam-Clay
            lam = SoilMaterialFactory._num(kwargs, "lam", "lambda", "λ",
                                           required=False, gt=0.0, default=0.10, name="lam")
            kar = SoilMaterialFactory._num(kwargs, "kar", "kappa", "κ",
                                           required=False, gt=0.0, default=0.02, name="kar")
            M_CSL = SoilMaterialFactory._num(kwargs, "M_CSL", "M", "M_csl",
                                             required=False, gt=0.0, default=1.20, name="M_CSL")

            v_ur = SoilMaterialFactory._num(kwargs, "v_ur", "nu_ur",
                                            required=False, ge=0.0, lt=0.5, default=(nu if nu is not None else 0.30),
                                            name="v_ur")
            c_inter = SoilMaterialFactory._num(kwargs, "c_inter", required=False, ge=0.0, default=30e3, name="c_inter")
            phi_in = SoilMaterialFactory._num(kwargs, "phi_in", required=False, ge=0.0, default=30.0, name="phi_in")
            psi_inter = SoilMaterialFactory._num(kwargs, "psi_inter", required=False, default=15.0, name="psi_inter")
            R_inter = SoilMaterialFactory._num(kwargs, "_R_inter", "R_inter", required=False, ge=0.0, default=0.67, name="_R_inter")

            return MCCMaterial(
                name=name,
                type=material_type,  # noqa: A003
                comment=comment,
                gamma=gamma if gamma is not None else 18.0,
                E=E_base if E_base is not None else 1.0,
                nu=nu if nu is not None else 0.30,
                gamma_sat=gamma_sat if gamma_sat is not None else 20.0,
                e_init=e_init if e_init is not None else 0.60,
                lam=lam if lam is not None else 0.10,
                kar=kar if kar is not None else 0.02,
                M_CSL=M_CSL if M_CSL is not None else 1.20,
                v_ur=v_ur if v_ur is not None else 0.30,
                c_inter=c_inter if c_inter is not None else 30e3,
                phi_in=phi_in if phi_in is not None else 30.0,
                psi_inter=psi_inter if psi_inter is not None else 15.0,
                _R_inter=R_inter,
            )

        if material_type == SoilMaterialsType.HSS:
            # Hardening Soil Small-strain
            E_oed = SoilMaterialFactory._num(kwargs, "E_oed", "Eoed",
                                             required=False, gt=0.0, default=E_base, name="E_oed")
            E_ur = SoilMaterialFactory._num(kwargs, "E_ur", "Eur",
                                            required=False, gt=0.0, default=(3.0 * (E_base if E_base else 1.0)), name="E_ur")
            m = SoilMaterialFactory._num(kwargs, "m", required=False, gt=0.0, default=0.50, name="m")
            P_ref = SoilMaterialFactory._num(kwargs, "P_ref", "Pref", "p_ref", "pRef",
                                             required=False, gt=0.0, default=100.0, name="P_ref")
            G0 = SoilMaterialFactory._num(kwargs, "G0", "G_0", "G0_ref",
                                          required=False, gt=0.0,
                                          default=(E_base / (2.0 * (1.0 + (nu if nu is not None else 0.30)))) if E_base else 1.0,
                                          name="G0")
            gamma_07 = SoilMaterialFactory._num(kwargs, "gamma_07", "gamma07",
                                                required=False, gt=0.0, default=1e-4, name="gamma_07")
            c = SoilMaterialFactory._num(kwargs, "c", "c_ref", "cohesion",
                                         required=False, ge=0.0, default=0.0, name="c")
            phi = SoilMaterialFactory._num(kwargs, "phi", "phi_deg",
                                           required=False, ge=0.0, default=30.0, name="phi")
            psi = SoilMaterialFactory._num(kwargs, "psi", "dilatancy", "psi_deg",
                                           required=False, default=0.0, name="psi")

            return HSSMaterial(
                name=name,
                type=material_type,  # noqa: A003
                comment=comment,
                gamma=gamma if gamma is not None else 16.0,
                E=E_base if E_base is not None else 1.0,
                nu=nu if nu is not None else 0.35,
                gamma_sat=gamma_sat if gamma_sat is not None else 18.0,
                e_init=e_init if e_init is not None else 0.90,
                E_oed=E_oed if E_oed is not None else (E_base if E_base is not None else 1.0),
                E_ur=E_ur if E_ur is not None else (3.0 * (E_base if E_base else 1.0)),
                m=m if m is not None else 0.50,
                P_ref=P_ref if P_ref is not None else 100.0,
                G0=G0 if G0 is not None else (E_ur / (2.0 * (1.0 + (nu if nu is not None else 0.30))) if E_ur else 1.0),
                gamma_07=gamma_07 if gamma_07 is not None else 1e-4,
                c=c if c is not None else 0.0,
                phi=phi if phi is not None else 30.0,
                psi=psi if psi is not None else 0.0,
            )

        raise ValueError(f"Unknown soil material type: {material_type}")
