from .basematerial import BaseMaterial
from enum import Enum, auto
from typing import Optional

class PlateType(Enum):
    Elastic = "Elastic"
    Elastoplastic = "Elastoplastic"


class ElasticPlate(BaseMaterial):
    """
    Reinforced concrete (RC) elastic plate material.
    Units:
      - gamma: kN/m^3
      - E, G*: kPa (use 30e6 for ~30 GPa)
      - nu: [-]
      - d: m
    Notes:
      - If `isotropic=True`, in-plane shear moduli (G12, G13, G23) will be
        auto-computed from E and nu unless explicitly provided.
    """

    def __init__(
        self,
        name: str,
        type: PlateType = PlateType.Elastic,
        comment: str = "",
        gamma: float = 25.0,      # RC unit weight (kN/m^3)
        E: float = 30e6,          # kPa (≈ 30 GPa)
        nu: float = 0.20,         # Poisson's ratio for concrete (0.18-0.22)
        d: float = 1.0,          # plate thickness (m)
        G12: Optional[float] = 12.50E6,  # kPa; auto = E/(2*(1+nu)) if isotropic
        preventpunch: bool = True,
        isotropic: bool = True,
        E2: Optional[float] = 30e6,   # kPa; used only if orthotropic
        G13: Optional[float] = 50E6,  # kPa
        G23: Optional[float] = 50E6   # kPa
    ) -> None:
        super().__init__(name, type, comment, gamma, E, nu)

        # Geometry / flags
        self._d = float(d)
        self._preventpunch = bool(preventpunch)
        self._isotropic = bool(isotropic)

        # Orthotropic secondary modulus (only meaningful if isotropic=False)
        self._E2 = float(E2) if (E2 is not None and not self._isotropic) else 0.0

        # Shear moduli handling
        if self._isotropic:
            # Auto-compute shear moduli if not provided
            G_iso = self._compute_isotropic_G(self.E, self.nu)
            self._G12 = float(G12) if G12 is not None else G_iso
            self._G13 = float(G13) if G13 is not None else G_iso
            self._G23 = float(G23) if G23 is not None else G_iso
        else:
            # Orthotropic: take provided values (default 0.0 if missing)
            self._G12 = float(G12) if G12 is not None else 0.0
            self._G13 = float(G13) if G13 is not None else 0.0
            self._G23 = float(G23) if G23 is not None else 0.0

    # ############################# helpers #############################

    @staticmethod
    def _compute_isotropic_G(E: float, nu: float) -> float:
        """Return isotropic shear modulus: G = E / [2(1+nu)] (kPa)."""
        return float(E) / (2.0 * (1.0 + float(nu)))

    def bending_rigidity(self) -> float:
        """
        Return bending rigidity D = E d^3 / [12 (1 - nu^2)] in kPa·m^3.
        This is per unit width, consistent with Kirchhoff-Love theory.
        """
        return (self.E * self._d ** 3) / (12.0 * (1.0 - self.nu ** 2))

    def self_weight_load(self) -> float:
        """
        Return plate self-weight as a surface load (kN/m^2): q = gamma * d.
        """
        return self.gamma * self._d

    def __repr__(self) -> str:
        return "<plx.materials.elastic_plate>"

    # ########################### properties ############################

    @property
    def d(self) -> float:
        """Plate thickness (m)."""
        return self._d

    @property
    def preventpunch(self) -> bool:
        """Whether punching shear prevention is enabled (True/False)."""
        return self._preventpunch

    @property
    def isotropic(self) -> bool:
        """Whether the plate is modeled as isotropic (True/False)."""
        return self._isotropic

    @property
    def E2(self) -> float:
        """Elastic modulus in the secondary in-plane direction (kPa, orthotropic only)."""
        return self._E2

    @property
    def G12(self) -> float:
        """In-plane shear modulus in the 1-2 direction (kPa)."""
        return self._G12

    @property
    def G13(self) -> float:
        """Transverse shear modulus in the 1-3 direction (kPa)."""
        return self._G13

    @property
    def G23(self) -> float:
        """Transverse shear modulus in the 2-3 direction (kPa)."""
        return self._G23


class ElastoplasticPlate(ElasticPlate):
    """
    Reinforced concrete elastoplastic plate material.
    Adds yield stress and section modulus per principal direction for simple
    plasticity checks in plate elements.

    - sigma_y_11, sigma_y_22: yield stress in kPa (e.g., 40e3 ~ 40 MPa)
    - W_11, W_22: section modulus in m^3 (per unit width if consistent)
    """

    def __init__(
        self,
        name: str,
        type: PlateType = PlateType.Elastoplastic,
        comment: str = "",
        gamma: float = 25.0,         # kN/m^3
        E: float = 30e6,             # kPa
        nu: float = 0.20,
        d: float = 1.0,             # m
        sigma_y_11: float = 40e3,    # kPa (~40 MPa)
        W_11: float = 0.05,           # m^3
        preventpunch: bool = True,
        isotropic: bool = True,
        G12: Optional[float] = 50E6, # kPa
        E2: Optional[float] = 30e6,  # kPa (orthotropic only)
        G13: Optional[float] = 50E6, # kPa
        G23: Optional[float] = 50E6, # kPa
        sigma_y_22: float = 40e3,    # kPa
        W_22: float = 0.04            # m^3
    ) -> None:

        # Correct super() call with proper parameter order and defaults
        super().__init__(
            name=name,
            type=type,
            comment=comment,
            gamma=gamma,
            E=E,
            nu=nu,
            d=d,
            G12=G12,
            preventpunch=preventpunch,
            isotropic=isotropic,
            E2=E2,
            G13=G13,
            G23=G23
        )

        self._sigma_y_11 = float(sigma_y_11)
        self._W_11 = float(W_11)
        self._sigma_y_22 = float(sigma_y_22)
        self._W_22 = float(W_22)

    def __repr__(self) -> str:
        return "<plx.materials.elastoplastic_plate>"

    # ########################### properties ############################

    @property
    def sigma_y_11(self) -> float:
        """Yield strength in the 1-1 principal direction (kPa)."""
        return self._sigma_y_11

    @property
    def W_11(self) -> float:
        """Section modulus in the 1-1 principal direction (m^3)."""
        return self._W_11

    @property
    def sigma_y_22(self) -> float:
        """Yield strength in the 2-2 principal direction (kPa)."""
        return self._sigma_y_22

    @property
    def W_22(self) -> float:
        """Section modulus in the 2-2 principal direction (m^3)."""
        return self._W_22
