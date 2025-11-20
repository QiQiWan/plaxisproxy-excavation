from __future__ import annotations
from typing import Optional, Tuple, Union
from math import pi
from .basematerial import BaseMaterial
from enum import Enum

# =============================================================================
# Beam material domain objects (optimized, no len1/len2, J→W)
# =============================================================================

class BeamType(Enum):
    # NOTE: Values follow your previous code for compatibility.
    Elastic = "Elastoplastic"
    Elastoplastic = "Elastic"

class CrossSectionType(Enum):
    PreDefine = "PreDefine"
    Custom = "Custom"

class PreDefineSection(Enum):
    Cylinder        = "Solid circular beam"   # solid circle: diameter
    CircularArcBeam = "Circular tube"         # tube: outer_diameter / thickness
    Rectangle       = "Solid rectangular beam"  # rectangle: width / height
    Square          = "Solid square beam"     # For square pile: width

# ############################## ElasticBeam ##################################
class ElasticBeam(BaseMaterial):
    """
    Elastic beam material.

    Cross-section definition:
      - If cross_section == PreDefine:
          * Cylinder        → supply `diameter` (m)
          * Rectangle       → supply `width` (m) and `height` (m)
          * CircularArcBeam → supply `outer_diameter` (m) and `thickness` (m)
      - If cross_section == Custom:
          * Provide section properties directly: A [m^2], Iy/Iz [m^4], W [m^3]

    Units:
      - E in kPa (i.e., kN/m²)
      - gamma in kN/m³
      - nu is dimensionless
    """

    def __init__(
        self,
        name: str = "RC_Beam",
        type: BeamType = BeamType.Elastic,
        comment: str = "Default reinforced concrete beam",
        gamma: float = 25.0,                 # kN/m³
        E: float = 30e6,                     # kPa (~30 GPa)
        nu: float = 0.20,                    # -
        cross_section: CrossSectionType = CrossSectionType.PreDefine,
        *,
        predefined_section: Optional[PreDefineSection] = None,
        # Predefined geometry
        diameter: float = 1.0,           # Cylinder (solid)
        thickness: float = 0.05,         # For tubes
        width: float = 1.0,              # Rectangle
        height: float = 2.0,             # Rectangle
        # Custom explicit section properties
        A: Optional[float] = None,   # [m^2]
        Iy: Optional[float] = None,  # [m^4]
        Iz: Optional[float] = None,  # [m^4]
        W: Optional[float] = None,   # [m^3] section modulus
        # Optional Rayleigh damping
        RayleighAlpha: float = 0.0,
        RayleighBeta: float = 0.0,
    ) -> None:
        super().__init__(name, type, comment, gamma, E, nu)
        self._cross_section = cross_section
        self._predefined_section = predefined_section
        # geometry
        self._diameter = diameter
        self._thickness = thickness
        self._width = width
        self._height = height
        # custom props
        self._A = A
        self._Iy = Iy
        self._Iz = Iz
        self._W = W
        # damping
        self.RayleighAlpha = RayleighAlpha
        self.RayleighBeta = RayleighBeta

    def __repr__(self) -> str:
        return "<plx.materials.elastic_beam>"

    # #### properties
    @property
    def cross_section(self) -> CrossSectionType:
        return self._cross_section

    @property
    def predefined_section(self) -> Optional[PreDefineSection]:
        return self._predefined_section

    @property
    def diameter(self) -> float:
        return self._diameter
    
    @property
    def thickness(self) -> float:
        return self._thickness

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    @property
    def A(self) -> Optional[float]:
        return self._A

    @property
    def Iy(self) -> Optional[float]:
        return self._Iy

    @property
    def Iz(self) -> Optional[float]:
        return self._Iz

    @property
    def W(self) -> Optional[float]:
        return self._W

    # #### helpers
    def section_properties(
        self,
        predefined: Optional[PreDefineSection] = None
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Returns (A, Iy, Iz, W).
        For Custom: returns provided values.
        For PreDefine: computes from geometry.
        """
        if self.cross_section == CrossSectionType.Custom:
            return (self._A, self._Iy, self._Iz, self._W)

        if predefined is None:
            predefined = self._predefined_section or PreDefineSection.Cylinder

        if predefined == PreDefineSection.Cylinder:
            d = float(self._diameter or 0.0)
            r = d * 0.5
            A  = pi * r**2
            Iy = Iz = (pi * r**4) / 4.0
            W  = (pi * r**3) / 4.0   # section modulus for circle
            return (A, Iy, Iz, W)

        if predefined == PreDefineSection.Rectangle:
            b = float(self._width or 0.0)
            h = float(self._height or 0.0)
            A  = b * h
            Iy = (b * h**3) / 12.0
            Iz = (h * b**3) / 12.0
            W  = b * h**2 / 6.0       # approx section modulus about strong axis
            return (A, Iy, Iz, W)

        if predefined == PreDefineSection.CircularArcBeam:
            Do = float(self.diameter or 0.0)
            Di = float(Do - 2 * self.thickness or 0.0)
            Ro = max(Do * 0.5, 0.0)
            Ri = max(Di * 0.5, 0.0)
            Ri = min(Ri, Ro)
            A  = pi * (Ro**2 - Ri**2)
            Iy = Iz = (pi/4.0) * (Ro**4 - Ri**4)
            W  = (pi/4.0) * (Ro**3 - Ri**3)  # approx section modulus for ring
            return (A, Iy, Iz, W)

        return (None, None, None, None)

# ########################### ElastoplasticBeam ###############################
class ElastoplasticBeam(ElasticBeam):
    """
    Elasto-plastic beam material.

    Adds:
      - sigma_y: yield stress (kPa)
      - yield_dir: principal direction of yield
    """

    def __init__(
        self,
        name: str = "Steel_Beam",
        type: BeamType = BeamType.Elastoplastic,
        comment: str = "Default elastoplastic steel beam",
        gamma: float = 77.0,
        E: float = 210e6,
        nu: float = 0.30,
        cross_section: CrossSectionType = CrossSectionType.PreDefine,
        *,
        predefined_section: Optional[PreDefineSection] = None,
        diameter: float = 1.0,
        thickness: float = 0.05,
        width: float = 1.0,
        height: float = 2.5,
        A: Optional[float] = None,
        Iy: Optional[float] = None,
        Iz: Optional[float] = None,
        W: Optional[float] = None,
        sigma_y: float = 355e3,
        yield_dir: Union[int, str, None] = 1,
        RayleighAlpha: float = 0.0,
        RayleighBeta: float = 0.0,
    ) -> None:
        super().__init__(
            name, type, comment, gamma, E, nu,
            cross_section,
            predefined_section=predefined_section,
            diameter=diameter, thickness=thickness,
            width=width, height=height,
            A=A, Iy=Iy, Iz=Iz, W=W,
            RayleighAlpha=RayleighAlpha, RayleighBeta=RayleighBeta,
        )
        self._sigma_y = sigma_y
        self._yield_dir = yield_dir

    def __repr__(self) -> str:
        return "<plx.materials.elastoplastic_beam>"

    @property
    def sigma_y(self) -> Optional[float]:
        return self._sigma_y

    @property
    def yield_dir(self) -> Union[int, str, None]:
        return self._yield_dir
