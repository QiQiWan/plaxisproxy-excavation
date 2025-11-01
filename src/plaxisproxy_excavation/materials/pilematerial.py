from __future__ import annotations
from typing import Optional, Union, Iterable, Tuple
from enum import Enum

from .beammaterial import (
    BeamType, CrossSectionType, PreDefineSection,
    ElasticBeam, ElastoplasticBeam,
)

class LateralResistanceType(Enum):
    """Axial skin resistance modes as in PLAXIS UI."""
    Linear = "Linear"
    MultiLinear = "Multi-linear"
    LayerDependent = "Layer dependent"  # was RelatedSoil

class ElasticPile(ElasticBeam):
    """
    Elastic pile material.

    Geometry & elastic fields come from ElasticBeam.
    Predefined sections supported for piles: Cylinder / CircularArcBeam / Square.
    (Rectangle is not supported by embedded pile/beam in your setup.)

    Axial skin resistance:
      - Linear         → needs T_skin_start_max (kN/m), T_skin_end_max (kN/m)
      - Multi-linear   → optional friction_curve: iterable[(u, T)] or similar
      - Layer dependent→ no direct numeric inputs on the material (taken from layers)

    Base resistance:
      - F_max (kN)
    """

    def __init__(
        self,
        name: str = "Elastic_Pile",
        type: BeamType = BeamType.Elastic,
        comment: str = "Elastic pile material",
        gamma: float = 25.0,                          # kN/m3 (RC pile)
        E: float = 30e6,                              # kPa (~30 GPa)
        nu: float = 0.25,
        cross_section: CrossSectionType = CrossSectionType.PreDefine,
        *,
        predefined_section: PreDefineSection = PreDefineSection.Cylinder,
        # Predefined geometry
        diameter: float = 1.0,                        # m, solid circular pile
        thickness: float = 0.05,                      # m, for tube pile
        width: float = 1.0,                           # m, for Square
        height: float = 1.0,                          # m, kept for compatibility
        # Custom section (auto-compute defaults for Cylinder)
        A: Optional[float] = None,         # m2, = 0.785 for d=1.0
        Iy: Optional[float] = None,  # m4, = 0.0491
        Iz: Optional[float] = None,  # m4
        W: Optional[float]  = None,  # m3, = 0.196
        # Axial skin resistance
        lateral_type: LateralResistanceType = LateralResistanceType.Linear,
        T_skin_start_max: Optional[float] = 0.0,      # kN/m (top)
        T_skin_end_max: Optional[float] = 800.0,      # kN/m (toe)
        friction_curve: Optional[Iterable[Tuple[float, float]]] = None,
        # Backward-compat shim (old field)
        fric_table_Tmax: Optional[float] = None,
        # Base resistance
        F_max: Optional[float] = 8000.0,              # kN
        # Damping
        RayleighAlpha: float = 0.0,
        RayleighBeta: float = 0.0,
    ) -> None:
        if predefined_section == PreDefineSection.Rectangle:
            raise Exception("Embedded pile/beam has no rectangular section. Use Square instead.")
        super().__init__(
            name=name, type=type, comment=comment, gamma=gamma, E=E, nu=nu,
            cross_section=cross_section,
            predefined_section=predefined_section,
            diameter=diameter, thickness=thickness,
            width=width, height=height,
            A=A, Iy=Iy, Iz=Iz, W=W,
            RayleighAlpha=RayleighAlpha, RayleighBeta=RayleighBeta,
        )
        self._lateral_type = lateral_type
        self._T_skin_start_max = T_skin_start_max
        # prefer explicit T_skin_end_max; else fall back to legacy fric_table_Tmax
        self._T_skin_end_max = T_skin_end_max if T_skin_end_max is not None else fric_table_Tmax
        self._friction_curve = friction_curve
        self._F_max = F_max

    def __repr__(self) -> str:
        return "<plx.materials.elastic_pile>"

    # --- pile-specific properties
    @property
    def lateral_type(self) -> LateralResistanceType:
        return self._lateral_type

    # legacy alias kept (read-only) to avoid breaking old code
    @property
    def laterial_type(self) -> LateralResistanceType:
        return self._lateral_type

    @property
    def T_skin_start_max(self) -> Optional[float]:
        return self._T_skin_start_max

    @property
    def T_skin_end_max(self) -> Optional[float]:
        return self._T_skin_end_max

    @property
    def friction_curve(self) -> Optional[Iterable[Tuple[float, float]]]:
        return self._friction_curve

    @property
    def F_max(self) -> Optional[float]:
        return self._F_max

    # --- override section_properties to support Square ---
    def section_properties(self, predefined: Optional[PreDefineSection] = None):
        """
        Returns (A, Iy, Iz, W).
        For Custom → returns user values.
        For PreDefine → supports Cylinder / CircularArcBeam / Square (width==height).
        """
        if self.cross_section == CrossSectionType.Custom:
            return super().section_properties(predefined)

        predefined = predefined or self.predefined_section or PreDefineSection.Cylinder

        if predefined.name == "Square":
            b = float(self.width or 0.0)
            A  = b * b
            Iy = (b**4) / 12.0
            Iz = Iy
            W  = (b**3) / 6.0 if b > 0 else 0.0
            return (A, Iy, Iz, W)

        return super().section_properties(predefined)


class ElastoplasticPile(ElasticPile, ElastoplasticBeam):
    """
    Elastoplastic pile material.
    Inherits Elastic & geometry from ElasticPile, and plastic fields from ElastoplasticBeam.
    """

    def __init__(
        self,
        name: str = "EP_Pile",
        type: BeamType = BeamType.Elastoplastic,
        comment: str = "Elasto-plastic pile material",
        gamma: float = 26.0,                         # kN/m³
        E: float = 31e6,                             # kPa (~31 GPa)
        nu: float = 0.25,
        cross_section: CrossSectionType = CrossSectionType.PreDefine,
        *,
        predefined_section: PreDefineSection = PreDefineSection.Cylinder,
        # geometry
        diameter: float = 1.0,                       # m
        thickness: float = 0.05,                     # m
        width: float = 1.0,                          # m
        height: float = 1.0,                         # m
        # custom (section properties left for auto-compute)
        A: Optional[float] = None,
        Iy: Optional[float] = None,
        Iz: Optional[float] = None,
        W: Optional[float] = None,
        # axial skin resistance
        lateral_type: LateralResistanceType = LateralResistanceType.Linear,
        T_skin_start_max: Optional[float] = 0.0,     # kN/m (top)
        T_skin_end_max: Optional[float] = 800.0,     # kN/m (toe)
        friction_curve: Optional[Iterable[Tuple[float, float]]] = None,
        fric_table_Tmax: Optional[float] = None,
        # base resistance
        F_max: Optional[float] = 8000.0,             # kN
        # plasticity
        sigma_y: float = 355e3,                      # kPa (~355 MPa)
        yield_dir: Union[int, str, None] = 1,
        # damping
        RayleighAlpha: float = 0.0,
        RayleighBeta: float = 0.0,
    ) -> None:

        ElasticPile.__init__(
            self,
            name=name, type=type, comment=comment, gamma=gamma, E=E, nu=nu,
            cross_section=cross_section,
            predefined_section=predefined_section,
            diameter=diameter, thickness=thickness,
            width=width, height=height,
            A=A, Iy=Iy, Iz=Iz, W=W,
            lateral_type=lateral_type,
            T_skin_start_max=T_skin_start_max,
            T_skin_end_max=T_skin_end_max,
            friction_curve=friction_curve,
            fric_table_Tmax=fric_table_Tmax,
            F_max=F_max,
            RayleighAlpha=RayleighAlpha, RayleighBeta=RayleighBeta,
        )

        ElastoplasticBeam.__init__(
            self,
            name=name, type=type, comment=comment, gamma=gamma, E=E, nu=nu,
            cross_section=cross_section,
            predefined_section=predefined_section,
            diameter=diameter, thickness=thickness,
            width=width, height=height,
            A=A, Iy=Iy, Iz=Iz, W=W,
            sigma_y=sigma_y, yield_dir=yield_dir,
            RayleighAlpha=RayleighAlpha, RayleighBeta=RayleighBeta,
        )

    def __repr__(self) -> str:
        return "<plx.materials.elastoplastic_pile>"
