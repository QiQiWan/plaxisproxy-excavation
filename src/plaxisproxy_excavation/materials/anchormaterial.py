from __future__ import annotations
from typing import Optional, Union
from enum import Enum
from ..core.plaxisobject import PlaxisObject

class AnchorType(Enum):
    Elastic = "Elastic"
    Elastoplastic = "Elasto-plastic"
    ElastoPlasticResidual = "Elastoplastic with residual strength"

def _normalize_anchor_type(t: Union[str, AnchorType]) -> str:
    """Accept Enum or string and return a clean string label."""
    if isinstance(t, AnchorType):
        return t.value
    return str(t)

# =============================== BaseAnchor ==================================
class BaseAnchor(PlaxisObject):
    """
    Base class for all anchor materials. Holds common fields:

    - type : AnchorType
    - EA   : axial stiffness (kN)
    - F_max_tens / F_max_comp : ultimate tensile / compressive capacities (kN)
    - F_res_tens / F_res_comp : residual tensile / compressive capacities (kN)
    """

    def __init__(
        self,
        name: str,
        comment: str,
        *,
        type: AnchorType,
        EA: float,
        F_max_tens: Optional[float] = None,
        F_max_comp: Optional[float] = None,
        F_res_tens: Optional[float] = None,
        F_res_comp: Optional[float] = None,
    ) -> None:
        super().__init__(name, comment)
        self._type = type
        self._EA = float(EA)
        self._F_max_tens = None if F_max_tens is None else float(F_max_tens)
        self._F_max_comp = None if F_max_comp is None else float(F_max_comp)
        self._F_res_tens = None if F_res_tens is None else float(F_res_tens)
        self._F_res_comp = None if F_res_comp is None else float(F_res_comp)

    # --- properties (shared) ---
    @property
    def type(self) -> AnchorType:
        return self._type

    @property
    def EA(self) -> float:
        """Axial stiffness EA (kN)."""
        return self._EA

    @property
    def F_max_tens(self) -> Optional[float]:
        """Ultimate tensile capacity (kN)."""
        return self._F_max_tens

    @property
    def F_max_comp(self) -> Optional[float]:
        """Ultimate compressive capacity (kN)."""
        return self._F_max_comp

    @property
    def F_res_tens(self) -> Optional[float]:
        """Residual tensile capacity after yielding (kN)."""
        return self._F_res_tens

    @property
    def F_res_comp(self) -> Optional[float]:
        """Residual compressive capacity after yielding (kN)."""
        return self._F_res_comp


# ============================== ElasticAnchor ================================
class ElasticAnchor(BaseAnchor):
    """
    Elastic (linear) node-to-node / cable anchor material.
    """

    def __init__(
        self,
        name: str = "Elastic_Anchor",
        type: AnchorType = AnchorType.Elastic,
        comment: str = "Elastic anchor material",
        EA: float = 1.0e5,  # kN (example default)
    ) -> None:
        super().__init__(
            name=name,
            comment=comment,
            type=type,
            EA=EA,
            F_max_tens=None,
            F_max_comp=None,
            F_res_tens=None,
            F_res_comp=None,
        )

    def __repr__(self) -> str:
        return "<plx.materials.elastic_anchor>"


# =========================== ElastoplasticAnchor =============================
class ElastoplasticAnchor(BaseAnchor):
    """
    Elastoplastic anchor material with symmetric tensile/compressive capacity.
    """

    def __init__(
        self,
        name: str = "EP_Anchor",
        type: AnchorType = AnchorType.Elastoplastic,
        comment: str = "Elasto-plastic anchor material",
        EA: float = 2.0e5,                  # kN (axial stiffness)
        F_max_tens: Optional[float] = 2000.0,  # kN
        F_max_comp: Optional[float] = 500.0,   # kN
    ) -> None:
        super().__init__(
            name=name,
            comment=comment,
            type=type,
            EA=EA,
            F_max_tens=F_max_tens,
            F_max_comp=F_max_comp,
            F_res_tens=None,
            F_res_comp=None,
        )

    def __repr__(self) -> str:
        return "<plx.materials.elastoplastic_anchor>"


# ======================= ElastoPlasticResidualAnchor =========================
class ElastoPlasticResidualAnchor(ElastoplasticAnchor):
    """
    Elastoplastic anchor with residual capacity after yielding.
    """

    def __init__(
        self,
        name: str = "EP_Residual_Anchor",
        type: AnchorType = AnchorType.ElastoPlasticResidual,
        comment: str = "Elasto-plastic residual anchor material",
        EA: float = 2.0e5,                     # kN
        F_max_tens: Optional[float] = 2000.0,  # kN
        F_max_comp: Optional[float] = 500.0,   # kN
        F_res_tens: Optional[float] = 400.0,   # kN
        F_res_comp: Optional[float] = 100.0,   # kN
    ) -> None:
        # Initialize ultimate capacities via parent
        super().__init__(
            name=name,
            type=type,
            comment=comment,
            EA=EA,
            F_max_tens=F_max_tens,
            F_max_comp=F_max_comp,
        )
        # Add residuals on top of ElastoplasticAnchor
        self._F_res_tens = None if F_res_tens is None else float(F_res_tens)
        self._F_res_comp = None if F_res_comp is None else float(F_res_comp)

    def __repr__(self) -> str:
        return "<plx.materials.elastoplastic_residual_anchor>"
