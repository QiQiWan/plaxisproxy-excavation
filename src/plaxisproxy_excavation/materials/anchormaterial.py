from enum import Enum, auto
from ..core.plaxisobject import PlaxisObject

class AnchorType(Enum):
    Elastic = auto()
    Elastoplastic = auto()
    ElastoPlasticResidual = auto()

class ElasticAnchor(PlaxisObject):
    """
    Defines an elastic anchor material.
    """
    def __init__(self, name, type: AnchorType, comment, EA):
        """
        Initializes the elastic anchor material.

        Args:
            name (str): The name of the material.
            EA (float): Axial stiffness of the anchor material [kN].
        """
        super.__init__(name, comment)
        self._type = type
        self._EA = EA

    def __repr__(self) -> str:
        return f"<plx.materials.elastic_anchor>"
    
    @property
    def type(self):
        """Type of the object/material."""
        return self._type

    @property
    def EA(self):
        """Axial stiffness EA (kN)."""
        return self._EA

class ElastoplasticAnchor(ElasticAnchor):
    
    def __init__(self, name, type: AnchorType, comment, EA, F_max_tens, F_max_comp) -> None:
        super().__init__(name, type, comment, EA)
    
        self._F_max_tens = F_max_tens
        self._F_max_comp = F_max_comp

    def __repr__(self) -> str:
        return f"<plx.materials.elastoplastic_anchor>"
    
    @property
    def F_max_tens(self):
        """Maximum allowable tensile force (kN)."""
        return self._F_max_tens

    @property
    def F_max_comp(self):
        """Maximum allowable compressive force (kN)."""
        return self._F_max_comp

class ElastoPlasticResidualAnchor(ElastoplasticAnchor):

    def __init__(self, name, type: AnchorType, comment, EA, F_max_tens, F_max_comp,
                 F_res_tens, F_res_comp) -> None:
        super().__init__(name, type, comment, EA, F_max_tens, F_max_comp)

        self._F_res_tens = F_res_tens
        self._F_res_comp = F_res_comp
    
    def __repr__(self) -> str:
        return f"<plx.materials.elastoplastic_residual_anchor>"
    
    @property
    def F_res_tens(self):
        """Residual tensile force (kN)."""
        return self._F_res_tens

    @property
    def F_res_comp(self):
        """Residual compressive force (kN)."""
        return self._F_res_comp
