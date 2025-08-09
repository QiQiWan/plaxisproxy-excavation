from .basematerial import BaseMaterial
from enum import Enum, auto

class PlateType(Enum):
    Elastic = auto()
    Elastoplastic = auto()

class ElasticPlate(BaseMaterial):

    def __init__(self, name, type, comment, gamma, E, nu, preventpunch, isotropic, 
                 E2=0.0, G12=0.0, G13=0.0, G23=0.0) -> None:
        super().__init__(name, type, comment, gamma, E, nu)

        self._preventpuch = preventpunch
        self._isotropic = isotropic
        self._E2 = E2
        self._G12 = G12
        self._G13 = G13
        self._G23 = G23

    def __repr__(self) -> str:
        return f"<plx.materials.elastic_plate>"
    
    @property
    def preventpunch(self):
        """Whether punching shear prevention is enabled (True/False)."""
        return self._preventpuch

    @property
    def isotropic(self):
        """Whether the material is isotropic (True/False)."""
        return self._isotropic

    @property
    def E2(self):
        """Elastic modulus in the secondary principal direction (kN/m²)."""
        return self._E2

    @property
    def G12(self):
        """Shear modulus in the 1-2 direction (kN/m²)."""
        return self._G12

    @property
    def G13(self):
        """Shear modulus in the 1-3 direction (kN/m²)."""
        return self._G13

    @property
    def G23(self):
        """Shear modulus in the 2-3 direction (kN/m²)."""
        return self._G23
    
class ElastoplasticPlate(ElasticPlate):

    def __init__(self, name, type, comment, gamma, E, nu, preventpunch, isotropic, 
                 E2=0.0, G12=0.0, G13=0.0, G23=0.0,
                 sigma_y_11=0.0, W_11=0.0, sigma_y_22=0.0, W_22=0.0) -> None:
        super().__init__(name, type, comment, gamma, E, nu, preventpunch, isotropic, E2, G12, G13, G23)

        self._sigma_y_11 = sigma_y_11
        self._W_11 = W_11
        self._sigma_y_22 = sigma_y_22
        self._W_22 = W_22
    
    def __repr__(self) -> str:
        return f"<plx.materials.elastoplastic_plate>"
        
    @property
    def sigma_y_11(self):
        """Yield strength in the 1-1 principal direction (kN/m²)."""
        return self._sigma_y_11

    @property
    def W_11(self):
        """Section modulus in the 1-1 principal direction (m³)."""
        return self._W_11

    @property
    def sigma_y_22(self):
        """Yield strength in the 2-2 principal direction (kN/m²)."""
        return self._sigma_y_22

    @property
    def W_22(self):
        """Section modulus in the 2-2 principal direction (m³)."""
        return self._W_22
