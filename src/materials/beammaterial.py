from basematerial import BaseMaterial
from enum import Enum, auto

class BeamType(Enum):
    Elastic = auto()
    Elastoplastic = auto()

class CrossSectionType(Enum):
    PreDefine = auto()
    Custom = auto()

class PreDefineSection(Enum):
    CircularArcBeam = auto()
    Cylinder = auto()
    Rectangle = auto()

class ElasticBeam(BaseMaterial):

    def __init__(self, name, type, comment, gamma, E, nu, cross_section, len1, len2) -> None:
        super().__init__(name, type, comment, gamma, E, nu)

        self._cross_section = cross_section
        self._len1 = len1
        self._len2 = len2

    def __repr__(self) -> str:
        return f"<plx.materials.elastic_beam>"

    @property
    def cross_section(self):
        """Cross-section type or parameters (usually string or dict)."""
        return self._cross_section

    @property
    def len1(self):
        """Length in the first principal direction (m)."""
        return self._len1

    @property
    def len2(self):
        """Length in the second principal direction (m)."""
        return self._len2

class ElastoplasticBeam(ElasticBeam):

    def __init__(self, name, type, comment, gamma, E, nu, cross_section, len1, len2, 
                 sigma_y, yield_dir, **kwargs) -> None:
        super().__init__(name, type, comment, gamma, E, nu, cross_section, len1, len2)

        self._sigma_y = sigma_y
        self._yield_dir = yield_dir

    def __repr__(self) -> str:
        return f"<plx.materials.elastoplastic_beam>"

    @property
    def sigma_y(self):
        """Yield strength of the material (kN/m²)."""
        return self._sigma_y

    @property
    def yield_dir(self):
        """Yield direction indicator (usually 1, 2, or a string description)."""
        return self._yield_dir
