from .beammaterial import ElasticBeam, ElastoplasticBeam
from enum import Enum, auto

class LaterialResistanceType(Enum):
    Linear = auto()
    MultiLinear = auto()
    RelatedSoil = auto()

class ElasticPile(ElasticBeam):
    
    def __init__(self, name, type, comment, gamma, E, nu, cross_section, len1, len2,
                 laterial_type: LaterialResistanceType, fric_table_Tmax, F_max, **kwargs) -> None:
        ElasticBeam.__init__(self, name, type, comment, gamma, E, nu, cross_section, len1, len2)

        self._laterial_type = laterial_type
        self._fric_table_Tmax = fric_table_Tmax
        self._F_max = F_max

    def __repr__(self) -> str:
        return f"<plx.materials.elastic_pile>"

    @property
    def laterial_type(self):
        """Type of lateral constraint or support (string or enum)."""
        return self._laterial_type

    @property
    def fric_table_Tmax(self):
        """Maximum friction value from the friction table (kN or kN/m)."""
        return self._fric_table_Tmax

    @property
    def F_max(self):
        """Maximum allowable force (kN)."""
        return self._F_max

class ElastoplasticPile(ElasticPile, ElastoplasticBeam):

    def __init__(self, name, type, comment, gamma, E, nu, cross_section, len1, len2,
                 laterial_type, fric_table_Tmax, F_max, sigma_y, yield_dir):
        
        ElasticPile.__init__(self, 
            name=name, type=type, comment=comment, gamma=gamma, E=E, nu=nu,
            cross_section=cross_section, len1=len1, len2=len2,
            laterial_type=laterial_type, fric_table_Tmax=fric_table_Tmax, F_max=F_max
        )

        ElastoplasticBeam.__init__(
            self,
            name=name,
            type=type,
            comment=comment,
            gamma=gamma,
            E=E,
            nu=nu,
            cross_section=cross_section,
            len1=len1,
            len2=len2,
            sigma_y=sigma_y,
            yield_dir=yield_dir,
        )

        if not isinstance(sigma_y, (int, float)):
            raise TypeError("sigma_y must be a number.")
        if not isinstance(yield_dir, str):
            raise TypeError("yield_dir must be a string.")
        self._sigma_y = float(sigma_y)
        self._yield_dir = yield_dir

    @property
    def sigma_y(self) -> float:
        return self._sigma_y

    @property
    def yield_dir(self) -> str:
        return self._yield_dir

    def __repr__(self) -> str:
        return f"<plx.materials.elastoplastic_pile>"
        