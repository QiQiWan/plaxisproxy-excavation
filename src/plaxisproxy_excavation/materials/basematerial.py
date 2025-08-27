from ..core.plaxisobject import PlaxisObject

class BaseMaterial(PlaxisObject):
    

    def __init__(self, name, type, comment, gamma, E, nu) -> None:
        super().__init__(name, comment)
        # 使用内部属性名存储，避免与property冲突
        self._type = type
        self._comment = comment
        self._gamma = gamma
        self._E = E
        self._nu = nu

    def __repr__(self) -> str:
        return f"<plx.materials.BaseMaterial>"
    
    @property
    def type(self):
        return self._type
    
    @property
    def comment(self):
        return self._comment
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def E(self):
        return self._E
    
    @property
    def nu(self):
        return self._nu
