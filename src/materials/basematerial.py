import uuid

class BaseMaterial:
    
    def __init__(self, name, type, comment, gamma, E, nu) -> None:
        self._id = uuid.uuid4()
        self._plx_id = None
        self._name = name
        self._type = type
        self._comment = comment
        self._gamma = gamma
        self._E = E
        self._nu = nu

    def __repr__(self) -> str:
        return f"<plx.materials.base>"
    
    @property
    def id(self):
        return self._id
    
    @property
    def plx_id(self):
        return self._plx_id
    
    @plx_id.setter
    def plx_id(self, value):
        self._plx_id = value

    @property
    def name(self):
        return self._name
    
    @property
    def mat_type(self):
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
