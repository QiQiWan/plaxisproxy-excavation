from ..core.plaxisobject import PlaxisObject

class BaseStructure(PlaxisObject):
    """
    Base class for Plaxis 3D structural objects (beam, anchor, embedded pile, etc).
    Provides unique ID, name, and Plaxis object reference.
    """
    __slots__ = ("_id", "_plx_id", "_name")

    def __init__(self, name: str, comment: str = "") -> None:
        """
        Initialize the base structure.

        Args:
            name (str): Name or label for the structure.
        """
        super().__init__(name=name, comment=comment)


    def __repr__(self) -> str:
        return f"<plx.structures.BaseStructure name='{self._name}'>"