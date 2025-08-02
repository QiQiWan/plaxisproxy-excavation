import uuid
from typing import Optional, Any

class BaseStructure:
    """
    Base class for Plaxis 3D structural objects (beam, anchor, embedded pile, etc).
    Provides unique ID, name, and Plaxis object reference.
    """
    __slots__ = ("_id", "_plx_id", "_name")

    def __init__(self, name: str) -> None:
        """
        Initialize the base structure.

        Args:
            name (str): Name or label for the structure.
        """
        self._id = uuid.uuid4()
        self._plx_id: Optional[str] = None  # Plaxis internal object id
        self._name = name

    @property
    def id(self) -> uuid.UUID:
        """Unique identifier (UUID) for the structure."""
        return self._id

    @property
    def name(self) -> str:
        """Name or label for the structure."""
        return self._name
    
    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def plx_id(self) -> Optional[str]:
        """Reference or ID in Plaxis (can be set externally, optional)."""
        return self._plx_id

    @plx_id.setter
    def plx_id(self, value: str):
        self._plx_id = value

    def __repr__(self) -> str:
        return f"<plx.structures.BaseStructure name='{self._name}'>"