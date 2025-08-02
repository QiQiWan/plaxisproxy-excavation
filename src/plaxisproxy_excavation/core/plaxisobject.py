import uuid
from typing import Optional

class PlaxisObject:
    """
    The common base class of PLAXIS objects。

    This class encapsulates all the common core attributes shared by all PLAXIS proxy objects,
    including a unique internal ID, a handle (plx_id) for the PLAXIS server reference,
    and a user-defined name.

    Attributes:
        _id (str): A unique internal identifier (UUID) used to track the object within the session.
        _plx_id (Optional[str]): The object handle returned by the PLAXIS script interface.
        This value is None until the object is synchronized to the PLAXIS server.
        _name (str): The name assigned by the user to this object, which will also be used in PLAXIS.
    """
    def __init__(self, name: str, comment: str):
        """
        Initialize PlaxisObject。

        Args:
            name (str): name of the Plaxis object.
            comment (str): comment of the Plaxis object.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("The type of name must be a string!")
            
        self._id: str = str(uuid.uuid4())
        self._plx_id: Optional[str] = None
        self._name: str = name
        self._comment: str = comment

    @property
    def id(self) -> str:
        """Get the unique ID of the object."""
        return self._id

    @property
    def plx_id(self) -> Optional[str]:
        """Get the handle of Plaxis object."""
        return self._plx_id
    
    @plx_id.setter
    def plx_id(self, value):
        self._plx_id = value
 
    @property
    def name(self) -> str:
        """Get the name of Plaxis object."""
        return self._name
    
    @name.setter
    def name(self, value):
        self.name = value

    @property
    def comment(self) -> str:
        """Get comment or description."""
        return self._comment

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', id='{self.id}')>"