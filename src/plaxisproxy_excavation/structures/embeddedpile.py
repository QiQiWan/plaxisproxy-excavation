from typing import List, Any
from ..geometry import Line3D
from .basestructure import BaseStructure
# Assuming PileMaterial is a defined class in the materials module
from ..materials.pilematerial import ElasticPile

class EmbeddedPile(BaseStructure):
    """
    Embedded pile object for Plaxis 3D, defined by a 3D line (with exactly two points)
    and a pile material/type.
    """
    __slots__ = ("_line", "_pile_type")

    def __init__(self, name: str, line: Line3D, pile_type: ElasticPile) -> None:
        """
        Initializes the embedded pile with a 3D line and pile material/type.

        Args:
            name (str): The name of the pile.
            line (Line3D): The 3D line defining the embedded pile (must be exactly two points).
            pile_type (PileMaterial): The pile material or type.

        Raises:
            ValueError: If the line does not have exactly two points.
        """
        super().__init__(name)
        if not isinstance(line, Line3D):
            raise TypeError("Embedded pile line must be a Line3D instance.")
        if len(line) != 2:
            raise ValueError("Embedded pile line must have exactly two points!")
        self._line = line
        if not isinstance(pile_type, ElasticPile):
            raise TypeError("pile_type must be an ElasticPile instance.")
        self._pile_type = pile_type


    @property
    def line(self) -> Line3D:
        """3D line object representing the embedded pile."""
        return self._line

    @property
    def pile_type(self) -> ElasticPile:
        """Pile material or type (material object or string/enum)."""
        return self._pile_type

    def get_points(self) -> List[Any]:
        """Get the start and end points of the pile as a list of Point objects."""
        return self._line.get_points()

    def length(self) -> float:
        """Calculate the geometric length of the pile."""
        return self._line.length

    def __repr__(self) -> str:
        return f"<plx.structures.EmbeddedPile name='{self.name}' type='{self._pile_type}'>"