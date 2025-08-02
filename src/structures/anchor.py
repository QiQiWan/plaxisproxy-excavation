from ..materials.anchormaterial import *
from ..geometry import *
from basestructure import BaseStructure

class Anchor(BaseStructure):
    """
    Anchor (tieback) object for Plaxis 3D, defined by a 3D line (with exactly two points)
    and an anchor material/type.
    """

    def __init__(self, line: Line3D, anchor_type) -> None:
        """
        Initialize the anchor with a 3D line and anchor material/type.

        Args:
            line (Line3D): The 3D line defining the anchor (must be exactly two points).
            anchor_type: The anchor material or type.

        Raises:
            ValueError: If the line does not have exactly two points.
        """
        super().__init__()
        if len(line) != 2:
            raise ValueError("Anchor line must have exactly two points!")
        self._line = line
        self._anchor_type = anchor_type

    @property
    def line(self) -> Line3D:
        """3D line object representing the anchor (must have exactly two points)."""
        return self._line

    @property
    def anchor_type(self):
        """Anchor material or type (material object or string/enum)."""
        return self._anchor_type

    def get_points(self):
        """Get the start and end points of the anchor as a list of Point objects."""
        return self._line.get_points()

    def __repr__(self) -> str:
        return "<plx.structures.anchor>"
