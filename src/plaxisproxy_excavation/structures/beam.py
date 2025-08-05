from ..materials.beammaterial import *
from ..geometry import *
from basestructure import BaseStructure

class Beam(BaseStructure):
    """
    Beam object for Plaxis 3D, defined by a 3D line (with exactly two points)
    and a beam material/type.
    """

    def __init__(self, name: str, line: Line3D, beam_type) -> None:
        """
        Initialize the beam with a 3D line and beam material/type.

        Args:
            line (Line3D): The 3D line defining the beam (must be exactly two points).
            beam_type: The beam material or type.

        Raises:
            ValueError: If the line does not have exactly two points.
        """
        super().__init__(name)
        if len(line) != 2:
            raise ValueError("Beam line must have exactly two points!")
        self._line = line
        self._beam_type = beam_type

    @property
    def line(self) -> Line3D:
        """3D line object representing the beam (must have exactly two points)."""
        return self._line

    @property
    def beam_type(self):
        """Beam material or type (material object or string/enum)."""
        return self._beam_type

    def get_points(self):
        """Get the start and end points of the beam as a list of Point objects."""
        return self._line.get_points()

    def __repr__(self) -> str:
        return "<plx.structures.beam>"
