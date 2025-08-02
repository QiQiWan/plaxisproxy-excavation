from basestructure import BaseStructure
from ..geometry import *
from enum import Enum

class WellType(Enum):
    Extraction = "Extraction"
    Infiltration = "Infiltration"

class Well(BaseStructure):
    """
    Well object for Plaxis 3D, supporting position modification via x/y or direct line replacement.
    """
    def __init__(
        self,
        line: Line3D,
        well_type: WellType,
        h_min: float = 0.0,
    ) -> None:
        super().__init__()
        if len(line) != 2:
            raise ValueError("Well line must have exactly two points (well top and bottom)!")
        if well_type not in (WellType.Extraction, WellType.Infiltration):
            raise ValueError("well_type must be WellType.Extraction or WellType.Infiltration")
        self._line = line
        self._pos = line.xy_location()
        self._well_type = well_type
        self._h_min = h_min

    @property
    def line(self) -> Line3D:
        """3D line object representing the well axis (two points: top and bottom)."""
        return self._line

    @line.setter
    def line(self, new_line: Line3D):
        """Replace the well's line object and update position info."""
        if len(new_line) != 2:
            raise ValueError("Well line must have exactly two points!")
        self._line = new_line
        self._pos = new_line.xy_location()

    @property
    def x(self):
        """Well x location in XY-plane (if Z-axis vertical line)."""
        return self._pos[0] if self._pos is not None else None

    @x.setter
    def x(self, value):
        if self._pos is None:
            raise ValueError("Well position undefined, cannot set x.")
        y = self._pos[1]
        # Modify the x value of the line, and keep y value and z value.
        pts = self._line.get_points()
        if len(pts) != 2:
            raise ValueError("Well line must have exactly two points!")
        # Create PointSet/Line3D，to prevent direct modification of the object from affecting other references
        new_pts = [Point(value, y, pts[0].z), Point(value, y, pts[1].z)]
        self._line = Line3D(PointSet(new_pts))
        self._pos = (value, y)

    @property
    def y(self):
        """Well y location in XY-plane (if Z-axis vertical line)."""
        return self._pos[1] if self._pos is not None else None

    @y.setter
    def y(self, value):
        if self._pos is None:
            raise ValueError("Well position undefined, cannot set y.")
        x = self._pos[0]
        pts = self._line.get_points()
        if len(pts) != 2:
            raise ValueError("Well line must have exactly two points!")
        new_pts = [Point(x, value, pts[0].z), Point(x, value, pts[1].z)]
        self._line = Line3D(PointSet(new_pts))
        self._pos = (x, value)

    @property
    def well_type(self) -> WellType:
        return self._well_type

    @well_type.setter
    def well_type(self, value):
        if value not in (WellType.Extraction, WellType.Infiltration):
            raise ValueError("well_type must be WellType.Extraction or WellType.Infiltration")
        self._well_type = value

    @property
    def h_min(self) -> float:
        return self._h_min

    @h_min.setter
    def h_min(self, value: float):
        self._h_min = value

    def get_points(self):
        return self._line.get_points()

    def __repr__(self) -> str:
        return "<plx.structures.well>"