from .basestructure import BaseStructure
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
        name: str, 
        line: Line3D,
        well_type: WellType,
        h_min: float = 0.0,
    ) -> None:
        super().__init__(name)
        if not isinstance(line, Line3D):
            raise TypeError("Well line must be a Line3D instance.")
        if len(line) != 2:
            raise ValueError("Well line must have exactly two points (well top and bottom)!")
        if not isinstance(well_type, WellType):
            raise TypeError("well_type must be a WellType enum value.")
        if not isinstance(well_type, WellType) or well_type not in (WellType.Extraction, WellType.Infiltration):
            raise ValueError("well_type must be WellType.Extraction or WellType.Infiltration")
        if not isinstance(h_min, (int, float)):
            raise TypeError("h_min must be a numeric value.")
        self._line = line
        self._pos = line.xy_location()
        self._well_type = well_type
        self._h_min = h_min

    def move(self, dx: float, dy: float, dz: float = 0.0) -> None:
        """Moves the well by a given displacement in the x, y, and z directions."""
        if not all(isinstance(d, (int, float)) for d in (dx, dy, dz)):
            raise TypeError("Move displacements dx, dy, dz must be numeric.")
        try:
            p_top, p_bottom = self.get_points()
        except ValueError:
            raise ValueError("Cannot move well: internal line is not properly defined.")

        # Calculate the new coordinates for both points.
        new_top = Point(p_top.x + dx, p_top.y + dy, p_top.z + dz)
        new_bottom = Point(p_bottom.x + dx, p_bottom.y + dy, p_bottom.z + dz)

        # Create a new Line3D object with the new points.
        new_line = Line3D(PointSet([new_top, new_bottom]))

        # Update the well's internal line and position
        self.line = new_line

    @property
    def line(self) -> Line3D:
        """3D line object representing the well axis (two points: top and bottom)."""
        return self._line

    @line.setter
    def line(self, new_line: Line3D):
        if not isinstance(new_line, Line3D):
            raise TypeError("Well line must be a Line3D instance.")
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
        if not isinstance(value, (int, float)):
            raise TypeError("x coordinate must be a number.")
        y = self._pos[1]
        pts = self._line.get_points()
        if len(pts) != 2:
            raise ValueError("Well line must have exactly two points!")
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
        if not isinstance(value, (int, float)):
            raise TypeError("y coordinate must be a number.")
        x = self._pos[0]
        pts = self._line.get_points()
        if len(pts) != 2:
            raise ValueError("Well line must have exactly two points!")
        new_pts = [Point(x, value, pts[0].z), Point(x, value, pts[1].z)]
        self._line = Line3D(PointSet(new_pts))
        self._pos = (x, value)

    @property
    def pos(self):
        return self._pos

    @property
    def well_type(self) -> WellType:
        return self._well_type

    @well_type.setter
    def well_type(self, value):
        if not isinstance(value, WellType):
            raise TypeError("well_type must be a WellType enum value.")
        if value not in (WellType.Extraction, WellType.Infiltration):
            raise ValueError("well_type must be WellType.Extraction or WellType.Infiltration")
        self._well_type = value

    @property
    def h_min(self) -> float:
        return self._h_min

    @h_min.setter
    def h_min(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError("h_min must be a numeric value.")
        self._h_min = value

    def get_points(self):
        return self._line.get_points()

    def __repr__(self) -> str:
        return "<plx.structures.well>"