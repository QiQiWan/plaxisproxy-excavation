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
        name: str, 
        line: Line3D,
        well_type: WellType,
        h_min: float = 0.0,
    ) -> None:
        super().__init__(name)
        if len(line) != 2:
            raise ValueError("Well line must have exactly two points (well top and bottom)!")
        if well_type not in (WellType.Extraction, WellType.Infiltration):
            raise ValueError("well_type must be WellType.Extraction or WellType.Infiltration")
        self._line = line
        self._pos = line.xy_location()
        self._well_type = well_type
        self._h_min = h_min

    def move(self, dx: float, dy: float, dz: float = 0.0) -> None:
        """
        Moves the well by a given displacement in the x, y, and z directions.

        This method calculates the new coordinates for the well's top and bottom
        points based on the provided displacements, and then reconstructs the
        internal Line3D object to reflect the new position.

        Args:
            dx (float): The displacement in the X-direction.
            dy (float): The displacement in the Y-direction.
            dz (float, optional): The displacement in the Z-direction. Defaults to 0.0.
        """
        # Get the current top and bottom points of the well.
        try:
            p_top, p_bottom = self.get_points()
        except ValueError:
            # Handle cases where the line might not have exactly two points, though
            # the __init__ should prevent this.
            raise ValueError("Cannot move well: internal line is not properly defined.")

        # Calculate the new coordinates for both points.
        new_top = Point(p_top.x + dx, p_top.y + dy, p_top.z + dz)
        new_bottom = Point(p_bottom.x + dx, p_bottom.y + dy, p_bottom.z + dz)

        # Create a new Line3D object with the new points.
        new_line = Line3D(PointSet([new_top, new_bottom]))

        # Use the existing line setter to update the well's state.
        #    This is the best practice as it ensures that both the internal
        #    _line and the cached _pos are updated consistently.
        self.line = new_line

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
    def pos(self):
        return self._pos

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