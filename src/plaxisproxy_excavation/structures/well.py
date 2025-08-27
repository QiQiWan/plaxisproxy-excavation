from __future__ import annotations
from typing import Optional, Union
from enum import Enum
from .basestructure import BaseStructure, TwoPointLineMixin
from ..geometry import Point, PointSet, Line3D

class WellType(Enum):
    Extraction = "Extraction"
    Infiltration = "Infiltration"

class Well(BaseStructure, TwoPointLineMixin):
    """
    Well with a two-point Line3D (top/bottom), type and h_min.
    """

    def __init__(
        self,
        name: str,
        line: Optional[Line3D] = None,
        *,
        p_start: Optional[Point] = None,
        p_end: Optional[Point] = None,
        well_type: Union[WellType, str] = WellType.Extraction,
        h_min: float = 0.0,
    ) -> None:
        super().__init__(name)
        self._line = self._init_line_from_args(line=line, p_start=p_start, p_end=p_end)

        # 兼容字符串（如 "ConstantHead"），保持宽松，映射交给 Mapper 层处理
        if not isinstance(well_type, (WellType, str)):
            raise TypeError("well_type must be WellType or str.")
        if not isinstance(h_min, (int, float)):
            raise TypeError("h_min must be numeric.")
        self._well_type = well_type
        self._h_min = float(h_min)
        self._pos = self._line.xy_location()

    def get_points(self):
        return self._line.get_points()

    def move(self, dx: float, dy: float, dz: float = 0.0) -> None:
        if not all(isinstance(d, (int, float)) for d in (dx, dy, dz)):
            raise TypeError("dx/dy/dz must be numeric.")
        p_top, p_bottom = self.get_points()
        new_top = Point(p_top.x + dx, p_top.y + dy, p_top.z + dz)
        new_bottom = Point(p_bottom.x + dx, p_bottom.y + dy, p_bottom.z + dz)
        self.line = Line3D(PointSet([new_top, new_bottom]))
        self._pos = self._line.xy_location()

    @property
    def x(self):
        return self._pos[0] if self._pos is not None else None

    @x.setter
    def x(self, value):
        if self._pos is None:
            raise ValueError("Well position undefined, cannot set x.")
        if not isinstance(value, (int, float)):
            raise TypeError("x must be numeric.")
        y = self._pos[1]
        p0, p1 = self.get_points()
        self.line = Line3D(PointSet([Point(value, y, p0.z), Point(value, y, p1.z)]))
        self._pos = (value, y)

    @property
    def y(self):
        return self._pos[1] if self._pos is not None else None

    @y.setter
    def y(self, value):
        if self._pos is None:
            raise ValueError("Well position undefined, cannot set y.")
        if not isinstance(value, (int, float)):
            raise TypeError("y must be numeric.")
        x = self._pos[0]
        p0, p1 = self.get_points()
        self.line = Line3D(PointSet([Point(x, value, p0.z), Point(x, value, p1.z)]))
        self._pos = (x, value)

    @property
    def pos(self):
        return self._pos

    @property
    def well_type(self) -> Union[WellType, str]:
        return self._well_type

    @well_type.setter
    def well_type(self, value: Union[WellType, str]):
        if not isinstance(value, (WellType, str)):
            raise TypeError("well_type must be WellType or str.")
        self._well_type = value

    @property
    def h_min(self) -> float:
        return self._h_min

    @h_min.setter
    def h_min(self, value: float):
        if not isinstance(value, (int, float)):
            raise TypeError("h_min must be numeric.")
        self._h_min = float(value)

    def __repr__(self) -> str:
        t = self._well_type.value if isinstance(self._well_type, WellType) else str(self._well_type)
        return f"<plx.structures.Well {self.describe()} type='{t}' h_min={self._h_min}>"
