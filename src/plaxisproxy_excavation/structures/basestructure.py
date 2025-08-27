from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from ..core.plaxisobject import PlaxisObject

class BaseStructure(PlaxisObject):
    """Base class for Plaxis 3D structural objects (beam, anchor, embedded pile, etc)."""
    def __init__(self, name: str, comment: str = "") -> None:
        super().__init__(name=name, comment=comment)

    def describe(self) -> str:
        return f"name='{self.name}'"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "comment": getattr(self, "comment", ""),
            "plx_id": getattr(self, "plx_id", None),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseStructure":
        obj = cls(name=data.get("name", cls.__name__), comment=data.get("comment", ""))
        if "plx_id" in data:
            try:
                obj.plx_id = data["plx_id"]
            except Exception:
                pass
        return obj

    def __repr__(self) -> str:
        return f"<plx.structures.BaseStructure name='{self.name}'>"


# ---------------------------------------------------------------------
# Mixin: exactly-two-point line support + helpers
# ---------------------------------------------------------------------
from ..geometry import Line3D, Point, PointSet

class TwoPointLineMixin:
    """Validate a Line3D with exactly two points and provide get_points()."""
    _line: Line3D

    @staticmethod
    def _ensure_two_point_line(line: Line3D) -> Line3D:
        if not isinstance(line, Line3D):
            raise TypeError("line must be a Line3D instance.")
        if len(line) != 2:
            raise ValueError("line must have exactly two points.")
        return line

    @staticmethod
    def _line_from_points(p_start: Point, p_end: Point) -> Line3D:
        if not all(isinstance(p, Point) for p in (p_start, p_end)):
            raise TypeError("p_start/p_end must be Point.")
        return Line3D(PointSet([p_start, p_end]))

    @staticmethod
    def _init_line_from_args(
        *, line: Optional[Line3D], p_start: Optional[Point], p_end: Optional[Point]
    ) -> Line3D:
        if line is not None:
            return TwoPointLineMixin._ensure_two_point_line(line)
        if (p_start is None) or (p_end is None):
            raise TypeError("You must provide either 'line' or both 'p_start' and 'p_end'.")
        return TwoPointLineMixin._line_from_points(p_start, p_end)

    @property
    def line(self) -> Line3D:
        return self._line

    @line.setter
    def line(self, val: Line3D) -> None:
        self._line = self._ensure_two_point_line(val)

    def get_points(self):
        return self._line.get_points()
