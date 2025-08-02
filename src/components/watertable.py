from ..geometry import Point
import uuid
from typing import List, Optional

class WaterLevel(Point):
    """
    3D Water Level Point for Plaxis 3D seepage analysis.
    Inherits coordinates from Point, adds label and time for time-dependent or spatially-varying water table.
    """
    def __init__(self, x: float, y: float, z: float,
                 label: Optional[str] = None,
                 time: Optional[float] = None):
        """
        Args:
            x, y, z (float): Coordinates of water level point.
            label (str, optional): Description or usage (e.g. 'Initial', 'Boundary').
            time (float, optional): Time for time-dependent water level (days).
        """
        super().__init__(x, y, z)
        self._id = uuid.uuid4()
        self._plx_id = None
        self._label = label
        self._time = time

    @property
    def id(self):
        return self._id

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value: float):
        self._time = value

    @property
    def elevation(self):
        """Alias for z, compatible with WaterLevelTable."""
        return self._z

    def __repr__(self):
        tag = f"{self._label}: " if self._label else ""
        if self._time is not None:
            tag += f"t={self._time:.2f}d, "
        return f"<plx.components.WaterLevelPoint {tag}({self.x:.3f}, {self.y:.3f}, {self.z:.3f})>"

class WaterLevelTable:
    """
    Water level table for Plaxis 3D, supporting time-dependent or piecewise water level definition.
    Can be used for boundary conditions or dynamic water table.
    """
    def __init__(self, levels: Optional[List[WaterLevel]] = None, label: Optional[str]=None):
        self._levels = levels if levels is not None else []
        self._label = label

    def add_level(self, level: WaterLevel):
        self._levels.append(level)

    @property
    def levels(self) -> List[WaterLevel]:
        return self._levels

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    def as_time_series(self):
        """
        Return time series for use in time-dependent boundary condition.
        Returns:
            List[Tuple[float, float]]: [(time, elevation), ...]
        """
        return sorted([(lvl.time, lvl.elevation) for lvl in self._levels if lvl.time is not None])

    def as_elevation_list(self):
        """
        Return elevation list (for spatial, non-time-dependent tables).
        """
        return [lvl.elevation for lvl in self._levels]

    def __repr__(self):
        if self._label:
            return f"<plx.components.water_level_table {self._label}, {len(self._levels)} levels>"
        else:
            return f"<plx.components.water_level_table, {len(self._levels)} levels>"
