from typing import List, Dict
from core.plaxisobject import PlaxisObject
from materials.soilmaterial import BaseSoilMaterial


class Borehole(PlaxisObject):
    """
    Represents a single borehole with its geological layers.

    Attributes:
        _id (uuid.UUID): A unique identifier for the borehole.
        _x (float): The x-coordinate of the borehole location.
        _y (float): The y-coordinate of the borehole location.
        _h (float): The height or elevation of the borehole (m).
        _top_list (List[float]): A list of the top elevations of each soil layer.
        _bottom_list (List[float]): A list of the bottom elevations of each soil layer.
        _soil_list (List[BaseMaterial]): A list of BaseMaterial objects representing the soil types.
    """

    def __init__(self, name: str, comment: str, x: float, y: float, h: float, 
                 top_list: List[float], bottom_list: List[float], soil_list: List[BaseSoilMaterial]) -> None:
        super().__init__(name, comment)
        self._x = x
        self._y = y
        self._h = h
        self._top_list = top_list
        self._bottom_list = bottom_list
        self._soil_list = soil_list

    # --- Properties ---
    @property
    def x(self) -> float:
        """X-coordinate (position or location)."""
        return self._x

    @property
    def y(self) -> float:
        """Y-coordinate (position or location)."""
        return self._y

    @property
    def h(self) -> float:
        """Height or elevation (m)."""
        return self._h

    @property
    def top_table(self) -> List[float]:
        """Reference to the top table data or object."""
        return self._top_list

    @property
    def bottom_table(self) -> List[float]:
        """Reference to the bottom table data or object."""
        return self._bottom_list

    @property
    def soil_table(self) -> List[BaseSoilMaterial]:
        """Reference to the soil table data or object."""
        return self._soil_list

    # --- Dunder methods ---
    def __repr__(self) -> str:
        # A unified and informative string representation
        return f"<plx.materials.Borehole id={str(self._id)[:8]} @({self.x:.2f}, {self.y:.2f}) n_layers={len(self._soil_list)}>"

class BoreholeSet(PlaxisObject):
    """
    An aggregate object for a collection of boreholes.

    This class collects all boreholes and indexes the unique soil materials
    found within them.
    """

    def __init__(self, borehole_list: List[Borehole]) -> None:
        self._borehole_list = borehole_list
        self._soils: Dict[str, BaseSoilMaterial] = {}
        self.pick_up_soils()

    def pick_up_soils(self) -> None:
        """Collects unique soil material objects from all boreholes."""
        for borehole in self._borehole_list:
            for soil in borehole.soil_table:
                self._soils[soil.id] = soil

    # --- Properties ---
    @property
    def borehole_list(self) -> List["Borehole"]:
        """List of Borehole objects contained in the set."""
        return self._borehole_list

    @property
    def soils(self) -> Dict[str, BaseSoilMaterial]:
        """Dictionary of soils collected from all boreholes (key: soil.id, value: soil object)."""
        return self._soils

    # --- Dunder methods ---
    def __repr__(self) -> str:
        # A unified and informative string representation
        return f"<plx.materials.BoreholeSet n_boreholes={len(self._borehole_list)} n_unique_soils={len(self._soils)}>"
