from typing import List, Dict
from .core.plaxisobject import PlaxisObject
from .materials.soilmaterial import BaseSoilMaterial

class Borehole(PlaxisObject):
    """
    Represents a single borehole with its geological layers.
    ...
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

    def to_dict(self) -> Dict:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "comment": self.comment,
            "x": self._x,
            "y": self._y,
            "h": self._h,
            "top_list": list(self._top_list),
            "bottom_list": list(self._bottom_list),
            # The elements in the "soil_table" could be objects or serialized dictionaries; all of them are passed up to the upper-level for general serialization.
            "soil_table": [
                (s.to_dict() if hasattr(s, "to_dict") else s)
                for s in self._soil_list
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Borehole":
        # Restore soils: If "from_dict" is present, restore the object; otherwise, keep it as a dictionary (the subsequent "pick_up_soils" function can also recognize the 'id' in the dictionary)
        soils_in = data.get("soil_table", [])
        restored_soils = []
        for s in soils_in:
            if isinstance(s, dict):
                # 优先尝试 BaseSoilMaterial.from_dict
                try:
                    restored_soils.append(BaseSoilMaterial.from_dict(s)) 
                except Exception:
                    restored_soils.append(s) 
            else:
                restored_soils.append(s)

        return cls(
            name=data.get("name", ""),
            comment=data.get("comment", ""),
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            h=float(data.get("h", 0.0)),
            top_list=list(data.get("top_list", [])),
            bottom_list=list(data.get("bottom_list", [])),
            soil_list=restored_soils,
        )

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

    def __repr__(self) -> str:
        n_layers = len(self.soil_table) if hasattr(self, "soil_table") else 0
        return (
            f"<plx.Borehole(id={self.id}, @({self.x:.2f}, {self.y:.2f}), "
            f"h={self.h:.2f}, n_layers={n_layers})>"
        )


class BoreholeSet(PlaxisObject):
    """
    An aggregate object for a collection of boreholes.
    This class collects all boreholes and indexes the unique soil materials found within them.
    """
    def __init__(self, borehole_list: List[Borehole], name="boreholeSet", comment="") -> None:
        super().__init__(name, comment)
        self._borehole_list = borehole_list
        self._soils: Dict[str, BaseSoilMaterial] = {}
        self.pick_up_soils()

    def pick_up_soils(self) -> None:
        """Collects unique soil material objects from all boreholes."""
        for borehole in self._borehole_list:
            for soil in borehole.soil_table:
                if hasattr(soil, "id"):
                    self._soils[soil.id] = soil
                elif isinstance(soil, dict) and "id" in soil:
                    self._soils[soil["id"]] = soil

    def to_dict(self) -> Dict:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "comment": self.comment,
            # Note: Do not export self._soils (as it is a derived attribute and would result in serializing unnecessary parameters)
            "borehole_list": [
                (bh.to_dict() if hasattr(bh, "to_dict") else bh)
                for bh in self._borehole_list
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BoreholeSet":
        # Discard any incoming 'soils' field and avoid passing it to __init__
        boreholes_in = data.get("borehole_list", [])
        restored_bhs: List[Borehole] = []
        for b in boreholes_in:
            if isinstance(b, dict):
                # 优先使用 Borehole.from_dict
                if b.get("__type__", "").endswith(".Borehole"):
                    restored_bhs.append(Borehole.from_dict(b))
                else:
                    # Fallback: The key names of the dictionary should be the same as those in Borehole.__init__
                    restored_bhs.append(Borehole.from_dict(b))
            else:
                restored_bhs.append(b)

        inst = cls(
            borehole_list=restored_bhs,
            name=data.get("name", "boreholeSet"),
            comment=data.get("comment", ""),
        )
        # Recollect the soils (in both object and dict forms)
        inst._soils.clear()
        inst.pick_up_soils()
        return inst

    @property
    def borehole_list(self) -> List["Borehole"]:
        """List of Borehole objects contained in the set."""
        return self._borehole_list

    @property
    def soils(self) -> Dict[str, BaseSoilMaterial]:
        """Dictionary of soils collected from all boreholes (key: soil.id, value: soil object)."""
        return self._soils

    def __repr__(self) -> str:
        unique_soils = len(self.soils)
        return (
            f"<plx.BoreholeSet(id={self._id}, "
            f"n_boreholes={len(self.borehole_list)}, n_unique_soils={unique_soils})>"
        )