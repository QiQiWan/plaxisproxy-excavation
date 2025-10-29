from __future__ import annotations
from typing import Any
from ..geometry import Polygon3D
from .basestructure import BaseStructure
from ..materials.platematerial import ElasticPlate

class RetainingWall(BaseStructure):
    """Retaining wall: surface (Polygon3D) + plate material/type."""

    def __init__(self, name: str, surface: Polygon3D, plate_type: ElasticPlate | str) -> None:
        super().__init__(name)
        if not isinstance(surface, Polygon3D):
            raise TypeError("surface must be a Polygon3D instance.")
        if not isinstance(plate_type, (ElasticPlate, str)):
            raise TypeError("plate_type must be an ElasticPlate or str.")
        self._surface = surface
        self._plate_type = plate_type

    @property
    def surface(self) -> Polygon3D:
        return self._surface

    @property
    def plate_type(self) -> ElasticPlate | str:
        return self._plate_type

    def __repr__(self) -> str:
        p = self._plate_type if isinstance(self._plate_type, str) else getattr(self._plate_type, "name", type(self._plate_type).__name__)
        return f"<plx.structures.RetainingWall {self.describe()} type='{p}'>"
