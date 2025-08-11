from typing import Any
from ..geometry import Polygon3D
from .basestructure import BaseStructure
from ..materials.platematerial import ElasticPlate

class RetainingWall(BaseStructure):
    """
    Retaining wall object for Plaxis 3D, defined by a 3D polygonal surface
    and a plate material/type.
    """
    __slots__ = ("_surface", "_plate_type")

    def __init__(self, name: str, surface: Polygon3D, plate_type: ElasticPlate) -> None:
        """
        Initializes the retaining wall with a surface and plate material/type.

        Args:
            name (str): The name of the wall.
            surface (Polygon3D): The 3D polygonal surface geometry of the wall.
            plate_type (PlateMaterial): The plate material or type assigned to the wall.
        """
        super().__init__(name)
        if not isinstance(surface, Polygon3D):
            raise TypeError("Surface must be a Polygon3D instance.")
        if not isinstance(plate_type, (ElasticPlate, str)):
            raise TypeError("plate_type must be an ElasticPlate or str.")
        self._surface = surface
        self._plate_type = plate_type

    @property
    def surface(self) -> Polygon3D:
        """3D polygonal surface geometry of the wall."""
        return self._surface

    @property
    def plate_type(self) -> ElasticPlate:
        """Plate type or material assigned to the wall."""
        return self._plate_type

    def __repr__(self) -> str:
        p = self._plate_type if isinstance(self._plate_type, str) else getattr(self._plate_type, 'name', type(self._plate_type).__name__)
        return f"<plx.structures.RetainingWall name='{self.name}' type='{p}'>"
