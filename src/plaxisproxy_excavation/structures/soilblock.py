from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any

from ..core.plaxisobject import PlaxisObject         # Base class for Plaxis objects
from ..materials.soilmaterial import BaseSoilMaterial  # Represents the actual soil material class (assumed to exist)
from ..geometry import Polygon3D, Polyhedron           # Your geometry module

class SoilBlock(PlaxisObject):
    """Represents a soil volume, associating a geometry with a soil material."""

    # Version number for serialization purposes.
    _SERIAL_VERSION: int = 1

    # ----------------- Constructor -----------------
    def __init__(
        self,
        name: str,
        comment: str = "",
        material: Optional[BaseSoilMaterial] = None,
        geometry: Optional[Polyhedron | Polygon3D | List[Tuple[float, float, float]]] = None,
    ):
        """
        Initializes a SoilBlock object.

        Args:
            name (str): The name of the soil block.
            comment (str, optional): An optional comment or description.
            material (Optional[BaseSoilMaterial], optional): The soil material to assign.
            geometry (Optional[...], optional): The geometric definition of the soil volume.
        """
        super().__init__(name, comment)
        self._material = material
        self._geometry = geometry
        # The unique ID of the corresponding volume in the Plaxis software, synced later.
        self._plx_volume_id: Optional[str] = None

    # ----------------- Properties & Methods -----------------
    @property
    def material(self) -> Optional[BaseSoilMaterial]:
        """Gets the assigned soil material object."""
        return self._material

    def set_material(self, mat: BaseSoilMaterial) -> None:
        """Sets or updates the soil material for this block."""
        self._material = mat

    @property
    def geometry(self):
        """Gets the geometric representation of the soil block."""
        return self._geometry

    def set_geometry(self, geom: Polyhedron | Polygon3D | List[Tuple[float, float, float]]):
        """Sets or updates the geometry for this block."""
        self._geometry = geom

    @classmethod
    def _from_dict_core(cls, data: Dict[str, Any]) -> "SoilBlock":
        """Core helper method for deserialization from a dictionary (for internal use)."""
        name = data["name"]
        comment = data.get("comment", "")
        mat = data.get("material")
        geom = data.get("geometry")

        return cls(name=name, material=mat, geometry=geom, comment=comment)

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the object."""
        mat = self._material.name if self._material else "None"
        plx_id = self._plx_volume_id or "unsynced"
        return (f"<plx.materials.SoilBlock name='{self._name}', "
                f"mat='{mat}', geom={'set' if self._geometry else 'None'} | {plx_id}>")