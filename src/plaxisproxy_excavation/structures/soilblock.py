from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from types import SimpleNamespace
from ..core.plaxisobject import PlaxisObject
from ..materials.soilmaterial import BaseSoilMaterial
from .basestructure import BaseStructure
from ..geometry import Polygon3D, Polyhedron, Volume

class SoilBlock(BaseStructure):
    """Soil volume: geometry + soil material, with consistent plx_id access."""

    _SERIAL_VERSION: int = 1

    def __init__(
        self,
        name: str,
        comment: str = "",
        material: Optional[BaseSoilMaterial] = None,
        geometry: Optional[Polyhedron | Polygon3D | Volume | List[Tuple[float, float, float]]] = None,
    ):
        super().__init__(name, comment)
        self._material = self._coerce_material(material)
        self._geometry = geometry
        # Unified with Mapper: Store the handle of the volume in _plx_volume_id in the project, and access it uniformly through plx_id.
        self._plx_volume_id: Optional[Any] = None  # Any: Handle object or ID

    # ------------ The "plx_id" and "_plx_volume_id" are interchangeable. ----------------
    @property
    def plx_id(self):
        return self._plx_volume_id

    @plx_id.setter
    def plx_id(self, value):
        self._plx_volume_id = value

    # ----------------- Properties & Methods -----------------
    @property
    def material(self) -> Optional[BaseSoilMaterial]:
        return self._material

    def set_material(self, mat: BaseSoilMaterial) -> None:
        self._material = mat

    @property
    def geometry(self):
        return self._geometry

    def set_geometry(self, geom: Polyhedron | Polygon3D | List[Tuple[float, float, float]]):
        self._geometry = geom

    @classmethod
    def _from_dict_core(cls, data: Dict[str, Any]) -> "SoilBlock":
        name = data.get("name", "SoilBlock")
        comment = data.get("comment", "")
        mat = data.get("material")
        geom = data.get("geometry")
        return cls(name=name, material=mat, geometry=geom, comment=comment)

    @staticmethod
    def _coerce_material(material: Any) -> Optional[Any]:
        if material is None:
            return None
        if isinstance(material, str):
            return SimpleNamespace(name=material)
        if isinstance(material, dict):
            nm = material.get("name") or material.get("mat") or "Unknown"
            return SimpleNamespace(name=nm)
        if hasattr(material, "name"):
            return material
        raise TypeError("material must have a 'name' attribute, or be str/dict/None.")

    def __repr__(self) -> str:
        mat_name = getattr(self._material, "name", "None") if self._material is not None else "None"
        geom_state = "set" if self._geometry is not None else "None"
        sync = "unsynced" if self._plx_volume_id is None else "synced"
        return f"<plx.structures.SoilBlock name='{self.name}', mat='{mat_name}', geom={geom_state} | {sync}>"
