from __future__ import annotations
from typing import Any, Optional, Dict, Union, List
from ..geometry import Polygon3D
from .basestructure import BaseStructure
from ..materials.platematerial import ElasticPlate
from enum import Enum
from dataclasses import dataclass, field

class RetainingWall(BaseStructure):
    """Retaining wall: surface (Polygon3D) + plate material/type."""

    def __init__(self, name: str, surface: Polygon3D, plate_type: ElasticPlate | str,
                 pos_interface: Union[bool, PositiveInterface] = True,
                 neg_interface: Union[bool, NegativeInterface] = True
                ) -> None:
        super().__init__(name)
        if not isinstance(surface, Polygon3D):
            raise TypeError("surface must be a Polygon3D instance.")
        if not isinstance(plate_type, (ElasticPlate, str)):
            raise TypeError("plate_type must be an ElasticPlate or str.")
        self._surface = surface
        self._plate_type = plate_type

        def _norm_pos(v):
            if v is True:  return PositiveInterface(name=f"{name}_PosInterface")
            if v is False or v is None: return None
            if isinstance(v, PositiveInterface): return v
            raise TypeError("pos_interface must be bool|PositiveInterface|None")

        def _norm_neg(v):
            if v is True:  return NegativeInterface(name=f"{name}_NegInterface")
            if v is False or v is None: return None
            if isinstance(v, NegativeInterface): return v
            raise TypeError("neg_interface must be bool|NegativeInterface|None")

        self._iface_pos: Optional[PositiveInterface] = _norm_pos(pos_interface)
        self._iface_neg: Optional[NegativeInterface] = _norm_neg(neg_interface)

    @property
    def surface(self) -> Polygon3D:
        return self._surface

    @property
    def plate_type(self) -> ElasticPlate | str:
        return self._plate_type

    @property
    def pos_interface(self) -> Optional[PositiveInterface]:
        return self._iface_pos

    @property
    def neg_interface(self) -> Optional[NegativeInterface]:
        return self._iface_neg
    
    @property
    def interfaces(self) -> List[Optional[PlateInterfaceBase]]:
        return [self.pos_interface, self.neg_interface]


    def __repr__(self) -> str:
        p = self._plate_type if isinstance(self._plate_type, str) else getattr(self._plate_type, "name", type(self._plate_type).__name__)
        return f"<plx.structures.RetainingWall {self.describe()} type='{p}'>"

class InterfaceMaterialMode(str, Enum):
    FROM_ADJACENT_SOIL = "From adjacent soil"
    USER_DEFINED       = "User defined" 

@dataclass
class PlateInterfaceBase(BaseStructure):
    """
    Public interface parameters (with plx_id/name, as inherited from BaseStructure)
    - material_mode: "Material mode" in the panel
    - apply_strength_reduction: "Apply strength reduction" in the panel
    - active_in_flow: "Active in flow" in the panel
    - virtual_thickness_factor: "Virtual thickness factor" in the panel
    - r_inter: R_inter (mostly at the plate level in most versions; retained here for unified processing by Mapper)
    - extra_props: Native properties freely assigned (key = PLAXIS property name)
    - parent_plate_id: Handle of the parent plate (used as a fallback for setting properties via the "Plate.Interface.Property" path)
    """
    material_mode: InterfaceMaterialMode = InterfaceMaterialMode.FROM_ADJACENT_SOIL
    apply_strength_reduction: bool = True
    active_in_flow: bool = False
    virtual_thickness_factor: float = 0.10
    r_inter: float = 0.67
    # extra_props: Dict[str, Any] = field(default_factory=dict)
    parent_plate_id: Optional[Any] = None

    def __init__(self, name: str = "Default interface", **kwargs: Any) -> None:
        super().__init__(name=name or self.__class__.__name__)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"<plx.structures.Interface>"

class PositiveInterface(PlateInterfaceBase):
    pass
    def __repr__(self) -> str:
        return f"<plx.structures.Interface  type=PositiveInterface>"

class NegativeInterface(PlateInterfaceBase):
    pass
    def __repr__(self) -> str:
        return f"<plx.structures.Interface  type=NegativeInterface>"