import uuid
from typing import Optional, Any
from ..geometry import Point

class CurvePoint(Point):
    """
    Base abstract class for a point on a curve in Plaxis 3D post-processing.
    Can be specialized as NodePoint or StressPoint.
    """

    def __init__(self, x: float, y: float, z: float, label: Optional[str] = None,
                 datafrom = None, pre_calc: bool = False):
        """
        Initializes a CurvePoint instance.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            z (float): The z-coordinate.
            label (Optional[str], optional): An optional label for the point. Defaults to None.
            datafrom (Any, optional): The entity from which the curvepoint was created. Defaults to None.
            pre_calc (bool, optional): Flag indicating if the data is pre-calculated. Defaults to False.
        """
        super().__init__(x, y, z)
        self._id = uuid.uuid4()
        self._plx_id = None
        self._label = label
        self._datafrom = datafrom
        self._pre_calc = pre_calc

    @property
    def id(self) -> uuid.UUID:
        """Unique identifier for the curve point."""
        return self._id

    @property
    def plx_id(self) -> Optional[str]:
        """Unique identifier in Plaxis software."""
        return self._plx_id
    
    @plx_id.setter
    def plx_id(self, value: str):
        self._plx_id = value

    @property
    def label(self) -> Optional[str]:
        """Optional label or description for the point."""
        return self._label

    @property
    def datafrom(self) -> Any:
        """Entity from which the curvepoint was created."""
        return self._datafrom
    
    @datafrom.setter
    def datafrom(self, value: Any):
        self._datafrom = value

    @property
    def pre_calc(self) -> bool:
        """Flag indicating if the data is pre-calculated."""
        return self._pre_calc

    def __repr__(self) -> str:
        """Provides a unified and informative string representation."""
        label_info = f"'{self._label}' " if self._label else ""
        return f"<plx.curve.CurvePoint {label_info}@({self.x:.3f}, {self.y:.3f}, {self.z:.3f})>"

class NodePoint(CurvePoint):
    """
    Curve point that refers to a finite element mesh node (for extracting
    displacement, reaction, etc).
    """
    def __init__(self, x: float, y: float, z: float, label: Optional[str] = None, node_id: Optional[int] = None):
        """
        Initializes a NodePoint instance.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            z (float): The z-coordinate.
            label (Optional[str], optional): An optional label for the point.
            node_id (Optional[int], optional): The mesh node number.
        """
        super().__init__(x, y, z, label)
        self._node_id = node_id 

    @property
    def node_id(self) -> Optional[int]:
        """Mesh node number (if available, from Plaxis)."""
        return self._node_id

    def __repr__(self) -> str:
        """Provides a unified and informative string representation."""
        label_info = f"'{self._label}' " if self._label else ""
        node_info = f"node_id={self._node_id}" if self._node_id is not None else "no_id"
        return f"<plx.curve.NodePoint {label_info}{node_info} @({self.x:.3f}, {self.y:.3f}, {self.z:.3f})>"

class StressPoint(CurvePoint):
    """
    Curve point that refers to a finite element stress (integration) point
    (for extracting stress/strain).
    """
    def __init__(self, x: float, y: float, z: float, label: Optional[str] = None, element_id: Optional[int] = None, local_index: Optional[int] = None):
        """
        Initializes a StressPoint instance.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            z (float): The z-coordinate.
            label (Optional[str], optional): An optional label for the point.
            element_id (Optional[int], optional): The associated FE element number.
            local_index (Optional[int], optional): The local index of the stress point within the element.
        """
        super().__init__(x, y, z, label)
        self._element_id = element_id
        self._local_index = local_index

    @property
    def element_id(self) -> Optional[int]:
        """Element number where the stress point is located."""
        return self._element_id

    @property
    def local_index(self) -> Optional[int]:
        """Local index of the stress point within the element."""
        return self._local_index

    def __repr__(self) -> str:
        """Provides a unified and informative string representation."""
        label_info = f"'{self._label}' " if self._label else ""
        elem_info = f"elem_id={self._element_id}" if self._element_id is not None else "no_elem_id"
        idx_info = f"idx={self._local_index}" if self._local_index is not None else ""
        return f"<plx.curve.StressPoint {label_info}{elem_info} {idx_info}@({self.x:.3f}, {self.y:.3f}, {self.z:.3f})>"
