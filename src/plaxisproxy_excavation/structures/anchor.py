from __future__ import annotations
from typing import Optional, Union
from ..materials.anchormaterial import (
    AnchorType, ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor
)
from ..geometry import Line3D, Point
from .basestructure import BaseStructure, TwoPointLineMixin

class Anchor(BaseStructure, TwoPointLineMixin):
    """Anchor (tieback), defined by a 2-point line and an anchor material/type."""

    def __init__(
        self,
        name: str,
        # 兼容两种写法：line 或 p_start/p_end
        line: Optional[Line3D] = None,
        *,
        p_start: Optional[Point] = None,
        p_end: Optional[Point] = None,
        anchor_type: Union[AnchorType, ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor, str] = "Elastic"
    ) -> None:
        super().__init__(name)
        self._line = self._init_line_from_args(line=line, p_start=p_start, p_end=p_end)
        if not isinstance(anchor_type, (AnchorType, ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor, str)):
            raise TypeError("anchor_type must be AnchorType/AnchorMaterial/str.")
        self._anchor_type = anchor_type

    @property
    def anchor_type(self):
        return self._anchor_type

    def __repr__(self) -> str:
        t = self._anchor_type if isinstance(self._anchor_type, str) else type(self._anchor_type).__name__
        return f"<plx.structures.Anchor {self.describe()} type='{t}'>"
