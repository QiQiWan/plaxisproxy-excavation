from __future__ import annotations
from typing import Optional, Union, List, Any
from ..geometry import Line3D, Point
from .basestructure import BaseStructure, TwoPointLineMixin
from ..materials.pilematerial import ElasticPile, ElastoplasticPile

class EmbeddedPile(BaseStructure, TwoPointLineMixin):
    """Embedded pile: two-point Line3D + pile material/type."""

    def __init__(
        self,
        name: str,
        line: Optional[Line3D] = None,
        *,
        p_start: Optional[Point] = None,
        p_end: Optional[Point] = None,
        pile_type: Union[ElasticPile, ElastoplasticPile, str] = "Elastic"
    ) -> None:
        super().__init__(name)
        self._line = self._init_line_from_args(line=line, p_start=p_start, p_end=p_end)
        if not isinstance(pile_type, (ElasticPile, ElastoplasticPile, str)):
            raise TypeError("pile_type must be ElasticPile/ElastoplasticPile or str.")
        self._pile_type = pile_type

    @property
    def pile_type(self):
        return self._pile_type

    def get_points(self) -> List[Any]:
        return self._line.get_points()

    def length(self) -> float:
        return self._line.length

    def __repr__(self) -> str:
        t = self._pile_type if isinstance(self._pile_type, str) else type(self._pile_type).__name__
        return f"<plx.structures.EmbeddedPile {self.describe()} type='{t}'>"
