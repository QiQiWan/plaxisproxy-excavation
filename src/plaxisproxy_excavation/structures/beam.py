from __future__ import annotations
from typing import Optional, Union
from ..materials.beammaterial import BeamType, ElasticBeam, ElastoplasticBeam
from ..geometry import Line3D, Point
from .basestructure import BaseStructure, TwoPointLineMixin

class Beam(BaseStructure, TwoPointLineMixin):
    """Beam: two-point Line3D + beam material/type."""

    def __init__(
        self,
        name: str,
        line: Optional[Line3D] = None,
        *,
        p_start: Optional[Point] = None,
        p_end: Optional[Point] = None,
        beam_type: Union[BeamType, ElasticBeam, ElastoplasticBeam, str] = BeamType.Elastic
    ) -> None:
        super().__init__(name)
        self._line = self._init_line_from_args(line=line, p_start=p_start, p_end=p_end)
        if not isinstance(beam_type, (BeamType, ElasticBeam, ElastoplasticBeam, str)):
            raise TypeError("beam_type must be BeamType/BeamMaterial or str.")
        self._beam_type = beam_type

    @property
    def beam_type(self):
        return self._beam_type

    def __repr__(self) -> str:
        t = self._beam_type if isinstance(self._beam_type, str) else getattr(self._beam_type, "name", type(self._beam_type).__name__)
        return f"<plx.structures.Beam {self.describe()} type='{t}'>"
