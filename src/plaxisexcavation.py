from __future__ import annotations

"""High‑level wrapper representing a deep‑excavation (foundation pit) system.

Revision 2025‑07‑30
-------------------
*   Added ``ProjectInformation`` linkage so each pit is aware of its parent
    metadata block.
*   Updated import paths to reflect new package layout (components/, structures/).
*   ``finalize()`` validates ring closure **and** presence of project info.
"""

from typing import List, Optional, Dict, Any
import uuid

# --- updated import map -----------------------------------------------------
from geometry import * 
from components.mesh import Mesh
from components.watertable import WaterLevelTable
from components.projectinformation import ProjectInformation
from structures.retainingwall import RetainingWall
from structures.anchor import Anchor
from structures.embeddedpile import EmbeddedPile
from structures.well import Well
from borehole import BoreholeSet
from components.phase import ConstructionStage
from structures.basestructure import BaseStructure


class PlaxisFoundationPit(BaseStructure):
    """Aggregate object for deep‑excavation projects."""

    # ------------------------------------------------------------------
    # Construction / basic properties
    # ------------------------------------------------------------------
    def __init__(
        self,
        name: str,
        footprint: "Polygon3D",
        excavation_depth: float,
        mesh: Optional[Mesh] = None,
        water_table: Optional[WaterLevelTable] = None,
        project_info: Optional[ProjectInformation] = None,
    ) -> None:

        super().__init__()
        self._id: uuid.UUID = uuid.uuid4()
        self._name = name
        self._footprint = footprint
        self._depth = float(excavation_depth)
        self._mesh = mesh
        self._project_info = project_info

        # component containers
        self._walls: List[RetainingWall] = []
        self._anchors: List[Anchor] = []
        self._piles: List[EmbeddedPile] = []
        self._wells: List[Well] = []
        self._water_tables: List[WaterLevelTable] = []
        self._stages: List[ConstructionStage] = []
        self._boreholes: Optional[BoreholeSet] = None

    # ------------------------------------------------------------------
    # Public API - add / set
    # ------------------------------------------------------------------
    def set_project_info(self, info: ProjectInformation):
        """Attach a *ProjectInformation* block (overwrites any previous one)."""
        self._project_info = info

    def add_wall(self, wall: RetainingWall):
        self._walls.append(wall)

    def add_anchor(self, anchor: Anchor):
        self._anchors.append(anchor)

    def add_pile(self, pile: EmbeddedPile):
        self._piles.append(pile)

    def add_well(self, well: Well):
        self._wells.append(well)

    def add_stage(self, stage: ConstructionStage):
        self._stages.append(stage)

    def set_mesh(self, mesh: Mesh):
        self._mesh = mesh

    def set_boreholes(self, borehole_set: BoreholeSet):
        self._boreholes = borehole_set

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_surface(wall: RetainingWall) -> Polygon3D:
        """Return the single *Polygon3D* surface representing the wall."""
        if wall.surface:
            return wall.surface
        raise AttributeError("RetainingWall must expose 'surface'.")

    def validate_ring_closure(self, tol: float = 1e-3) -> None:
        """Raise *ValueError* if merged wall surfaces fail to enclose footprint."""
        if not self._walls:
            raise ValueError("No RetainingWall objects added - cannot validate closure.")

        surfaces = [self._get_surface(w) for w in self._walls]
        if not rings_close_to_footprint(surfaces, self._footprint, tol):
            raise ValueError(
                "Retaining‑wall surfaces do not enclose the footprint - check gaps or overlaps.")

    def finalize(self):
        self.validate_ring_closure()
        if self._project_info is None:
            raise ValueError("ProjectInformation not set - every pit must belong to a project.")
        # TODO: mesh/stage checks

    # ------------------------------------------------------------------
    # Convenience metrics
    # ------------------------------------------------------------------
    def plan_area(self) -> float:
        return self._footprint.area()

    def excavated_volume(self) -> float:
        return self.plan_area() * self._depth

    # ------------------------------------------------------------------
    # Summary / serialization
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "depth": self._depth,
            "area": self.plan_area(),
            "volume": self.excavated_volume(),
            "walls": len(self._walls),
            "anchors": len(self._anchors),
            "piles": len(self._piles),
            "wells": len(self._wells),
            "stages": len(self._stages),
            "mesh": repr(self._mesh) if self._mesh else None,
            "water_table": repr(self._water_table) if self._water_table else None,
            "project": self._project_info.title if self._project_info else None,
        }

    # ------------------------------------------------------------------
    # Read‑only properties
    # ------------------------------------------------------------------
    @property
    def project_info(self):
        return self._project_info

    @property
    def name(self):
        return self._name

    @property
    def footprint(self):
        return self._footprint

    @property
    def depth(self):
        return self._depth

    @property
    def mesh(self):
        return self._mesh

    @property
    def water_table(self):
        return self._water_table

    @property
    def retaining_walls(self):
        return self._walls

    @property
    def anchors(self):
        return self._anchors

    @property
    def embedded_piles(self):
        return self._piles

    @property
    def wells(self):
        return self._wells

    @property
    def stages(self):
        return self._stages

    @property
    def boreholes(self):
        return self._boreholes

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self):
        proj = f" | proj={self._project_info.title}" if self._project_info else ""
        return (
            f"<plx.FoundationPit {self._name}: area={self.plan_area():.2f} m², "
            f"depth={self._depth:.2f} m, walls={len(self._walls)}{proj}>" )
