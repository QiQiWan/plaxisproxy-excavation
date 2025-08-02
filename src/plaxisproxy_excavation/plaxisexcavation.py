from __future__ import annotations

"""
High-level wrapper for PLAXIS 3D, representing a complete deep excavation (foundation pit) system.

This module provides the PlaxisFoundationPit class, which serves as a central container 
to manage all components related to a deep excavation analysis. This includes geometry, 
structures, geotechnical information, construction stages, and calculation settings.

Key Features:
- Aggregates all relevant model components (e.g., retaining walls, anchors, piles, loads) 
  into a single, unified object.
- Provides convenient methods for adding and managing these components.
- Performs validation checks via the `finalize()` method before sending the model to 
  the PLAXIS calculation kernel.

Example Usage:
>>> # 1. Initialize a foundation pit object
>>> pit_footprint = Polygon3D(...)
>>> my_pit = PlaxisFoundationPit(name="Main_Excavation", footprint=pit_footprint, depth=15.0)

>>> # 2. Add components
>>> wall = RetainingWall(...)
>>> my_pit.add_retaining_wall(wall)
>>> my_pit.set_mesh(Mesh(mesh_coarseness=MeshCoarseness.Medium))
>>> my_pit.add_stage(ConstructionStage(name="Excavate_to_5m"))

>>> # 3. Finalize and validate the model
>>> my_pit.finalize()

>>> # 4. Send to PLAXIS using a PlaxisSession
>>> with PlaxisSession("localhost", 10000, "password") as sess:
...     FoundationPitMapper(sess).create(my_pit)

"""

from typing import List, Optional, Dict, Any
import uuid

# --- Import necessary classes from other project modules ---
# Geometry and Core Objects
from .geometry import Polygon3D, rings_close_to_footprint
from .structures.basestructure import BaseStructure

# Model Components
from .components.mesh import Mesh
from .components.watertable import WaterLevelTable
from .components.projectinformation import ProjectInformation
from .components.phase import ConstructionStage

# Structural Elements
from .structures.retainingwall import RetainingWall
from .structures.anchor import Anchor
from .structures.embeddedpile import EmbeddedPile
from .structures.well import Well
from .structures.load import _BaseLoad

# Geotechnical
from .borehole import BoreholeSet


class PlaxisFoundationPit(BaseStructure):
    """
    A high-level wrapper representing a complete deep excavation engineering system.

    This class aggregates all components required to define a foundation pit model 
    and provides methods for managing these components and performing validation 
    before finalization.

    Attributes:
        footprint (Polygon3D): A 2D/3D polygon defining the excavation perimeter.
        depth (float): The total excavation depth of the pit (in meters).
        project_info (Optional[ProjectInformation]): Metadata for the model.
        retaining_walls (List[RetainingWall]): A list of retaining walls.
        anchors (List[Anchor]): A list of anchors.
        embedded_piles (List[EmbeddedPile]): A list of embedded piles.
        wells (List[Well]): A list of dewatering wells.
        loads (List[_BaseLoad]): A list of loads applied to the model.
        borehole_set (Optional[BoreholeSet]): A set of boreholes defining the soil stratigraphy.
        stages (List[ConstructionStage]): The sequence of construction stages.
        mesh (Optional[Mesh]): The mesh settings for the model.
    """

    def __init__(
        self,
        name: str,
        footprint: Polygon3D,
        depth: float,
        comment: str = "Deep Excavation Model"
    ) -> None:
        """
        Initializes the PlaxisFoundationPit instance.

        Args:
            name (str): The name of the foundation pit model.
            footprint (Polygon3D): The exterior boundary defining the excavation area.
            depth (float): The total excavation depth of the pit (in meters).
            comment (str, optional): A comment or description for the model.
        """
        super().__init__(name=name, comment=comment)

        if not isinstance(footprint, Polygon3D):
            raise TypeError("The footprint must be an instance of Polygon3D.")
        if not depth > 0:
            raise ValueError("The excavation depth must be a positive number.")

        self._footprint = footprint
        self._depth = depth
        self._project_info: Optional[ProjectInformation] = None

        # --- Component Containers ---
        self._retaining_walls: List[RetainingWall] = []
        self._anchors: List[Anchor] = []
        self._embedded_piles: List[EmbeddedPile] = []
        self._wells: List[Well] = []
        self._loads: List[_BaseLoad] = []
        self._borehole_set: Optional[BoreholeSet] = None
        self._stages: List[ConstructionStage] = []
        self._mesh: Optional[Mesh] = None

    # --- Public Properties ---
    @property
    def footprint(self) -> Polygon3D:
        """Gets the Polygon3D that defines the excavation perimeter."""
        return self._footprint

    @property
    def depth(self) -> float:
        """Gets the total excavation depth of the pit (in meters)."""
        return self._depth

    @property
    def project_info(self) -> Optional[ProjectInformation]:
        """Gets the project's metadata information."""
        return self._project_info

    @property
    def retaining_walls(self) -> List[RetainingWall]:
        """Gets the list of retaining walls."""
        return self._retaining_walls

    @property
    def anchors(self) -> List[Anchor]:
        """Gets the list of anchors."""
        return self._anchors

    @property
    def embedded_piles(self) -> List[EmbeddedPile]:
        """Gets the list of embedded piles."""
        return self._embedded_piles
        
    @property
    def wells(self) -> List[Well]:
        """Gets the list of dewatering wells."""
        return self._wells

    @property
    def loads(self) -> List[_BaseLoad]:
        """Gets the list of loads applied to the model."""
        return self._loads

    @property
    def borehole_set(self) -> Optional[BoreholeSet]:
        """Gets the set of boreholes that define the soil stratigraphy."""
        return self._borehole_set

    @property
    def stages(self) -> List[ConstructionStage]:
        """Gets the sequence of construction stages."""
        return self._stages

    @property
    def mesh(self) -> Optional[Mesh]:
        """Gets the mesh settings."""
        return self._mesh

    # --- Component Management Methods ---
    def set_project_info(self, info: ProjectInformation) -> None:
        """Sets the project information."""
        if not isinstance(info, ProjectInformation):
            raise TypeError("The provided info must be an instance of ProjectInformation.")
        self._project_info = info

    def add_retaining_wall(self, wall: RetainingWall) -> None:
        """Adds a retaining wall to the model."""
        if not isinstance(wall, RetainingWall):
            raise TypeError("The provided object must be an instance of RetainingWall.")
        self._retaining_walls.append(wall)

    def add_anchor(self, anchor: Anchor) -> None:
        """Adds an anchor to the model."""
        if not isinstance(anchor, Anchor):
            raise TypeError("The provided object must be an instance of Anchor.")
        self._anchors.append(anchor)

    def add_embedded_pile(self, pile: EmbeddedPile) -> None:
        """Adds an embedded pile to the model."""
        if not isinstance(pile, EmbeddedPile):
            raise TypeError("The provided object must be an instance of EmbeddedPile.")
        self._embedded_piles.append(pile)

    def add_well(self, well: Well) -> None:
        """Adds a dewatering well to the model."""
        if not isinstance(well, Well):
            raise TypeError("The provided object must be an instance of Well.")
        self._wells.append(well)
        
    def add_load(self, load: _BaseLoad) -> None:
        """Adds a load to the model."""
        if not isinstance(load, _BaseLoad):
            raise TypeError("The provided object must be an instance of _BaseLoad or its subclasses.")
        self._loads.append(load)

    def set_boreholes(self, borehole_set: BoreholeSet) -> None:
        """Sets the borehole data for the model."""
        if not isinstance(borehole_set, BoreholeSet):
            raise TypeError("The provided object must be an instance of BoreholeSet.")
        self._borehole_set = borehole_set

    def add_stage(self, stage: ConstructionStage) -> None:
        """Adds a construction stage to the model."""
        if not isinstance(stage, ConstructionStage):
            raise TypeError("The provided object must be an instance of ConstructionStage.")
        self._stages.append(stage)
        
    def set_mesh(self, mesh: Mesh) -> None:
        """Sets the mesh parameters for the model."""
        if not isinstance(mesh, Mesh):
            raise TypeError("The provided object must be an instance of Mesh.")
        self._mesh = mesh

    # --- Model Finalization and Validation ---
    def finalize(self, ring_closure_tol: float = 1e-2) -> None:
        """
        Performs final validation checks before committing the session to PLAXIS.

        This method validates:
        1. That project information exists.
        2. That the top edges of the retaining walls form a closed ring that is 
           sufficiently close to the excavation footprint.

        Args:
            ring_closure_tol (float, optional): The area tolerance (in m²) for the 
                                                retaining wall ring closure check. Defaults to 1e-2.

        Raises:
            RuntimeError: If a validation check fails.
        """
        # 1. Validate that project information exists
        if not self._project_info:
            raise RuntimeError("Validation Failed: Model is missing ProjectInformation. Please add it using set_project_info().")

        # 2. Validate that retaining walls form a closed ring
        if not self._retaining_walls:
            print("Warning: No retaining walls in the model. Skipping ring closure check.")
            return
        
        # Extract the wall surfaces
        wall_surfaces = [wall.surface for wall in self._retaining_walls]
        
        try:
            # `rings_close_to_footprint` depends on the `shapely` library
            if not rings_close_to_footprint(wall_surfaces, self._footprint, tol=ring_closure_tol):
                raise RuntimeError(
                    f"Validation Failed: The retaining walls do not form a closed ring around the footprint (tolerance={ring_closure_tol} m²)."
                )
        except RuntimeError as e:
            # Catch exception if shapely is not installed and re-raise with a more specific message
            if "Shapely is not installed" in str(e) or "Shapely is required" in str(e):
                raise RuntimeError(
                    "Cannot perform retaining wall closure check because the 'shapely' library is not installed. "
                    "Please run 'pip install shapely' or skip this check."
                ) from e
            raise e # Re-raise other runtime errors

        print("Model validation successful.")

    def __repr__(self) -> str:
        """Provides an informative string representation."""
        return (
            f"<PlaxisFoundationPit(name='{self.name}', depth={self.depth}m, "
            f"walls={len(self._retaining_walls)}, stages={len(self._stages)})>"
        )