from enum import Enum
from typing import List, Dict, Any, Optional

class MeshCoarseness(Enum):
    HighRoughness = 0.1     # Very coarse mesh
    Roughness = 0.075       # Coarse mesh
    Medium = 0.05           # Medium mesh (default)
    Refine = 0.035          # Refined mesh
    HighRefine = 0.025      # Highly refined mesh

class Mesh:
    """
    Mesh generation settings object for Plaxis 3D, including coarseness,
    refinement and swept mesh options.
    """

    def __init__(self,
                 mesh_coarseness: MeshCoarseness,
                 enhanced_refine: bool = True,
                 emr_global_scale: float = 1.2,
                 emr_min_elem: float = 5e-3,
                 swept_mesh: bool = True):
        """
        Initialize the mesh settings object.

        Args:
            mesh_coarseness (MeshCoarseness): The target mesh coarseness (enum value).
            enhanced_refine (bool, optional): Whether to enable enhanced mesh refinement. Default True.
            emr_global_scale (float, optional): Global mesh scaling factor for enhanced mesh refinement (EMR). Default 1.2.
            emr_min_elem (float, optional): Minimum mesh element size allowed (m). Default 5e-3.
            swept_mesh (bool, optional): Whether to enable swept mesh generation (if applicable). Default True.
        """
        self._mesh_coarseness = mesh_coarseness
        self._enhanced_refine = enhanced_refine
        self._emr_global_scale = emr_global_scale
        self._emr_min_elem = emr_min_elem
        self._swept_mesh = swept_mesh

    @property
    def mesh_coarseness(self) -> MeshCoarseness:
        """Mesh coarseness setting (enum)."""
        return self._mesh_coarseness

    @mesh_coarseness.setter
    def mesh_coarseness(self, value: MeshCoarseness):
        self._mesh_coarseness = value

    @property
    def enhanced_refine(self) -> bool:
        """Enhanced mesh refinement flag."""
        return self._enhanced_refine

    @enhanced_refine.setter
    def enhanced_refine(self, value: bool):
        self._enhanced_refine = value

    @property
    def emr_global_scale(self) -> float:
        """Global scaling factor for enhanced mesh refinement."""
        return self._emr_global_scale

    @emr_global_scale.setter
    def emr_global_scale(self, value: float):
        self._emr_global_scale = value

    @property
    def emr_min_elem(self) -> float:
        """Minimum mesh element size."""
        return self._emr_min_elem

    @emr_min_elem.setter
    def emr_min_elem(self, value: float):
        self._emr_min_elem = value

    @property
    def swept_mesh(self) -> bool:
        """Whether swept mesh mode is enabled (for suitable geometries)."""
        return self._swept_mesh

    @swept_mesh.setter
    def swept_mesh(self, value: bool):
        self._swept_mesh = value

    def __repr__(self):
        """Provides a unified and informative string representation."""
        return (f"<plx.components.Mesh coarseness={self._mesh_coarseness.name}, "
                f"enhanced_refine={self._enhanced_refine}>")