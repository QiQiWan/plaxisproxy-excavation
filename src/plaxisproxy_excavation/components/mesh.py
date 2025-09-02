# mesh.py
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, Optional


class MeshCoarseness(Enum):
    """
    Discrete presets for the target element relative size (mesh factor).

    Notes
    -----
    The numeric values are typical element-relative-size factors used by
    PLAXIS 3D. You can override these with `element_relative_size` if you
    want a custom value.
    """
    HighRoughness = 0.10   # very coarse
    Roughness     = 0.075  # coarse
    Medium        = 0.05   # medium (default)
    Refine        = 0.035  # refined
    HighRefine    = 0.025  # highly refined


class Mesh:
    """
    Mesh generation settings for PLAXIS 3D.

    This object is meant to be consumed by a `MeshMapper` that:
      1) writes options to `g_i.MeshOptions` (or equivalent), and
      2) calls the `mesh` command with the right signature.

    Design goals
    ------------
    - Backward compatible with your previous `Mesh` object:
      * `mesh_coarseness`, `enhanced_refine`, `emr_global_scale`,
        `emr_min_elem`, `swept_mesh`.
    - Extended control matching the official command/options:
      * `max_cpus`, `emr_proximity`, `element_relative_size`.
    - If an optional field is left as `None`, the mapper skips it.
    """

    def __init__(
        self,
        mesh_coarseness: MeshCoarseness = MeshCoarseness.Medium,
        *,
        enhanced_refine: bool = True,           # maps to MeshOptions.UseEnhancedRefinements
        emr_global_scale: float = 1.2,          # maps to MeshOptions.EMRGlobalScale
        emr_min_elem: float = 5e-3,             # maps to MeshOptions.EMRMinElementSize (m)
        swept_mesh: bool = True,                # kept for your project flow; not directly used by `mesh` command

        # Optional: direct plumbing to documented `mesh` arguments and options
        max_cpus: Optional[int] = None,         # maps to MeshOptions.MaxCPUs and `mesh(..., maxCPUs, ...)`
        emr_proximity: Optional[float] = None,  # maps to MeshOptions.EMRProximity and `mesh(..., EMRProximity)`
        element_relative_size: Optional[float] = None,  # if set, overrides enum value for the mesh factor
    ) -> None:
        self._mesh_coarseness = mesh_coarseness
        self._enhanced_refine = bool(enhanced_refine)
        self._emr_global_scale = float(emr_global_scale)
        self._emr_min_elem = float(emr_min_elem)
        self._swept_mesh = bool(swept_mesh)

        self._max_cpus = max_cpus if (max_cpus is None or isinstance(max_cpus, int)) else int(max_cpus)
        self._emr_proximity = float(emr_proximity) if emr_proximity is not None else None
        # Custom mesh size
        self._element_relative_size = float(element_relative_size) if element_relative_size is not None else None

    # ---------------- Properties (backward compatible) ----------------

    @property
    def mesh_coarseness(self) -> MeshCoarseness:
        """Discrete coarseness preset. Ignored if `element_relative_size` is set."""
        return self._mesh_coarseness

    @mesh_coarseness.setter
    def mesh_coarseness(self, value: MeshCoarseness) -> None:
        self._mesh_coarseness = value

    @property
    def enhanced_refine(self) -> bool:
        """Enable Enhanced Mesh Refinements (EMR)."""
        return self._enhanced_refine

    @enhanced_refine.setter
    def enhanced_refine(self, value: bool) -> None:
        self._enhanced_refine = bool(value)

    @property
    def emr_global_scale(self) -> float:
        """Global scale factor used by EMR."""
        return self._emr_global_scale

    @emr_global_scale.setter
    def emr_global_scale(self, value: float) -> None:
        self._emr_global_scale = float(value)

    @property
    def emr_min_elem(self) -> float:
        """Minimum element size (meters) used by EMR."""
        return self._emr_min_elem

    @emr_min_elem.setter
    def emr_min_elem(self, value: float) -> None:
        self._emr_min_elem = float(value)

    @property
    def swept_mesh(self) -> bool:
        """
        Whether to prefer swept mesh in your workflow (not a direct `mesh` argument).
        Kept for compatibility; your mapper may use it to enforce swept-able geometry.
        """
        return self._swept_mesh

    @swept_mesh.setter
    def swept_mesh(self, value: bool) -> None:
        self._swept_mesh = bool(value)

    # ---------------- New optional fields (full command coverage) ----------------

    @property
    def max_cpus(self) -> Optional[int]:
        """Maximum CPU cores for meshing; if None, PLAXIS decides."""
        return self._max_cpus

    @max_cpus.setter
    def max_cpus(self, value: Optional[int]) -> None:
        if value is None:
            self._max_cpus = None
        else:
            iv = int(value)
            if iv <= 0:
                raise ValueError("max_cpus must be positive when provided.")
            self._max_cpus = iv

    @property
    def emr_proximity(self) -> Optional[float]:
        """EMR proximity threshold; if None, keep PLAXIS default."""
        return self._emr_proximity

    @emr_proximity.setter
    def emr_proximity(self, value: Optional[float]) -> None:
        self._emr_proximity = None if value is None else float(value)

    @property
    def element_relative_size(self) -> Optional[float]:
        """
        Directly set the mesh factor used by `mesh`.
        If provided, it overrides `mesh_coarseness.value`.
        """
        return self._element_relative_size

    @element_relative_size.setter
    def element_relative_size(self, value: Optional[float]) -> None:
        self._element_relative_size = None if value is None else float(value)

    # ---------------- Helpers ----------------

    def factor(self) -> float:
        """
        Return the mesh factor that the `mesh` command should use:
          - If `element_relative_size` is set, return it;
          - Otherwise return `mesh_coarseness.value`.
        """
        if self._element_relative_size is not None:
            return float(self._element_relative_size)
        return float(self._mesh_coarseness.value)

    def validate(self) -> None:
        """
        Lightweight validation to catch obvious mistakes early.
        Your mapper may call this before applying the settings.
        """
        if self._emr_global_scale <= 0:
            raise ValueError("emr_global_scale must be > 0.")
        if self._emr_min_elem <= 0:
            raise ValueError("emr_min_elem must be > 0.")
        if self._element_relative_size is not None and self._element_relative_size <= 0:
            raise ValueError("element_relative_size must be > 0 when provided.")
        if self._max_cpus is not None and self._max_cpus <= 0:
            raise ValueError("max_cpus must be > 0 when provided.")

    # ---------------- Serialization ----------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a simple dict for persistence/UI."""
        return {
            "mesh_coarseness": self._mesh_coarseness.name,
            "enhanced_refine": self._enhanced_refine,
            "emr_global_scale": self._emr_global_scale,
            "emr_min_elem": self._emr_min_elem,
            "swept_mesh": self._swept_mesh,
            "max_cpus": self._max_cpus,
            "emr_proximity": self._emr_proximity,
            "element_relative_size": self._element_relative_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Mesh":
        """
        Rebuild a Mesh object from a dict created by `to_dict()`.
        Missing keys fall back to sensible defaults.
        """
        name = d.get("mesh_coarseness", MeshCoarseness.Medium.name)
        coarse = MeshCoarseness[name] if isinstance(name, str) else MeshCoarseness.Medium
        return cls(
            mesh_coarseness=coarse,
            enhanced_refine=bool(d.get("enhanced_refine", True)),
            emr_global_scale=float(d.get("emr_global_scale", 1.2)),
            emr_min_elem=float(d.get("emr_min_elem", 5e-3)),
            swept_mesh=bool(d.get("swept_mesh", True)),
            max_cpus=(int(d["max_cpus"]) if d.get("max_cpus") is not None else None),
            emr_proximity=(float(d["emr_proximity"]) if d.get("emr_proximity") is not None else None),
            element_relative_size=(float(d["element_relative_size"]) if d.get("element_relative_size") is not None else None),
        )

    def __repr__(self) -> str:
        return (f"<Mesh coarseness={self._mesh_coarseness.name} "
                f"(factor={self.factor():g}), EMR={self._enhanced_refine}>")
