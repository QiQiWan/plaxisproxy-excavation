# borehole.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar

# -----------------------------------------------------------------------------
# Project imports (keep the `type: ignore` if Pylance cannot resolve in your IDE)
# -----------------------------------------------------------------------------
from .core.plaxisobject import PlaxisObject as _PlxBase  # type: ignore
from .geometry import Point  # type: ignore
from .materials.soilmaterial import BaseSoilMaterial  # type: ignore

__all__ = [
    "SoilLayer",
    "BoreholeLayer",
    "Borehole",
    "BoreholeSet",
]

EPS = 1e-9
T_BH = TypeVar("T_BH", bound="Borehole")


# -----------------------------------------------------------------------------
# SoilLayer (thickness-centric, also keeps absolute elevations for bookkeeping)
# -----------------------------------------------------------------------------
class SoilLayer(_PlxBase):
    """
    Geological layer represented primarily by **thickness** (height).
    For bookkeeping/UI and mapper needs we also store **top/bottom elevations**.

    Fields
    ------
    name      : unique layer name within a project (mapper will ensure uniqueness)
    material  : in-memory soil material (should have `.plx_id` after material mapper)
    top_z     : optional top elevation (m)
    bottom_z  : optional bottom elevation (m)
    height    : optional thickness (m) = (top_z - bottom_z) if both defined
    plx_id    : optional PLAXIS-side handle (filled by mapper)
    """

    def __init__(
        self,
        name: str,
        comment: str = "SoilLayer",
        material: Optional[BaseSoilMaterial] = None,
        *,
        top_z: Optional[float] = None,
        bottom_z: Optional[float] = None,
        height: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, comment=comment)
        self.material: Optional[BaseSoilMaterial] = material
        self.top_z: Optional[float] = float(top_z) if top_z is not None else None
        self.bottom_z: Optional[float] = float(bottom_z) if bottom_z is not None else None
        self.height: Optional[float] = float(height) if height is not None else None
        self.plx_id: Any = None

    # ------------------------------ helpers ---------------------------------

    def thickness(self) -> float:
        """
        Return a non-negative thickness best-effort:
        - If both top/bottom are known → top-bottom
        - Else if height is known      → height
        - Else → 0.0
        """
        if self.top_z is not None and self.bottom_z is not None:
            return float(self.top_z) - float(self.bottom_z)
        if self.height is not None:
            return float(self.height)
        return 0.0

    def sync_height_from_bounds(self) -> None:
        """If both bounds exist, refresh `height = top - bottom`."""
        if self.top_z is not None and self.bottom_z is not None:
            self.height = float(self.top_z) - float(self.bottom_z)

    # ---------------------------- (de)serialization --------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "comment": self.comment,
            "material": getattr(self.material, "name", None),
            "top_z": None if self.top_z is None else float(self.top_z),
            "bottom_z": None if self.bottom_z is None else float(self.bottom_z),
            "height": None if self.height is None else float(self.height),
        }

    @classmethod
    def from_dict(
        cls: Type["SoilLayer"],
        data: Mapping[str, Any],
        *,
        material_resolver: Optional[Mapping[str, BaseSoilMaterial]] = None,
    ) -> "SoilLayer":
        """
        Build a SoilLayer from dict. If `material_resolver` is given, it maps
        material names to actual objects.
        """
        name = str(data.get("name", "Layer"))
        comment = str(data.get("comment", "SoilLayer"))
        mat_name = data.get("material")
        material: Optional[BaseSoilMaterial] = None
        if isinstance(material_resolver, Mapping) and isinstance(mat_name, str):
            material = material_resolver.get(mat_name)
        top_z = data.get("top_z")
        bottom_z = data.get("bottom_z")
        height = data.get("height")
        return cls(
            name=name,
            comment=comment,
            material=material,
            top_z=None if top_z is None else float(top_z),
            bottom_z=None if bottom_z is None else float(bottom_z),
            height=None if height is None else float(height),
        )


# -----------------------------------------------------------------------------
# BoreholeLayer (absolute-elevation-centric; references a SoilLayer)
# -----------------------------------------------------------------------------
class BoreholeLayer(_PlxBase):
    """
    Geological layer described by **absolute elevations**.

    Conventions
    -----------
    - `top_z` and `bottom_z` are elevations (m), usually 0.0 at ground and negative below.
    - `top_z >= bottom_z` must hold.
    - `soil_layer` references the canonical `SoilLayer` (shared across boreholes).
    """

    def __init__(
        self,
        name: str,
        top_z: float,
        bottom_z: float,
        soil_layer: SoilLayer,
        comment: str = "BoreholeLayer",
    ) -> None:
        super().__init__(name=name, comment=comment)
        self.top_z: float = float(top_z)
        self.bottom_z: float = float(bottom_z)
        self.soil_layer: SoilLayer = soil_layer
        self.plx_id: Any = None

    # ------------------------------ helpers ---------------------------------

    def thickness(self) -> float:
        return float(self.top_z) - float(self.bottom_z)

    # ---------------------------- (de)serialization --------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "comment": self.comment,
            "soil_layer": self.soil_layer.name,
            "top_z": float(self.top_z),
            "bottom_z": float(self.bottom_z),
        }


# -----------------------------------------------------------------------------
# Borehole (single location with multiple BoreholeLayer)
# -----------------------------------------------------------------------------
class Borehole(_PlxBase):
    """
    Represents a single borehole with location, ground level, and ordered
    `BoreholeLayer` entries.

    Fields
    ------
    name          : borehole ID
    location      : Point (Z usually ignored in PLAXIS borehole creation)
    ground_level  : elevation at this borehole (m)
    water_head    : optional groundwater head (m)
    layers        : List[BoreholeLayer] (top to bottom)
    plx_id        : PLAXIS-side handle (set by mapper)
    """

    def __init__(
        self,
        name: str,
        location: Point,
        ground_level: float = 0.0,
        comment: str = "",
        *,
        water_head: Optional[float] = None,
        layers: Optional[Sequence[BoreholeLayer]] = None,
    ) -> None:
        super().__init__(name, comment)
        if not isinstance(location, Point):
            raise TypeError("location must be a Point.")
        self.location: Point = location
        self.ground_level: float = float(ground_level)
        self.water_head: Optional[float] = float(water_head) if water_head is not None else None
        self.layers: List[BoreholeLayer] = list(layers) if layers else []
        self.plx_id: Any = None

    # ---- editing ----
    def add_layer(self, layer: BoreholeLayer, *, enforce_continuity: bool = True) -> None:
        """Append a layer; optionally enforce continuity with previous bottom."""
        if not isinstance(layer, BoreholeLayer):
            raise TypeError("layer must be a BoreholeLayer instance.")
        if self.layers and enforce_continuity:
            prev_bot = float(self.layers[-1].bottom_z)
            if abs(prev_bot - float(layer.top_z)) > EPS:
                raise ValueError(
                    f"Layer continuity violated: new.top_z={layer.top_z} != prev.bottom_z={prev_bot}"
                )
        self.layers.append(layer)

    def validate(self) -> None:
        """Validate ordering and continuity inside this borehole."""
        for i, ly in enumerate(self.layers):
            t = float(ly.top_z)
            b = float(ly.bottom_z)
            if t < b:
                raise ValueError(f"Layer[{i}] violates top_z >= bottom_z.")
            if i > 0:
                prev_bot = float(self.layers[i - 1].bottom_z)
                if abs(prev_bot - t) > EPS:
                    raise ValueError(
                        f"Layer[{i}] top_z must equal Layer[{i-1}] bottom_z (got {t} vs {prev_bot})."
                    )

    def sort_layers_topdown(self) -> None:
        """Sort layers by top elevation descending (top to bottom)."""
        self.layers.sort(key=lambda L: (float(L.top_z), float(L.bottom_z)), reverse=True)

    def replace_layers(self, new_layers: Sequence[BoreholeLayer]) -> None:
        """Replace internal layer list (no validation)."""
        self.layers = list(new_layers)

    def depth(self) -> float:
        """Total depth from ground level to bottom of last layer (>=0)."""
        if not self.layers:
            return 0.0
        last_bottom = float(self.layers[-1].bottom_z)
        return float(self.ground_level) - last_bottom

    def layers_by_name(self) -> Dict[str, BoreholeLayer]:
        """
        Return a mapping from **canonical soil-layer name** to BoreholeLayer.
        If a BoreholeLayer has no `soil_layer` (should not happen here), fall back to its own name.
        """
        out: Dict[str, BoreholeLayer] = {}
        for ly in self.layers:
            key = ly.soil_layer.name if getattr(ly, "soil_layer", None) and getattr(ly.soil_layer, "name", None) else ly.name
            out[key] = ly
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "comment": self.comment,
            "location": {"x": self.location.x, "y": self.location.y, "z": getattr(self.location, "z", 0.0)},
            "ground_level": self.ground_level,
            "water_head": self.water_head,
            "layers": [ly.to_dict() for ly in self.layers],
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Borehole name={self.name} n_layers={len(self.layers)} at ({self.location.x},{self.location.y})>"


# -----------------------------------------------------------------------------
# BoreholeSet (collection + normalization + unique-naming + queries)
# -----------------------------------------------------------------------------
class BoreholeSet(_PlxBase):
    """
    Container for multiple Borehole objects.

    Highlights
    ----------
    - `ensure_unique_names()` makes names globally unique across **materials,
      soil layers, boreholes, borehole layers** (avoids PLAXIS naming conflicts).
    - `normalize_all_boreholes()` unifies the **ordered layer list** for all
      boreholes, and **inserts zero-thickness layers** for missing ones using
      an *adjacent-neighbor* rule:

        • only-above exists → z = above.bottom
        • only-below exists → z = below.top
        • above & below exist:
            - if |above.bottom - below.top| <= eps: z = common boundary
            - else: z = 0.5*(above.bottom + below.top) and **also fix**
              above.bottom = z, below.top = z (eliminates overlap/gap)

      After normalization, each borehole's layers are continuous:
      layer[i].top_z == layer[i-1].bottom_z (within eps).
    """

    def __init__(
        self,
        name: str = "Boreholes",
        comment: str = "",
        boreholes: Optional[Sequence[Borehole]] = None,
    ) -> None:
        super().__init__(name=name, comment=comment)
        self.boreholes: List[Borehole] = list(boreholes) if boreholes else []
        self.plx_id: Any = None
        # canonical SoilLayer store by name (built on demand)
        self._global_soil_layers: Dict[str, SoilLayer] = {}

    # -------------------------- name uniqueness -----------------------------

    def ensure_unique_names(self) -> None:
        """
        Make **all names** unique across:
          - materials
          - soil layers
          - boreholes
          - borehole layers

        Strategy: keep a single global `used` set; when a conflict occurs,
        append a type-specific suffix + an increasing index. Mutates names in-place.
        """
        used: Set[str] = set()

        def _reserve(base: str, suffix: str) -> str:
            base = base or suffix
            if base not in used:
                used.add(base)
                return base
            k = 1
            while True:
                cand = f"{base}_{suffix}{k}"
                if cand not in used:
                    used.add(cand)
                    return cand
                k += 1

        # 1) materials (unique by object identity)
        seen_mats: Set[int] = set()
        for bh in self.boreholes:
            for ly in bh.layers:
                mat = getattr(getattr(ly, "soil_layer", None), "material", None)
                if mat is None:
                    continue
                oid = id(mat)
                if oid in seen_mats:
                    continue
                # set (or reset) a unique material name
                setattr(mat, "name", _reserve(str(getattr(mat, "name", "") or "SoilMat"), "MAT"))
                seen_mats.add(oid)

        # 2) soil layers (unique by identity)
        for sl in self._iter_unique_soil_layers():
            sl.name = _reserve(str(getattr(sl, "name", "") or "Layer"), "LY")

        # 3) boreholes
        for bh in self.boreholes:
            bh.name = _reserve(str(getattr(bh, "name", "") or "BH"), "BH")

        # 4) borehole layers
        for bh in self.boreholes:
            for ly in bh.layers:
                base = str(getattr(ly, "name", "") or (getattr(ly.soil_layer, "name", None) or "BHL"))
                ly.name = _reserve(base, "BHL")

    def _iter_unique_soil_layers(self) -> Sequence[SoilLayer]:
        """Return unique SoilLayer objects (by identity) across the set."""
        seen: Set[int] = set()
        out: List[SoilLayer] = []
        for bh in self.boreholes:
            for ly in bh.layers:
                sl = getattr(ly, "soil_layer", None)
                if sl is None:
                    continue
                oid = id(sl)
                if oid in seen:
                    continue
                seen.add(oid)
                out.append(sl)
        return out

    @property
    def unique_soil_layers(self) -> List[SoilLayer]:
        """Public accessor used by mappers."""
        return list(self._iter_unique_soil_layers())

    # -------------------------- normalization -------------------------------

    def _collect_ordered_layer_names(self) -> List[str]:
        """Collect union of all SoilLayer names in order of first appearance."""
        names: List[str] = []
        seen: Set[str] = set()
        for bh in self.boreholes:
            for ly in bh.layers:
                lname = (
                    ly.soil_layer.name
                    if getattr(ly, "soil_layer", None) and getattr(ly.soil_layer, "name", None)
                    else ly.name
                )
                if lname not in seen:
                    names.append(lname)
                    seen.add(lname)
        return names

    def _get_or_make_global_soil_layer(self, lname: str) -> SoilLayer:
        """Return a canonical SoilLayer object for `lname` (create if missing)."""
        sl = self._global_soil_layers.get(lname)
        if sl is not None:
            return sl
        # Adopt an existing object if any borehole carries it
        for bh in self.boreholes:
            for ly in bh.layers:
                if getattr(ly, "soil_layer", None) and getattr(ly.soil_layer, "name", None) == lname:
                    self._global_soil_layers[lname] = ly.soil_layer
                    return ly.soil_layer
        # Otherwise, create a placeholder (no material yet)
        sl = SoilLayer(name=lname, comment="UnifiedSoilLayer", material=None)
        self._global_soil_layers[lname] = sl
        return sl

    def normalize_all_boreholes(
        self,
        *,
        eps: float = EPS,
        in_place: bool = True,
    ) -> Dict[str, Any]:
        """
        Normalize layer ordering across all boreholes and insert **zero-thickness**
        layers for missing ones using the *adjacent-neighbor* rule:

          • only-above → z = above.bottom
          • only-below → z = below.top
          • both exist:
                if |ab - bt| <= eps → z = boundary
                else:
                    z = 0.5*(ab + bt)
                    above.bottom = z
                    below.top    = z

        Also enforces intra-borehole continuity:
        new_layers[i].top_z == new_layers[i-1].bottom_z (snapped).

        Returns
        -------
        {
          "ordered_names": [layer_name_top_to_bottom, ...],
          "boreholes": {
             "<bh.name>": { "<layer_name>": (top_z, bottom_z), ... },
             ...
          }
        }
        """
        ordered_names = self._collect_ordered_layer_names()
        if not ordered_names:
            return {"ordered_names": [], "n_boreholes": len(self.boreholes), "boreholes": {}}

        # Prepare canonical SoilLayer objects for all names
        for nm in ordered_names:
            _ = self._get_or_make_global_soil_layer(nm)

        info: Dict[str, Any] = {"ordered_names": ordered_names, "boreholes": {}}

        for bh in self.boreholes:
            present = bh.layers_by_name()  # lname -> BoreholeLayer
            new_layers: List[BoreholeLayer] = []

            for idx, lname in enumerate(ordered_names):
                cur = present.get(lname)
                if cur is None:
                    # Missing layer → insert zero-thickness one,
                    # placed against nearest neighbor(s)
                    above_z: Optional[float] = None
                    below_z: Optional[float] = None

                    # find above: last existing layer before idx
                    for j in range(idx - 1, -1, -1):
                        nm = ordered_names[j]
                        if nm in present:
                            above_z = float(present[nm].bottom_z)
                            break
                    # find below: first existing layer after idx
                    for j in range(idx + 1, len(ordered_names)):
                        nm = ordered_names[j]
                        if nm in present:
                            below_z = float(present[nm].top_z)
                            break

                    # fallback when no neighbor found → use local ground level
                    if above_z is None and below_z is None:
                        z = float(bh.ground_level)
                    elif above_z is not None and below_z is None:
                        z = float(above_z)
                    elif above_z is None and below_z is not None:
                        z = float(below_z)
                    else:
                        # both exist
                        assert above_z is not None and below_z is not None
                        if abs(above_z - below_z) <= eps:
                            z = above_z  # common boundary
                        else:
                            # resolve gap/overlap by splitting the difference
                            z = 0.5 * (above_z + below_z)
                            # adjust neighbors if they exist in present
                            if idx - 1 >= 0:
                                nm_above = ordered_names[idx - 1]
                                if nm_above in present:
                                    present[nm_above].bottom_z = float(z)
                            if idx + 1 < len(ordered_names):
                                nm_below = ordered_names[idx + 1]
                                if nm_below in present:
                                    present[nm_below].top_z = float(z)

                    sl = self._get_or_make_global_soil_layer(lname)
                    cur = BoreholeLayer(name=f"{lname}@{bh.name}", top_z=float(z), bottom_z=float(z), soil_layer=sl)

                # snap continuity with previous if needed
                if new_layers:
                    prev = new_layers[-1]
                    if abs(float(prev.bottom_z) - float(cur.top_z)) > eps:
                        # prefer snapping the current top to previous bottom
                        cur.top_z = float(prev.bottom_z)
                    # guard against accidental inversion
                    if float(cur.top_z) < float(cur.bottom_z):
                        cur.bottom_z = float(cur.top_z)

                new_layers.append(cur)

            # Final pass: strict continuity & non-negative thickness
            for k in range(1, len(new_layers)):
                prev = new_layers[k - 1]
                cur = new_layers[k]
                cur.top_z = float(prev.bottom_z)
                if float(cur.top_z) < float(cur.bottom_z):
                    cur.bottom_z = float(cur.top_z)

            if in_place:
                bh.layers = new_layers

            # summary per borehole
            info["boreholes"][bh.name] = {
                ly.soil_layer.name: (float(ly.top_z), float(ly.bottom_z)) for ly in (bh.layers if in_place else new_layers)
            }

        return info

    # ------------------------------- queries ---------------------------------

    def layer_index_in_borehole(self, lname: str, bh_index: int) -> int:
        """Return the index of layer `lname` in borehole #`bh_index`, or -1."""
        if bh_index < 0 or bh_index >= len(self.boreholes):
            return -1
        bh = self.boreholes[bh_index]
        for i, ly in enumerate(bh.layers):
            key = ly.soil_layer.name if getattr(ly, "soil_layer", None) and getattr(ly.soil_layer, "name", None) else ly.name
            if key == lname:
                return i
        return -1

    def summary_by_borehole(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Return { BH_name: { layer_name: (top_z, bottom_z) } } for all boreholes.
        """
        out: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for bh in self.boreholes:
            out[bh.name] = {
                (ly.soil_layer.name if getattr(ly, "soil_layer", None) and getattr(ly.soil_layer, "name", None) else ly.name):
                (float(ly.top_z), float(ly.bottom_z))
                for ly in bh.layers
            }
        return out

    # ---------------------------- (de)serialization --------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "comment": self.comment,
            "boreholes": [bh.to_dict() for bh in self.boreholes],
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BoreholeSet name={self.name} count={len(self.boreholes)}>"
