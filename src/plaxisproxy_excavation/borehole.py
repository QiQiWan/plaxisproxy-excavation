# borehole.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar

# #############################################################################
# Project imports (keep the `type: ignore` if Pylance cannot resolve in your IDE)
# #############################################################################
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


# #############################################################################
# SoilLayer (thickness-centric, also keeps absolute elevations for bookkeeping)
# #############################################################################
class SoilLayer(_PlxBase):
    """
    Geological layer represented primarily by **thickness** (height).
    For bookkeeping/UI and mapper needs we also store **top/bottom elevations**.

    Fields
    ######
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

    # ############################## helpers #################################

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

    # ############################ (de)serialization ##########################

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


# #############################################################################
# BoreholeLayer (absolute-elevation-centric; references a SoilLayer)
# #############################################################################
class BoreholeLayer(_PlxBase):
    """
    Geological layer described by **absolute elevations**.

    Conventions
    ###########
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

    # ############################## helpers #################################

    def thickness(self) -> float:
        return float(self.top_z) - float(self.bottom_z)

    # ############################ (de)serialization ##########################

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "comment": self.comment,
            "soil_layer": self.soil_layer.name,
            "top_z": float(self.top_z),
            "bottom_z": float(self.bottom_z),
        }


# #############################################################################
# Borehole (single location with multiple BoreholeLayer)
# #############################################################################
class Borehole(_PlxBase):
    """
    Represents a single borehole with location, ground level, and ordered
    `BoreholeLayer` entries.

    Fields
    ######
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

    # #### editing ####
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


# #############################################################################
# BoreholeSet (collection + normalization + unique-naming + queries)
# #############################################################################
class BoreholeSet(_PlxBase):
    """
    Container for multiple Borehole objects.

    Highlights
    ##########
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

    # ########################## name uniqueness #############################

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

    # ########################## normalization ###############################

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

    def normalize_all_boreholes(self, *, eps: float = EPS, in_place: bool = True) -> Dict[str, Any]:
        """
        Normalize across boreholes WITHOUT using layer names.
        Global order is derived from per-borehole SoilLayer object sequences (top→bottom)
        plus cross-hole adjacency (topological sort). If the same SoilLayer reappears
        after being interrupted by others within one borehole, it becomes a NEW VARIANT
        (#2/#3/...). Missing variants can be inserted as zero-thickness at consistent
        pinch-out interfaces.

        I/O contract: in-place rewrite of `bh.layers`. Return a summary dict with the
        same shape as the original implementation.
        """
        # #### Internal policies (do not change external interface) ####
        SPLIT_REAPPEARING = True          # split non-consecutive reappearance into variants
        INSERT_ZERO_THICKNESS = True      # ensure each BH is a subsequence of global order
        PINCHOUT_POLICY = "top_of_next"   # or "bottom_of_prev"

        # ########## helpers (object-based; NO name-based logic) ##########
        def base_id_of(bl: BoreholeLayer) -> int:
            sl = getattr(bl, "soil_layer", None)
            return id(sl) if sl is not None else id(bl)  # fallback to BL object id

        # 1) Per-borehole sequences of variants: key = (base_id, occurrence_index)
        per_bh_sequences: Dict[str, List[Tuple[Tuple[int, int], BoreholeLayer]]] = {}
        global_max_occ: Dict[int, int] = {}

        for bh in self.boreholes:
            counts: Dict[int, int] = {}
            seq: List[Tuple[Tuple[int, int], BoreholeLayer]] = []
            last_base: Optional[int] = None
            for bl in bh.layers:
                bid = base_id_of(bl)
                if SPLIT_REAPPEARING:
                    if last_base is None or bid != last_base:
                        counts[bid] = counts.get(bid, 0) + 1
                else:
                    counts[bid] = counts.get(bid, 1)
                occ = counts[bid]
                key = (bid, occ)
                seq.append((key, bl))
                last_base = bid
                if occ > global_max_occ.get(bid, 0):
                    global_max_occ[bid] = occ
            per_bh_sequences[bh.name] = seq

        # 2) Build partial order + topological sort
        from collections import defaultdict, deque
        edges: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
        indeg: Dict[Tuple[int, int], int] = defaultdict(int)
        nodes: Set[Tuple[int, int]] = set()

        for _, seq in per_bh_sequences.items():
            prev_key: Optional[Tuple[int, int]] = None
            for key, _ in seq:
                nodes.add(key)
                if prev_key is not None and prev_key != key:
                    if key not in edges[prev_key]:
                        edges[prev_key].add(key)
                        indeg[key] += 1
                        nodes.add(prev_key)
                prev_key = key

        # ensure all variants exist as nodes
        for bid, max_occ in global_max_occ.items():
            for occ in range(1, max_occ + 1):
                nodes.add((bid, occ))

        q = deque([n for n in nodes if indeg.get(n, 0) == 0])
        global_order: List[Tuple[int, int]] = []
        while q:
            u = q.popleft()
            global_order.append(u)
            for v in edges.get(u, ()):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(global_order) != len(nodes):
            # deterministic fallback (cycle): order by first-seen base then occurrence
            first_seen: Dict[int, int] = {}
            idx = 0
            for _, seq in per_bh_sequences.items():
                for (bid, _), __ in seq:
                    if bid not in first_seen:
                        first_seen[bid] = idx
                        idx += 1
            global_order = sorted(nodes, key=lambda k: (first_seen.get(k[0], 10**9), k[1]))

        # 3) Create canonical SoilLayer per variant (clone base + "#occ" suffix for occ>=2)
        def clone_soil_layer(base: SoilLayer, occ: int) -> SoilLayer:
            suffix = f"#{occ}" if occ > 1 else ""
            nm = getattr(base, "name", f"Layer_{id(base)}") + suffix
            return SoilLayer(nm, material=getattr(base, "material", None))

        variant_layer_obj: Dict[Tuple[int, int], SoilLayer] = {}
        for key in global_order:
            bid, occ = key
            base: Optional[SoilLayer] = None
            # exact variant base
            for _, seq in per_bh_sequences.items():
                for (kb, ko), bl in seq:
                    if (kb, ko) == key:
                        base = getattr(bl, "soil_layer", None)
                        break
                if base is not None:
                    break
            # sibling base if exact not found
            if base is None:
                for _, seq in per_bh_sequences.items():
                    for (kb, _), bl in seq:
                        if kb == bid:
                            base = getattr(bl, "soil_layer", None)
                            if base is not None:
                                break
                    if base is not None:
                        break
            # fabricate neutral if still None
            variant_layer_obj[key] = clone_soil_layer(base, occ) if base else SoilLayer(f"Layer_{bid}#{occ}", material=None)

        # 4) Rewrite each borehole to align with global order
        def top_of_next_present(i: int, present_map: Dict[Tuple[int, int], BoreholeLayer]) -> Optional[float]:
            for j in range(i, len(global_order)):
                k = global_order[j]
                bl = present_map.get(k)
                if bl is not None:
                    return float(bl.top_z)
            return None

        def bottom_of_prev_present(i: int, present_map: Dict[Tuple[int, int], BoreholeLayer]) -> Optional[float]:
            for j in range(i - 1, -1, -1):
                k = global_order[j]
                bl = present_map.get(k)
                if bl is not None:
                    return float(bl.bottom_z)
            return None

        summary: Dict[str, Any] = {"ordered_names": [variant_layer_obj[k].name for k in global_order], "boreholes": {}}

        for bh in self.boreholes:
            # map variants present in this BH
            counts_local: Dict[int, int] = {}
            present: Dict[Tuple[int, int], BoreholeLayer] = {}
            last_base: Optional[int] = None
            for bl in bh.layers:
                bid = base_id_of(bl)
                if SPLIT_REAPPEARING:
                    if last_base is None or bid != last_base:
                        counts_local[bid] = counts_local.get(bid, 0) + 1
                else:
                    counts_local[bid] = counts_local.get(bid, 1)
                occ = counts_local[bid]
                present[(bid, occ)] = bl
                last_base = bid

            new_layers: List[BoreholeLayer] = []
            for i, key in enumerate(global_order):
                if key in present:
                    old = present[key]
                    sl = variant_layer_obj[key]
                    # keep original BL name & geometry; only swap soil_layer
                    new_layers.append(BoreholeLayer(getattr(old, "name", sl.name), float(old.top_z), float(old.bottom_z), sl))
                else:
                    if not INSERT_ZERO_THICKNESS:
                        continue
                    # pin zero-thickness at a consistent interface
                    if PINCHOUT_POLICY == "top_of_next":
                        z = top_of_next_present(i, present)
                        if z is None:
                            z = bottom_of_prev_present(i, present) or 0.0
                    else:  # "bottom_of_prev"
                        z = bottom_of_prev_present(i, present)
                        if z is None:
                            z = top_of_next_present(i, present) or 0.0
                    sl = variant_layer_obj[key]
                    nm = f"{sl.name}@{bh.name}_zero"
                    new_layers.append(BoreholeLayer(nm, float(z), float(z), sl))

            if in_place:
                bh.layers = new_layers

            # fill summary per original shape
            summary["boreholes"][bh.name] = {
                (ly.soil_layer.name if getattr(ly, "soil_layer", None) else ly.name): (float(ly.top_z), float(ly.bottom_z))
                for ly in (bh.layers if in_place else new_layers)
            }

        return summary


    # ############################### queries #################################

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

    # ############################ (de)serialization ##########################

    def to_dict(self) -> Dict[str, Any]:
        return {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "comment": self.comment,
            "boreholes": [bh.to_dict() for bh in self.boreholes],
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<BoreholeSet name={self.name} count={len(self.boreholes)}>"
