"""
phase.py — Phase object integrating stage settings, structures, loads and a single water table.
All comments in English per convention.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Callable, Mapping
import importlib

# ---- base object ----
from ..core.plaxisobject import PlaxisObject  # package-style

# ---- stage settings ----
from .phasesettings import StageSettingsBase, StageSettingsFactory  # package-style

# ---- structures & soils ----
from ..structures.basestructure import BaseStructure  # package-style
from ..structures.soilblock import SoilBlock          # package-style

# ---- loads (static/dynamic) & multipliers ----
from ..structures.load import (                       # package-style
        _BaseLoad, LoadStage, LoadMultiplier,
    )

# ---- water table (single per phase) ----
from ..components.watertable import WaterLevelTable   # package-style


class Phase(PlaxisObject):
    """
    A single PLAXIS phase that ties together:
      - stage settings (numerics / calc type / consolidation / dynamics)
      - lists of structures/soil blocks/loads created in this phase
      - activation toggles: **structure objects** to be ACTIVATED / DEACTIVATED
      - a single water table per phase
      - inheritance: the PLAXIS phase this phase is created from (prev/base phase)
    """

    # ---------------------- construction ----------------------
    def __init__(
        self,
        name: str,
        *,
        comment: str = "",
        settings: StageSettingsBase,
        soil_blocks: Optional[Sequence[SoilBlock]] = None,
        structures: Optional[Sequence[BaseStructure]] = None,
        static_loads: Optional[Sequence[_BaseLoad]] = None,
        dynamic_loads: Optional[Sequence[_BaseLoad]] = None,
        load_multipliers: Optional[Dict[str, LoadMultiplier]] = None,
        water_table: Optional[WaterLevelTable] = None,
        activate: Optional[Sequence[BaseStructure]] = None,
        deactivate: Optional[Sequence[BaseStructure]] = None,
        inherits: Optional["Phase"] = None,   # NEW: the phase this one derives from
        # NEW: per-phase well parameter overrides, by well name
        well_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(name=name, comment=comment)

        # Stage settings (strongly typed)
        if not isinstance(settings, StageSettingsBase):
            raise TypeError("settings must be a StageSettingsBase (use StageSettingsFactory if needed).")
        self._settings: StageSettingsBase = settings

        # Domain members
        self._soil_blocks: List[SoilBlock] = list(soil_blocks) if soil_blocks else []
        self._structures: List[BaseStructure] = list(structures) if structures else []

        # Loads separated by stage (static/dynamic)
        self._static_loads: List[_BaseLoad] = []
        self._dynamic_loads: List[_BaseLoad] = []
        if static_loads:
            for L in static_loads:
                self.add_load(L)
        if dynamic_loads:
            for L in dynamic_loads:
                self.add_load(L)

        # Global multipliers registry (optional convenience dictionary)
        self._load_multipliers: Dict[str, LoadMultiplier] = dict(load_multipliers or {})

        # Single water table for this phase
        self._water_table: Optional[WaterLevelTable] = water_table

        # Activation toggles: **store structure objects (not ids)**
        self._activate: List[BaseStructure] = list(activate or [])
        self._deactivate: List[BaseStructure] = list(deactivate or [])

        # NEW: inheritance reference (may carry an existing plx_id)
        self._inherits: Optional["Phase"] = inherits
        self._well_overrides: Dict[str, Dict[str, Any]] = dict(well_overrides or {})

    # ---------------------- properties ----------------------
    @property
    def settings(self) -> StageSettingsBase:
        return self._settings

    @property
    def soil_blocks(self) -> List[SoilBlock]:
        return self._soil_blocks

    @property
    def structures(self) -> List[BaseStructure]:
        return self._structures

    @property
    def static_loads(self) -> List[_BaseLoad]:
        return self._static_loads

    @property
    def dynamic_loads(self) -> List[_BaseLoad]:
        return self._dynamic_loads

    @property
    def load_multipliers(self) -> Dict[str, LoadMultiplier]:
        return self._load_multipliers

    @property
    def water_table(self) -> Optional[WaterLevelTable]:
        return self._water_table
    
    @water_table.setter
    def water_table(self, wt: Any) -> None:
        self._water_table = wt

    @property
    def well_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Mapping: well_name -> { 'q_well': float, 'h_min': float, 'well_type': str|enum }"""
        return self._well_overrides

    @well_overrides.setter
    def well_overrides(self, mapping: Optional[Dict[str, Dict[str, Any]]]) -> None:
        self._well_overrides = dict(mapping or {})

    @property
    def activate(self) -> List[BaseStructure]:
        return self._activate

    @property
    def deactivate(self) -> List[BaseStructure]:
        return self._deactivate

    @property
    def inherits(self) -> Optional["Phase"]:
        """The previous/base Phase this one inherits from."""
        return self._inherits

    def set_inherits(self, base: Optional["Phase"]) -> "Phase":
        self._inherits = base
        return self

    # ---------------------- mutation helpers ----------------------
    def add_soils(self, *blocks: SoilBlock) -> "Phase":
        self._soil_blocks.extend(blocks)
        return self

    def add_structures(self, *objs: BaseStructure) -> "Phase":
        self._structures.extend(objs)
        return self

    def add_load(self, load: _BaseLoad) -> "Phase":
        stg = getattr(load, "stage", None)
        if stg == LoadStage.DYNAMIC or (isinstance(stg, str) and str(stg).lower().startswith("dynamic")):
            self._dynamic_loads.append(load)
        else:
            self._static_loads.append(load)
        return self

    def add_loads(self, *loads: _BaseLoad) -> "Phase":
        for L in loads:
            self.add_load(L)
        return self

    def set_water_table(self, tbl: WaterLevelTable) -> "Phase":
        self._water_table = tbl
        return self
    
    def set_well_overrides(self, mapping: Optional[Dict[str, Dict[str, Any]]]) -> "Phase":
        """Fluent helper."""
        self.well_overrides = mapping
        return self

    def activate_structures(self, objs: Sequence[BaseStructure]) -> "Phase":
        self._activate.extend(objs)
        return self

    def deactivate_structures(self, objs: Sequence[BaseStructure]) -> "Phase":
        self._deactivate.extend(objs)
        return self

    # ---------------------- exports for mappers ----------------------
    def settings_payload(self) -> Dict[str, Any]:
        # keep your original logic; this is a placeholder
        if hasattr(self._settings, "to_dict"):
            return self._settings.to_dict()
        return getattr(self._settings, "__dict__", {}) or {}

    def actions(self) -> Dict[str, Any]:
        """
        Snapshot for a PhaseMapper:
          - create: object dicts to be created in this phase
          - activate/deactivate: **full object dicts** (resolver will use name/id)
          - water_table: optional single table for this phase
          - inherits: minimal info of base phase (name + maybe plx_id)
        """
        return {
            "create": {
                "soils": [b.to_dict() for b in self._soil_blocks],
                "structures": [s.to_dict() for s in self._structures],
                "loads": {
                    "static": [L.to_dict() for L in self._static_loads],
                    "dynamic": [L.to_dict() for L in self._dynamic_loads],
                    "multipliers": {k: m.to_dict() for k, m in self._load_multipliers.items()},
                },
            },
            "activate": [s.to_dict() for s in self._activate],
            "deactivate": [s.to_dict() for s in self._deactivate],
            "water_table": None if self._water_table is None else self._water_table.to_dict(),
            "well_overrides": self._well_overrides,
            "inherits": None if self._inherits is None else {
                "name": self._inherits.name,
                "plx_id": getattr(self._inherits, "plx_id", None),
            }
        }

    # ---------------------- Soil block helper ----------------------
    def set_soil_overrides(self, overrides: Optional[Dict[str, Dict[str, bool]]]) -> "Phase":
        """
        Declare per-soil behavior for this phase. Example:
            {
              "Soil_2_1": {"active": False},                 # excavate this piece
              "Soil_2_2": {"deformable": False},             # freeze this piece (flow-only)
              "Soil_5_3": {"active": True, "deformable": True}
            }
        Notes:
          - Omitted keys are left unchanged by the builder.
          - This method only stores intent; the builder applies it to PLAXIS when creating the phase.
        """
        self._soil_overrides: Dict[str, Dict[str, bool]] = overrides or {}
        return self

    def get_soil_overrides(self) -> Dict[str, Dict[str, bool]]:
        return getattr(self, "_soil_overrides", {})

    # ---------------------- (de)serialization ----------------------
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "settings": self._settings.to_dict() if hasattr(self._settings, "to_dict") else self._settings,
            "soil_blocks": [b.to_dict() for b in self._soil_blocks],
            "structures": [s.to_dict() for s in self._structures],
            "static_loads": [L.to_dict() for L in self._static_loads],
            "dynamic_loads": [L.to_dict() for L in self._dynamic_loads],
            "load_multipliers": {k: m.to_dict() for k, m in self._load_multipliers.items()},
            "water_table": None if self._water_table is None else self._water_table.to_dict(),
            "well_overrides": self._well_overrides,
            "activate": [s.to_dict() for s in self._activate],
            "deactivate": [s.to_dict() for s in self._deactivate],
            "inherits": None if self._inherits is None else {
                "name": self._inherits.name,
                "plx_id": getattr(self._inherits, "plx_id", None),
            },
        })
        return d

    @classmethod
    def from_dict(cls: Type["Phase"], data: Dict[str, Any]) -> "Phase":
        """Rehydrate Phase with polymorphic nested objects via __type__."""
        d = dict(data or {})

        # Stage settings
        raw_settings = d.get("settings")
        if isinstance(raw_settings, StageSettingsBase):
            settings = raw_settings
        elif isinstance(raw_settings, dict):
            settings = StageSettingsFactory.from_dict(raw_settings)
        else:
            raise TypeError("Missing or invalid 'settings' for Phase.from_dict().")

        def _rehydrate_one(x: Any) -> Any:
            if not isinstance(x, dict):
                return x
            t = x.get("__type__")
            if isinstance(t, str) and "." in t:
                try:
                    mod_name, _, cls_name = t.rpartition(".")
                    mod = importlib.import_module(mod_name)
                    cls_dyn = getattr(mod, cls_name)
                    if hasattr(cls_dyn, "from_dict"):
                        return cls_dyn.from_dict(x)
                    ctor_kwargs = {k: v for k, v in x.items() if not (isinstance(k, str) and k.startswith("__"))}
                    return cls_dyn(**ctor_kwargs)  # type: ignore[misc]
                except Exception:
                    return x
            return x

        def _rehydrate_list(v: Any) -> List[Any]:
            if isinstance(v, list):
                return [(_rehydrate_one(i)) for i in v]
            return []

        name = d.get("name", "Phase")
        comment = d.get("comment", "")

        soil_blocks = _rehydrate_list(d.get("soil_blocks"))
        structures = _rehydrate_list(d.get("structures"))
        static_loads = _rehydrate_list(d.get("static_loads"))
        dynamic_loads = _rehydrate_list(d.get("dynamic_loads"))

        # Multipliers dict (typed values)
        raw_mults = d.get("load_multipliers") or {}
        mults: Dict[str, LoadMultiplier] = {}
        if isinstance(raw_mults, Mapping):
            for k, v in raw_mults.items():
                mv = _rehydrate_one(v)
                if isinstance(mv, LoadMultiplier):
                    mults[str(k)] = mv

        # Water table (single)
        wt = d.get("water_table")
        water_table = _rehydrate_one(wt) if wt is not None else None
        well_overrides = data.get("well_overrides") or {}

        # Activation lists: **objects**
        activate_objs = _rehydrate_list(d.get("activate"))
        deactivate_objs = _rehydrate_list(d.get("deactivate"))

        # Inherits (minimal)
        inh = d.get("inherits")
        inherits_phase: Optional["Phase"] = None
        if isinstance(inh, dict):
            # only keep name and possible plx_id reference
            tmp = cls(name=inh.get("name", "BasePhase"),
                      comment="(rehydrated base)",
                      settings=settings)  # settings not used here
            setattr(tmp, "plx_id", inh.get("plx_id", None))
            inherits_phase = tmp

        obj = cls(
            name=name,
            comment=comment,
            settings=settings,
            soil_blocks=soil_blocks,        # type: ignore[arg-type]
            structures=structures,          # type: ignore[arg-type]
            static_loads=static_loads,      # type: ignore[arg-type]
            dynamic_loads=dynamic_loads,    # type: ignore[arg-type]
            load_multipliers=mults,
            water_table=water_table,        # type: ignore[arg-type]
            well_overrides=well_overrides,
            activate=activate_objs,         # type: ignore[arg-type]
            deactivate=deactivate_objs,     # type: ignore[arg-type]
            inherits=inherits_phase,
        )
        return obj

    # ---------------------- queries & validation ----------------------
    def list_structures(self, pred: Optional[Callable[[BaseStructure], bool]] = None) -> List[BaseStructure]:
        if pred is None:
            return list(self._structures)
        return [s for s in self._structures if pred(s)]

    def list_loads(self, *, dynamic: Optional[bool] = None) -> List[_BaseLoad]:
        if dynamic is None:
            return [*self._static_loads, *self._dynamic_loads]
        return list(self._dynamic_loads if dynamic else self._static_loads)

    def validate(self) -> None:
        if not isinstance(self._settings, StageSettingsBase):
            raise TypeError("Phase.settings must be StageSettingsBase.")
        # Ensure no duplicates between activate/deactivate (by id or name)
        def _key(o: Any) -> str:
            oid = getattr(o, "id", None)
            return str(oid) if oid else getattr(o, "name", "")
        dup = { _key(o) for o in self._activate } & { _key(o) for o in self._deactivate }
        if dup:
            raise ValueError(f"Same object appears in both activate and deactivate: {sorted(dup)}")

    def __repr__(self):
        return f"<plx.components.Phase name={self.name} id={self.id}>"