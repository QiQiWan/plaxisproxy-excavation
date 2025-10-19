# phasemapper.py  — Phase-first API (reduced handle usage, no ModelHandle)

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from ..components.phase import Phase
from ..structures.basestructure import BaseStructure
from ..components.phasesettings import *
from .watertablemapper import WaterTableMapper
import re


FAMILY_REGISTRY: list[dict] = [
    {
        "tag": "plate",
        "regex": r"plate",                         # Plate/WALL 等本质上都是 Plate
        "containers": ["Plates", "Walls"],
        "child_attrs": ("Children", "Members", "SubObjects", "Items"),
        "suffix_regex": r"(?:_\d+)?",
    },
    {
        "tag": "n2n_anchor",
        "regex": r"(node.?to.?node.*anchor|n2n.*anchor|anchor)",
        "containers": ["NodeToNodeAnchors", "NodetoNodeAnchors", "Anchors", "Node_to_node_anchors"],
        "child_attrs": ("Children", "Members", "Items", "SubObjects"),
        "suffix_regex": r"(?:_\d+)?",
    },
    {
        "tag": "beam",
        "regex": r"\bbeam\b",
        "containers": ["Beams", "EmbeddedBeams"],
        "child_attrs": (),
        "suffix_regex": r"",
    },
    {
        "tag": "embedded_pile",
        "regex": r"(embedded.*(pile|beam))",
        "containers": ["EmbeddedBeams", "EmbeddedPiles"],
        "child_attrs": (),
        "suffix_regex": r"",
    },
    {
        "tag": "well",
        "regex": r"\bwell\b",
        "containers": ["Wells", "Pipes"],
        "child_attrs": (),
        "suffix_regex": r"",
    },
    {
        "tag": "soil",
        "regex": r"(soil|soilvolume|soilbody)",
        "containers": ["SoilVolumes", "Soils"],
        "child_attrs": (),
        "suffix_regex": r"",
    },
]

# 针对域对象类名的快速映射（仅兜底；优先使用句柄的 TypeName）
CLASS_TAGS = {
    "RetainingWall": "plate",
    "Beam": "beam",
    "EmbeddedPile": "embedded_pile",
    "Anchor": "n2n_anchor",
    "Well": "well",
    "SoilBlock": "soil",
}

# Internal alias for PLAXIS phase handle (only used in private helpers)
PhaseHandle = Any

# --- heplers: pick up objects by names ---

def _ident_str(x) -> str:
    if x is None: return ""
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)

def _canon(name: str) -> str:
    """normalize name: keep [A-Za-z0-9_], collapse _, strip edges, case-insensitive elsewhere."""
    s = str(name or "")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)   # non-word -> _
    s = re.sub(r"_+", "_", s)               # collapse __ -> _
    return s.strip("_")

def _family_base_strict(name: str) -> str:
    """
    Strict base: remove ONLY trailing '_<digits>' groups (if any).
      'A1'        -> 'A1'
      'A1_1'      -> 'A1'
      'PLATE_12'  -> 'PLATE'
      'WALL_2_3'  -> 'WALL_2'   (只去掉最后一个 _3；若要多级去掉，使用 *_loose)
    为了简洁与健壮，我们直接“一次性去掉所有结尾的 _数字段”：
      'WALL_2_3'  -> 'WALL' （更符合“同一族”的直觉）
    """
    c = _canon(name)
    return re.sub(r'(?:_\d+)+$', '', c)

def _family_base_loose(name: str) -> str:
    """
    Loose base: remove trailing '_<digits>' OR bare '<digits>' at end.
      'PLATE1'    -> 'PLATE'
      'A1'        -> 'A'     （仅作为兜底！）
    """
    c = _canon(name)
    return re.sub(r'(?:_?\d+)+$', '', c)

def _family_regex(base: str) -> re.Pattern:
    """
    Accept family object itself and any number of trailing '_<digits>' chunks.
      BASE, BASE_1, BASE_1_2, ...
    """
    b = _canon(base)
    # 允许 0 次或多次后缀；IGNORECASE
    return re.compile(rf'^{re.escape(b)}(?:_\d+)*$', re.IGNORECASE)

def _family_regexes_for_name(name: str) -> list[re.Pattern]:
    """
    Build a list of candidate regexes for the given name:
      1) strict:  base_strict + (?:_\\d+)*
      2) loose :  base_loose  + (?:_?\\d+)*   (fallback only)
    """
    base1 = _family_base_strict(name)
    base2 = _family_base_loose(name)
    pats = []
    if base1:
        pats.append(_family_regex(base1))                # ^BASE1(?:_\d+)*$
    if base2 and base2 != base1:
        # 宽松匹配允许无下划线的数字段：BASE2, BASE2_1, BASE2_12, BASE2_12_3, BASE2 1 -> 已在 _canon 中变成 BASE2_1
        pats.append(re.compile(rf'^{re.escape(_canon(base2))}(?:_?\d+)*$', re.IGNORECASE))
    return pats or [re.compile(r'^$', re.IGNORECASE)]

def _try_iter(col):
    try: return col[:]
    except Exception: return col

def _get_typename(o: Any) -> str:
    for k in ("TypeName", "typename", "type_name"):
        try:
            v = getattr(o, k, None)
            if v:
                return _ident_str(v).lower()
        except Exception:
            pass
    # Fallback: 类名或 repr 线索
    try:
        return (o.__class__.__name__ or "").lower()
    except Exception:
        return str(type(o)).lower()

def _match_rule_by_typename(typename: str) -> Optional[dict]:
    t = (typename or "").lower()
    for rule in FAMILY_REGISTRY:
        if re.search(rule["regex"], t):
            return rule
    return None

def _guess_type_tag(obj_or_handle: Any) -> Optional[str]:
    # If the handle can be obtained, directly determine it with TypeName.
    h = getattr(obj_or_handle, "plx_id", None) or obj_or_handle
    if h is not None:
        rule = _match_rule_by_typename(_get_typename(h))
        if rule:
            return rule["tag"]

    try:
        tag = CLASS_TAGS.get(obj_or_handle.__class__.__name__)
        if tag:
            return tag
    except Exception:
        pass
    return None

def _rule_by_tag(tag: Optional[str]) -> Optional[dict]:
    if not tag: 
        return None
    for r in FAMILY_REGISTRY:
        if r["tag"] == tag:
            return r
    return None

def _get_first_collection(g_i: Any, names: Sequence[str]) -> Optional[Any]:
    for n in names:
        try:
            c = getattr(g_i, n, None)
            if c:
                return c
        except Exception:
            pass
    return None

# --- helpers: flatten nested dicts---
def _flatten_options_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    def _rec(prefix: str, obj: Dict[str, Any]) -> None:
        for k, v in obj.items():
            kk = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _rec(kk, v)
            else:
                flat[kk] = v
    _rec("", d or {})
    return flat


class PhaseMapper:
    """
    A defensive, version-tolerant mapper for PLAXIS Staged Construction phases.

    Public API (Phase-first):
      - goto_stages(g_i) -> None
      - wrap_initial_as_phase(g_i, **opts) -> Phase
      - create(g_i, phase: Phase, inherits: Optional[Phase|Any]) -> Phase
      - apply_phase(g_i, phase: Phase, warn_on_missing=False) -> None
      - apply_plan(g_i, phases: Sequence[Phase], warn_on_missing=False) -> List[Phase]
      - update(g_i, phase: Phase, base: Optional[Phase|Any], warn_on_missing=False) -> Dict[str,Any]

    Compatibility helpers (internal use OK):
      - get_initial_phase(g_i) -> PhaseHandle
      - find_phase_handle(g_i, name: str) -> Optional[PhaseHandle]
    """

    # ---------------------------------------------------------------------
    # Stage navigation & discovery
    # ---------------------------------------------------------------------

    @staticmethod
    def goto_stages(g_i: Any) -> None:
        """Switch PLAXIS UI to Staged Construction; try common variants."""
        for fn in ("gotostages", "GoToStages", "goto_stages"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    f()
                    return
                except Exception:
                    pass
        # Some builds implicitly switch; do not hard-fail.

    @staticmethod
    def get_initial_phase(g_i: Any) -> Any:
        """
        Return the Initial Phase handle (robust).
        Tries: g_i.InitialPhase, first of g_i.Phases, or search by Identification.
        """
        PhaseMapper.goto_stages(g_i)
        # 1) direct property
        initial = getattr(g_i, "InitialPhase", None)
        if initial is not None:
            return initial

        # 2) first in phases collection
        try:
            phases = getattr(g_i, "Phases", None)
            if phases:
                try:
                    return phases[0]  # indexing
                except Exception:
                    for p in phases:
                        return p
        except Exception:
            pass

        # 3) best-effort name scan（通常不需要）
        try:
            for p in getattr(g_i, "Phases", []):
                name = getattr(p, "Identification", None) or getattr(p, "Name", None)
                if str(getattr(getattr(name, "value", name), "__str__", lambda: name)()) in ("Initial phase", "InitialPhase"):
                    return p
        except Exception:
            pass

        raise RuntimeError("Cannot resolve InitialPhase handle from g_i.")

    # --------------------- Wrap InitialPhase as a Phase domain object ---------------------

    @staticmethod
    def wrap_initial_as_phase(
        g_i: Any,
        *,
        name: str = "P0_Initial",
        comment: str = "Wrapped InitialPhase (Staged construction)",
        # 仅允许的少量设置（其余采用 PlasticStageSettings 的强类型默认值）
        ΣM_stage: float = 0.0,
        ΣM_weight: float = 1.0,
        pore_cal_type: PoreCalType | str = PoreCalType.Phreatic,
        tolerance_error: float = 1.0e-2,
        max_steps: int = 100,
        time_interval: float = 0.0,
        solver: SolverType = SolverType.PICO,
    ) -> Phase:
        """
        Build a Phase-domain object that *represents the existing InitialPhase*.
        Forcely let 'Staged construction'(PlasticStageSettings) + a few parameters.
        Returnned Phase will bind the plx_id=InitialPhase handle.
        """
        initial_handle = PhaseMapper.get_initial_phase(g_i)

        # 归一化孔压计算方式（字符串/枚举均可）
        if isinstance(pore_cal_type, str):
            pore_mode = PoreCalType.Phreatic if pore_cal_type.lower().startswith("phreatic") else PoreCalType.LastStage
        else:
            pore_mode = pore_cal_type

        settings = PlasticStageSettings(
            load_type=LoadType.StageConstruction,
            pore_cal_type=pore_mode,
            ΣM_stage=ΣM_stage,
            ΣM_weight=ΣM_weight,
            tolerance_error=tolerance_error,
            max_steps=max_steps,
            time_interval=time_interval,
            solver=solver,
        )

        phase = Phase(
            name=name,
            comment=comment,
            settings=settings,
            activate=[],
            deactivate=[],
            inherits=None,  # 初始相无继承
        )
        setattr(phase, "plx_id", initial_handle)
        return phase

    @staticmethod
    def wrap_initial_with_toggles(
        g_i: Any,
        *,
        name: str = "P0_Initial",
        comment: str = "Wrapped InitialPhase (Staged construction)",
        activate: Sequence[BaseStructure] = (),
        deactivate: Sequence[BaseStructure] = (),
        **kwargs,
    ) -> Phase:
        """语法糖：封装 InitialPhase 并附带激活/冻结清单。"""
        ph = PhaseMapper.wrap_initial_as_phase(g_i, name=name, comment=comment, **kwargs)
        if activate:
            ph.activate_structures(*activate)
        if deactivate:
            ph.deactivate_structures(*deactivate)
        return ph

    # Compatibility helper
    @staticmethod
    def find_phase_handle(g_i: Any, name: str) -> Optional[PhaseHandle]:
        """Find a phase by Identification/Name (best-effort)."""
        if not name:
            return None
        try:
            col = getattr(g_i, "Phases", None)
            if not col:
                return None
            try:
                for p in col[:]:
                    ident = getattr(p, "Identification", None) or getattr(p, "Name", None)
                    if ident == name:
                        return p
            except Exception:
                for p in col:
                    ident = getattr(p, "Identification", None) or getattr(p, "Name", None)
                    if ident == name:
                        return p
        except Exception:
            pass
        return None

    # ---------------------------------------------------------------------
    # Private low-level helpers (handle land; not public)
    # ---------------------------------------------------------------------

    @staticmethod
    def _create_phase_from_base(g_i: Any, previous: PhaseHandle) -> PhaseHandle:
        """Create a new phase that inherits from 'previous' (try several APIs)."""
        for fn in ("phase", "addphase", "Phase"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    return f(previous)
                except Exception:
                    pass

        # Empty create then set PreviousPhase/Parent
        for fn in ("phase", "addphase", "Phase"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    new_ph = f()
                    for attr in ("PreviousPhase", "Parent", "ParentPhase"):
                        try:
                            setattr(new_ph, attr, previous)
                            break
                        except Exception:
                            continue
                    return new_ph
                except Exception:
                    pass

        raise RuntimeError("No usable phase creation API found (phase/addphase/Phase).")

    @staticmethod
    def _maybe_set_identification(ph: PhaseHandle, name: Optional[str]) -> None:
        if not name:
            return
        for attr in ("Identification", "Name"):
            try:
                setattr(ph, attr, str(name))
                return
            except Exception:
                continue

    @staticmethod
    def _maybe_set_comment(ph: PhaseHandle, comment: Optional[str]) -> None:
        if not comment:
            return
        for attr in ("Comments", "Comment"):
            try:
                setattr(ph, attr, str(comment))
                return
            except Exception:
                continue

    @staticmethod
    def _ensure_phase_handle(g_i: Any, phase_obj: Phase, base: Optional[Phase | Any] = None) -> PhaseHandle:
        """
        Ensure a PLAXIS phase handle exists for the given Phase object.
        Rules:
        1. If phase_obj.plx_id already exists, return it directly.
        2. Otherwise, determine the base phase for inheritance: prioritize "base", followed by phase_obj.inherits, and finally fall back to "Initial".
        3. If the base phase for inh
        eritance is a Phase object and does not have a plx_id, first create a handle for it.
        4. Create a new phase and write the name/comment back to it.
        """
        h = getattr(phase_obj, "plx_id", None)
        if h is not None:
            return h

        base_src = base if base is not None else getattr(phase_obj, "inherits", None)

        base_handle = None
        if isinstance(base_src, Phase):
            base_handle = getattr(base_src, "plx_id", None)
            if base_handle is None:
                base_handle = PhaseMapper._ensure_phase_handle(g_i, base_src)
        elif base_src is not None:
            base_handle = base_src

        if base_handle is None:
            base_handle = PhaseMapper.get_initial_phase(g_i)

        new_ph = PhaseMapper._create_phase_from_base(g_i, base_handle)
        PhaseMapper._maybe_set_identification(new_ph, getattr(phase_obj, "name", None))
        PhaseMapper._maybe_set_comment(new_ph, getattr(phase_obj, "comment", None))
        setattr(phase_obj, "plx_id", new_ph)
        return new_ph

    # ---------------------------------------------------------------------
    # Public: create/apply using Phase objects
    # ---------------------------------------------------------------------

    @staticmethod
    def create(g_i: Any, phase: Phase, *, inherits: Optional[Phase | Any] = None,
            return_handle: bool = False, apply_structure: bool = True) -> Union[Phase, Any]:
        """
        Create a new PLAXIS phase for this Phase object (if needed), set Identification/Comment,
        bind handle back to phase.plx_id, and return the same Phase object.
        """
        PhaseMapper.goto_stages(g_i)
        base = inherits if inherits is not None else getattr(phase, "inherits", None)
        h = PhaseMapper._ensure_phase_handle(g_i, phase, base=base)
        if apply_structure:
            if (getattr(phase, "activate", None) or getattr(phase, "deactivate", None)):
                try:
                    PhaseMapper.apply_structures(g_i, phase, warn_on_missing=True)
                except Exception as e:
                    # Do not interrupt the process; errors will be reported again in the subsequent apply_phase.
                    print(f"Applying the structures in a phase is failed with the error message: {e}")
                    if return_handle:
                        return h
                    return phase
        
        if phase.water_table:
            try:
                WaterTableMapper.create_table(g_i, phase.water_table)
            except Exception as e:
                print(f"Create water table failed in phase {phase.name} with error message {e}")

        return h if return_handle else phase

    # ---- options/settings

    @staticmethod
    def _set_phase_attr(phase_handle: PhaseHandle, key: str, value: Any) -> bool:
        """
        Try to set phase attribute by key, with alias tweaks:
          - dotted keys 'Deform.ResetDisplacements'
          - Greek 'Σ' => 'Sum' / 'Sigma'
        """
        # dot path
        if "." in key:
            cur = phase_handle
            parts = key.split(".")
            try:
                for p in parts[:-1]:
                    cur = getattr(cur, p)
                setattr(cur, parts[-1], value)
                return True
            except Exception:
                return False

        candidates = [key, key.replace("Σ", "Sum"), key.replace("Σ", "Sigma")]
        for k in candidates:
            try:
                setattr(phase_handle, k, value)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def apply_options(*args, warn_on_missing: bool = False, **_ignored) -> None:
        """
        Dual-mode (compatible with both calling methods):
        1) New API (Phase-first): apply_options(g_i, phase: Phase)
        2) Old API (handle + options): apply_options(phase_handle, options_dict) 
        Function: Write the options to the target stage (via handle or phase.plx_id),
        Supporting dotted keys and 'Σ' alias.
        """
        # --------------- 判别形态 ---------------
        if len(args) == 2:
            a0, a1 = args
            # Case-B: 旧 API -> (phase_handle, options_dict)
            if isinstance(a1, dict):
                phase_handle = a0
                options = _flatten_options_dict(a1)
                for key, val in options.items():
                    if not PhaseMapper._set_phase_attr(phase_handle, key, val) and warn_on_missing:
                        print(f"[PhaseMapper.apply_options] Unknown/unsupported option '{key}' (ignored).")
                return

            # Case-A: 新 API -> (g_i, phase: Phase)
            g_i, phase = a0, a1
            # 确保有句柄
            ph_handle = PhaseMapper._ensure_phase_handle(g_i, phase)
            # 从 Phase 取 options
            options: Dict[str, Any] = {}
            if hasattr(phase, "settings_payload") and callable(getattr(phase, "settings_payload")):
                try:
                    options = phase.settings_payload() or {}
                except Exception:
                    options = {}
            elif hasattr(phase, "settings"):
                s = getattr(phase, "settings")
                if hasattr(s, "to_dict"):
                    try:
                        options = s.to_dict() or {}
                    except Exception:
                        options = {}
                elif isinstance(s, dict):
                    options = s
            options = _flatten_options_dict(options)
            for key, val in options.items():
                if not PhaseMapper._set_phase_attr(ph_handle, key, val) and warn_on_missing:
                    print(f"[PhaseMapper.apply_options] Unknown/unsupported option '{key}' (ignored).")
            return

        # 其它形态均视为用法错误
        raise TypeError("apply_options expects either (g_i, Phase) or (phase_handle, options_dict).")

    # ---- structures activation/deactivation
    @staticmethod
    def _resolve_handle(obj: Any) -> Optional[Any]:
        """Resolve a PLAXIS object handle from a domain object or a raw handle."""
        if obj is None:
            return None
        h = getattr(obj, "plx_id", None)
        if h is not None:
            return h
        return obj if PhaseMapper._looks_like_handle(obj) else None

    @staticmethod
    def _activate(g_i: Any, handle: Any, phase_handle: Any) -> bool:
        for fn in ("activate", "Activate"):
            f = getattr(g_i, fn, None)
            if callable(f):
                # 位置参数（你调试验证成功的调用方式）
                try:
                    f(handle, phase_handle)
                    return True
                except Exception:
                    pass
                # 关键字回退
                try:
                    f(handle, phase=phase_handle)
                    return True
                except Exception:
                    pass
        # 属性法回退
        try:
            active = getattr(handle, "Active", None)
            if active is not None:
                active[phase_handle] = True
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _deactivate(g_i: Any, handle: Any, phase_handle: Any) -> bool:
        for fn in ("deactivate", "Deactivate"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    f(handle, phase_handle)
                    return True
                except Exception:
                    pass
                try:
                    f(handle, phase=phase_handle)
                    return True
                except Exception:
                    pass
        try:
            active = getattr(handle, "Active", None)
            if active is not None:
                active[phase_handle] = False
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def apply_structures(g_i: Any,
                        phase_or_handle: Union[Phase, Any],
                        activate: Iterable[Any] = (),
                        deactivate: Iterable[Any] = (),
                        *, warn_on_missing: bool = False) -> None:
        """
        Dual-mode：
        1) New: apply_structures(g_i, phase: Phase, ...)
        - If activate/deactivate is not explicitly passed in, it will read phase.activate/phase.deactivate
        2) Old: apply_structures(g_i, phase_handle, activate=... , deactivate=...)
        """
        if isinstance(phase_or_handle, Phase):
            ph_handle = PhaseMapper._ensure_phase_handle(g_i, phase_or_handle)
            src_activate = list(activate or []) if activate else (getattr(phase_or_handle, "activate", None) or [])
            src_deactivate = list(deactivate or []) if deactivate else (getattr(phase_or_handle, "deactivate", None) or [])
        else:
            ph_handle = phase_or_handle
            src_activate = list(activate or [])
            src_deactivate = list(deactivate or [])

        # 1) 解析为基对象句柄（含按名兜底）
        act = [PhaseMapper._ensure_object_handle(g_i, o, warn_on_missing=warn_on_missing) for o in src_activate]
        deact = [PhaseMapper._ensure_object_handle(g_i, o, warn_on_missing=warn_on_missing) for o in src_deactivate]
        act = [h for h in act if h is not None]
        deact = [h for h in deact if h is not None]

        # 2) 🔹分块展开（关键：把 PLATE → 所有 PLATE_n 一起切换）
        act = PhaseMapper._expand_segments_for_handles(g_i, act)
        deact = PhaseMapper._expand_segments_for_handles(g_i, deact)

        # 3) 去重 + 冻结优先
        def _dedupe(seq: list[Any]) -> list[Any]:
            seen, out = set(), []
            for h in seq:
                k = id(h)
                if k not in seen:
                    seen.add(k)
                    out.append(h)
            return out
        act = _dedupe(act)
        deact = _dedupe(deact)
        deact_ids = {id(h) for h in deact}
        act = [h for h in act if id(h) not in deact_ids]

        # 4) 执行（先位置参数，再关键字，再属性法）
        for h in act:
            ok = PhaseMapper._activate(g_i, h, ph_handle)
            if not ok and warn_on_missing:
                name = _ident_str(getattr(h, "Identification", None)) or _ident_str(getattr(h, "Name", None))
                print(f"[PhaseMapper.apply_structures] activate failed for {name or h}.")
        for h in deact:
            ok = PhaseMapper._deactivate(g_i, h, ph_handle)
            if not ok and warn_on_missing:
                name = _ident_str(getattr(h, "Identification", None)) or _ident_str(getattr(h, "Name", None))
                print(f"[PhaseMapper.apply_structures] deactivate failed for {name or h}.")

    # =========================================================================
    # helper tools: Pick up objects by name from the stage phase
    # =========================================================================

    @staticmethod
    def _iter_candidate_collections(g_i: Any, type_tag: Optional[str] = None) -> List[Any]:
        cols: List[Any] = []
        # 先按族找容器
        rule = _rule_by_tag(type_tag)
        if rule:
            for n in rule["containers"]:
                try:
                    c = getattr(g_i, n, None)
                    if c:
                        _ = iter(c)
                        cols.append(c)
                except Exception:
                    pass
        # 兜底：全局常见容器
        if not cols:
            for n in [
                "Plates","Walls","Beams","EmbeddedBeams","EmbeddedPiles",
                "NodeToNodeAnchors","NodetoNodeAnchors","Anchors","Node_to_node_anchors",
                "Wells","Pipes","SoilVolumes","Soils","Structures","Objects",
            ]:
                try:
                    c = getattr(g_i, n, None)
                    if c:
                        _ = iter(c)
                        cols.append(c)
                except Exception:
                    pass
        return cols


    @staticmethod
    def _find_model_object_by_name(g_i: Any, name: str, *, type_tag: Optional[str] = None) -> Optional[Any]:
        if not name:
            return None

        # 1) 容器优先（你已有按 type_tag 限定的 _iter_candidate_collections，可保留）
        containers = PhaseMapper._iter_candidate_collections(g_i, type_tag=type_tag)

        # 2) 先做精确匹配
        targets = {_canon(name), _canon(name).replace("_", ""), name}
        for col in containers:
            it = _try_iter(col)
            for obj in it:
                ident = _ident_str(getattr(obj, "Identification", None)) or _ident_str(getattr(obj, "Name", None))
                if not ident:
                    continue
                cident = _canon(ident)
                if cident in targets or cident.replace("_", "") in targets:
                    # 对可能的“组”优先返回叶子
                    return (PhaseMapper._collect_children_from_attrs(obj, ("Children","Members","Items","SubObjects")) or [obj])[0]

        # 3) 家族兜底：使用健壮的家族匹配正则
        fam_pats = _family_regexes_for_name(name)  # ← 新的统一族匹配
        for col in containers:
            it = _try_iter(col)
            for obj in it:
                ident = _ident_str(getattr(obj, "Identification", None)) or _ident_str(getattr(obj, "Name", None))
                if not ident:
                    continue
                cident = _canon(ident)
                if any(pat.match(cident) for pat in fam_pats):
                    kids = PhaseMapper._collect_children_from_attrs(obj, ("Children","Members","Items","SubObjects"))
                    return kids[0] if kids else obj

        return None


    @staticmethod
    def _collect_family_in_containers(g_i: Any, containers: Sequence[str], base_name: str, suffix_regex: str) -> list[Any]:
        out: list[Any] = []
        pat = re.compile(rf"^{re.escape(_canon(base_name))}{suffix_regex}$", flags=re.I)
        for cn in containers:
            col = getattr(g_i, cn, None)
            if not col:
                continue
            for obj in _try_iter(col):
                ident = _ident_str(getattr(obj, "Identification", None)) or _ident_str(getattr(obj, "Name", None))
                if ident and pat.match(_canon(ident)):
                    out.append(obj)
        return out

    @staticmethod
    def _expand_segments_for_handle(g_i: Any, handle: Any) -> list[Any]:
        if handle is None:
            return []
        segs = [handle]

        # 1) 展开子对象（若存在）
        segs.extend(PhaseMapper._collect_children_from_attrs(handle, ("Children","Members","SubObjects","Items")))

        # 2) 家族聚合（基于 Identification/Name + 家族正则）
        ident = _ident_str(getattr(handle, "Identification", None)) or _ident_str(getattr(handle, "Name", None))
        if ident:
            fam_pats = _family_regexes_for_name(ident)
            # 在常见容器里按族扩展（不限定容器也行；为了性能你可以配合 type_tag 限定）
            for col in PhaseMapper._iter_candidate_collections(g_i, type_tag=_guess_type_tag(handle)):
                for obj in _try_iter(col):
                    oid = _ident_str(getattr(obj, "Identification", None)) or _ident_str(getattr(obj, "Name", None))
                    if oid and any(p.match(_canon(oid)) for p in fam_pats):
                        segs.append(obj)

        # 去重
        uniq, seen = [], set()
        for h in segs:
            k = id(h)
            if k not in seen:
                seen.add(k); uniq.append(h)
        return uniq


    @staticmethod
    def _expand_segments_for_handles(g_i: Any, handles: list[Any]) -> list[Any]:
        out: list[Any] = []
        for h in handles or []:
            out.extend(PhaseMapper._expand_segments_for_handle(g_i, h))
        # 去重
        uniq, seen = [], set()
        for h in out:
            k = id(h)
            if k not in seen:
                seen.add(k)
                uniq.append(h)
        return uniq

    @staticmethod
    def _collect_children_from_attrs(handle: Any, attrs: Sequence[str]) -> list[Any]:
        out: list[Any] = []
        for attr in attrs or ():
            try:
                col = getattr(handle, attr, None)
                if col:
                    try:
                        out.extend(col[:])
                    except Exception:
                        for x in col:
                            out.append(x)
            except Exception:
                continue
        return out

    @staticmethod
    def _extract_head_from_mapping(mp: Dict[str, Any]) -> Optional[float]:
        for k in ("head", "level", "z", "H", "h"):
            if k in mp:
                try:
                    return float(mp[k])
                except Exception:
                    return None
        return None

    @staticmethod
    def _extract_head(water_tbl: Any) -> Optional[float]:
        """Extract scalar head (z) from various payloads."""
        if isinstance(water_tbl, (int, float)):
            try:
                return float(water_tbl)
            except Exception:
                return None
        if isinstance(water_tbl, dict):
            return PhaseMapper._extract_head_from_mapping(water_tbl)
        for meth in ("to_payload", "to_dict", "as_dict"):
            fn = getattr(water_tbl, meth, None)
            if callable(fn):
                try:
                    mp = fn() or {}
                    if isinstance(mp, dict):
                        h = PhaseMapper._extract_head_from_mapping(mp)
                        if h is not None:
                            return h
                except Exception:
                    pass
        for k in ("head", "level", "z", "H", "h"):
            try:
                v = getattr(water_tbl, k, None)
                if v is not None:
                    return float(v)
            except Exception:
                continue
        return None

    @staticmethod
    def _apply_water_table(g_i: Any, phase: Phase, *, warn_on_missing: bool = False) -> None:
        """Best-effort: set scalar head on phase or call solver helper."""
        h = PhaseMapper._ensure_phase_handle(g_i, phase)
        head = PhaseMapper._extract_head(getattr(phase, "water_table", None))
        if head is not None:
            for attr in ("WaterLevel", "PhreaticLevel", "WaterLevelHead", "Head"):
                try:
                    setattr(h, attr, head)
                    return
                except Exception:
                    pass
            for fn_name in ("setwaterlevel", "SetWaterLevel", "set_water_level"):
                fn = getattr(g_i, fn_name, None)
                if callable(fn):
                    try:
                        fn(head, phase=h)
                        return
                    except Exception:
                        pass
            if warn_on_missing:
                print("[PhaseMapper] Could not set scalar head on phase; skipped.")
            return

        # object-level APIs (rare)
        for fn_name in ("setwaterlevelobj", "SetWaterLevelObject", "set_water_level_object"):
            fn = getattr(g_i, fn_name, None)
            if callable(fn):
                try:
                    fn(getattr(phase, "water_table", None), phase=h)
                    return
                except Exception:
                    pass
        if warn_on_missing:
            print("[PhaseMapper] No supported API found to apply water table object; skipped.")

    # ---- wells

    @staticmethod
    def _looks_like_handle(obj: Any) -> bool:
        return obj is not None and any(
            hasattr(obj, a) for a in ("Active", "Identification", "IsValid", "Phase", "Material")
        )

    @staticmethod
    def _ensure_object_handle(g_i: Any, obj: Any, *, warn_on_missing: bool = False) -> Optional[Any]:
        if obj is None:
            return None

        # 1) 已有句柄直接用（若是属性句柄，建议在 structuremapper.create 时就重绑为容器对象）
        h = getattr(obj, "plx_id", None)
        if h is not None:
            return h

        # 2) 按名字 + 族 查找
        type_tag = _guess_type_tag(obj)  # 会从 plx_id.TypeName 或类名 Anchor 推断 'n2n_anchor'
        name = getattr(obj, "name", None)
        if isinstance(name, str) and name:
            h = PhaseMapper._find_model_object_by_name(g_i, name, type_tag=type_tag)
            if h is not None:
                try: setattr(obj, "plx_id", h)
                except Exception: pass
                return h

        # 3) 退化：obj 本身像句柄
        if PhaseMapper._looks_like_handle(obj):
            return obj

        if warn_on_missing:
            print(f"[PhaseMapper] cannot resolve handle for object: {getattr(obj,'name',obj)}")
        return None


    @staticmethod
    def _dedupe_handles(handles: Iterable[Any]) -> list[Any]:
        seen, out = set(), []
        for h in handles or []:
            k = id(h)
            if k not in seen:
                seen.add(k)
                out.append(h)
        return out

    @staticmethod
    def _normalize_toggles(
        g_i: Any,
        activate: Iterable[Any],
        deactivate: Iterable[Any],
        *,
        warn_on_missing: bool = False
    ) -> tuple[list[Any], list[Any]]:
        """
        Normalized activation/freeze list:
        - All are converted to valid handles (with name-based fallback, and if successful, the plx_id is written back)
        - De-duplication
        - Mutual exclusion handling: If the same object is both activated and frozen -> prioritize freezing
        """
        act_handles, deact_handles = [], []

        for obj in (activate or []):
            h = PhaseMapper._ensure_object_handle(g_i, obj, warn_on_missing=warn_on_missing)
            if h is not None:
                act_handles.append(h)
        for obj in (deactivate or []):
            h = PhaseMapper._ensure_object_handle(g_i, obj, warn_on_missing=warn_on_missing)
            if h is not None:
                deact_handles.append(h)

        def _unique(seq: list[Any]) -> list[Any]:
            seen, out = set(), []
            for x in seq:
                k = id(x)
                if k not in seen:
                    seen.add(k)
                    out.append(x)
            return out

        act_handles = _unique(act_handles)
        deact_handles = _unique(deact_handles)

        deact_ids = {id(h) for h in deact_handles}
        act_handles = [h for h in act_handles if id(h) not in deact_ids]

        return act_handles, deact_handles

    @staticmethod
    def _set_object_param(g_i: Any, handle: Any, key: str, value: Any, phase_handle: Optional[PhaseHandle]) -> bool:
        """Try to set a parameter on a model object, optionally for a phase."""
        try:
            setattr(handle, key, value)
            return True
        except Exception:
            pass
        for fn in ("setparameter", "SetParameter", "set"):
            f = getattr(g_i, fn, None)
            if callable(f):
                try:
                    if phase_handle is not None:
                        f(handle, key, value, phase=phase_handle)
                    else:
                        f(handle, key, value)
                    return True
                except Exception:
                    continue
        return False

    @staticmethod
    def apply_well_overrides_dict(g_i: Any, phase: Phase,
                                  overrides: Dict[str, Dict[str, Any]],
                                  *, warn_on_missing: bool = False) -> None:
        """
        Apply well parameter overrides per phase:
            { "Well-1": {"q_well": 0.008, "h_min": -10.0, "well_type": "Extraction"} }
        """
        if not isinstance(overrides, dict):
            return

        phase_handle = PhaseMapper._ensure_phase_handle(g_i, phase)
        alias_map = {
            "q_well": ("q_well", "QWell", "qwell", "Discharge", "Q", "Qwell"),
            "h_min": ("h_min", "HMin", "HeadMin"),
            "well_type": ("well_type", "WellType", "Mode", "Type"),
        }

        for well_name, params in overrides.items():
            handle = PhaseMapper._find_model_object_by_name(g_i, well_name)
            if handle is None:
                if warn_on_missing:
                    print(f"[PhaseMapper.well_overrides] Well '{well_name}' not found; skipped.")
                continue

            for canonical, candidates in alias_map.items():
                value_present = None
                if canonical in params:
                    value_present = params[canonical]
                else:
                    for a in candidates:
                        if a in params:
                            value_present = params[a]
                            break
                if value_present is None:
                    continue

                applied = False
                for key_try in candidates:
                    if PhaseMapper._set_object_param(g_i, handle, key_try, value_present, phase_handle):
                        applied = True
                        break
                if not applied:
                    PhaseMapper._set_object_param(g_i, handle, canonical, value_present, phase_handle)

    @staticmethod
    def apply_well_overrides(g_i: Any, phase: Phase, *, warn_on_missing: bool = False) -> None:
        """Compatibility wrapper expecting an object with `.well_overrides`."""
        mapping = getattr(phase, "well_overrides", None)
        if isinstance(mapping, dict):
            PhaseMapper.apply_well_overrides_dict(g_i, phase, mapping, warn_on_missing=warn_on_missing)

    # ---------------------------------------------------------------------
    # High-level "apply phase" (Phase-first)
    # ---------------------------------------------------------------------

    @staticmethod
    def apply_phase(g_i: Any, phase: Phase, *, warn_on_missing: bool = False) -> None:
        """
        Apply one phase in four steps, using Phase object:
          1) options/settings
          2) water table (optional)
          3) structure activation/deactivation
          4) well overrides (optional)
        """
        PhaseMapper.goto_stages(g_i)
        PhaseMapper._ensure_phase_handle(g_i, phase)

        PhaseMapper.apply_options(g_i, phase, warn_on_missing=warn_on_missing)
        PhaseMapper._apply_water_table(g_i, phase, warn_on_missing=warn_on_missing)

        # 👉 强烈建议把 warn_on_missing=True 传下去
        PhaseMapper.apply_structures(g_i, phase, warn_on_missing=warn_on_missing)

        PhaseMapper.apply_well_overrides(g_i, phase, warn_on_missing=warn_on_missing)

    # ---------------------------------------------------------------------
    # Batch helpers (Phase-first)
    # ---------------------------------------------------------------------

    @staticmethod
    def apply_plan(g_i: Any, phases: Sequence[Phase], *, warn_on_missing: bool = False) -> List[Phase]:
        """
        Create and apply a chain of phases in order (each inherits from previous).
        Returns the *Phase objects* (after binding their plx_id).
        """
        if not phases:
            return []

        PhaseMapper.goto_stages(g_i)
        previous_handle = PhaseMapper.get_initial_phase(g_i)

        out: List[Phase] = []
        for ph in phases:
            # 优先读取 ph.inherits；没有则用链上前一个
            base_src = getattr(ph, "inherits", None)
            if base_src is None:
                base_handle = previous_handle
            elif isinstance(base_src, Phase):
                base_handle = getattr(base_src, "plx_id", None)
                if base_handle is None:
                    base_handle = PhaseMapper._ensure_phase_handle(g_i, base_src)
            else:
                base_handle = base_src  # 允许直接给句柄

            # 确保创建，并应用设置/结构/水位/井
            PhaseMapper._ensure_phase_handle(g_i, ph, base=base_handle)
            PhaseMapper.apply_phase(g_i, ph, warn_on_missing=warn_on_missing)

            previous_handle = getattr(ph, "plx_id", previous_handle)
            out.append(ph)
        return out

    @staticmethod
    def update(g_i: Any, phase: Phase, base: Optional[Phase | Any] = None,
               *, warn_on_missing: bool = False) -> Dict[str, Any]:
        """
        Idempotent update (Phase-first):
          - If phase has handle: apply on it.
          - Else create inheriting from `base` (or Initial), then apply.
        """
        report = {"created": False, "applied": False, "handle": None}

        h = getattr(phase, "plx_id", None)
        if h is None:
            PhaseMapper._ensure_phase_handle(g_i, phase, base=base)
            report["created"] = True

        try:
            PhaseMapper.apply_phase(g_i, phase, warn_on_missing=warn_on_missing)
            report["applied"] = True
        finally:
            report["handle"] = getattr(phase, "plx_id", None)
        return report
