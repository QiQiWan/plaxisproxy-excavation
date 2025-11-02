# structure_mappers.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ===== 按你的项目结构调整这些导入 =====
from ..geometry import Point, Line3D, Polygon3D
from ..materials.beammaterial import ElasticBeam, ElastoplasticBeam, BeamType
from ..materials.pilematerial import ElasticPile, ElastoplasticPile
from ..materials.platematerial import ElasticPlate
from ..materials.anchormaterial import (
    AnchorType, ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor,
)
from ..structures.anchor import Anchor
from ..structures.beam import Beam
from ..structures.embeddedpile import EmbeddedPile
from ..structures.retainingwall import RetainingWall, PositiveInterface, NegativeInterface, PlateInterfaceBase
from ..structures.well import Well, WellType
from ..structures.load import (
    PointLoad, LineLoad, SurfaceLoad,
    DynPointLoad, DynLineLoad, DynSurfaceLoad,
    DistributionType,
)
from ..structures.soilblock import SoilBlock

# =============================================================================
# Logging (single-line)
# =============================================================================
def _format_handle(h: Any) -> str:
    if h is None:
        return "None"
    for k in ("Id", "ID", "id", "Guid", "GUID", "guid", "Name", "MaterialName", "Identification"):
        try:
            v = getattr(h, k, None)
            if v is not None:
                if hasattr(v, "value"):  # some bindings wrap values
                    v = v.value
                return f"{k}={v}"
        except Exception:
            continue
    s = str(h).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= 120 else (s[:117] + "...")

def _one_line(msg: str) -> str:
    return " ".join(str(msg).split())

def _log_create(kind: str, desc: str, handle: Any, extra: str = "") -> None:
    h = _format_handle(handle)
    msg = f"[CREATE][{kind}] {desc} handle={h}"
    if extra:
        msg += f" {extra}"
    print(_one_line(msg), flush=True)

def _log_delete(kind: str, desc: str, handle: Any, ok: bool, extra: str = "") -> None:
    h = _format_handle(handle)
    status = "OK" if ok else "FAIL"
    msg = f"[DELETE][{kind}] {desc} handle={h} result={status}"
    if extra:
        msg += f" {extra}"
    print(_one_line(msg), flush=True)

# =============================================================================
# Utilities
# =============================================================================
def _normalize_created_handle(created: Any) -> Any:
    if isinstance(created, (list, tuple)) and created:
        return created[-1]
    return created

def _try_call(g_i: Any, names: Sequence[str], *args, **kwargs) -> Any:
    """
    Try calling g_i.<name>(*args, **kwargs) in order; return first success.
    Raises last exception if all candidates failed, or RuntimeError if none found.
    """
    last_exc: Optional[Exception] = None
    for nm in names:
        fn = getattr(g_i, nm, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                continue
    if last_exc:
        raise last_exc
    raise RuntimeError(f"No callable found among {names}")

def _try_delete_with_gi(g_i: Any, plx_obj: Any) -> bool:
    if plx_obj is None:
        return False
    try:
        if hasattr(plx_obj, "delete") and callable(plx_obj.delete):
            plx_obj.delete()
            return True
    except Exception:
        pass
    for fn_name in ("delete", "delobject", "deletematerial", "delmaterial", "remove"):
        try:
            fn = getattr(g_i, fn_name, None)
            if callable(fn):
                fn(plx_obj)
                return True
        except Exception:
            continue
    return False

def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
    """Robust property setting with None-skip + bulk/per-key fallbacks."""
    if not props:
        return
    filtered = {k: v for k, v in props.items() if v is not None}

    # bulk
    if hasattr(plx_obj, "setproperties"):
        try:
            kv: List[Any] = []
            for k, v in filtered.items():
                kv.extend([k, v])
            if kv:
                plx_obj.setproperties(*kv)  # type: ignore[misc]
                return
        except Exception:
            pass

    # per-key
    for k, v in filtered.items():
        try:
            setattr(plx_obj, k, v)
            continue
        except Exception:
            pass
        try:
            if hasattr(plx_obj, "setproperty"):
                plx_obj.setproperty(k, v)
        except Exception:
            pass

def _first_non_none(*vals):
    """Return the first value that is not None; otherwise None."""
    for v in vals:
        if v is not None:
            return v
    return -1

# =============================================================================
# Name helpers
# =============================================================================

# ---- helpers for name & rebind (put near other helpers) ----
import re

def _ident_str(x):
    if x is None: return ""
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)

def _sync_final_name(obj, handle) -> str:
    """Read Identification/Name from handle and write back to obj.name."""
    final = _ident_str(getattr(handle, "Identification", None)) or _ident_str(getattr(handle, "Name", None))
    if final:
        try: obj.name = final
        except Exception: pass
    return final

def _canon(s: str) -> str:
    s = _ident_str(s).replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _family_base(name: str) -> str:
    c = _canon(name)
    m = re.match(r"^(.*?)(?:_)?\d+$", c)  # A1 / A1_1 -> A1
    return m.group(1) if m else c

def _first_nonempty(*vals):
    for v in vals:
        if v: return v
    return ""

def _get_first_collection(g_i, names):
    for n in names:
        try:
            c = getattr(g_i, n, None)
            if c: return c
        except Exception: pass
    return None

def _try_iter(col):
    try: return col[:]  # slice when possible
    except Exception: return col

def _anchor_children(handle):
    """Return child anchors if the handle is a group-like object; else []."""
    out = []
    for attr in ("Children", "Members", "Items", "SubObjects"):
        try:
            c = getattr(handle, attr, None)
            if c:
                for x in _try_iter(c):
                    out.append(x)
        except Exception: pass
    return out

def _rebind_n2n_anchor_handle(g_i, obj, created_handle):
    """
    Return the container 'leaf' anchor handle:
      1) find 'Node-to-node anchors' container
      2) match by final name family (A1 ⇄ A1_1)
      3) prefer leaf (child); if only group -> use group
    """
    final_name = _first_nonempty(
        getattr(obj, "name", None),
        _ident_str(getattr(created_handle, "Identification", None)),
        _ident_str(getattr(created_handle, "Name", None)),
    )
    fam = _family_base(final_name)
    col = _get_first_collection(g_i, ["NodeToNodeAnchors", "NodetoNodeAnchors", "Anchors", "Node_to_node_anchors"])
    if not col:
        return created_handle

    # pass 1: exact match on canon name
    for o in _try_iter(col):
        ident = _ident_str(getattr(o, "Identification", None)) or _ident_str(getattr(o, "Name", None))
        if not ident: continue
        if _canon(ident) == _canon(final_name):
            ch = _anchor_children(o)
            return ch[0] if ch else o

    # pass 2: family match (A1 ↔ A1_1, A1_2)
    pat = re.compile(rf"^{re.escape(fam)}(?:_\d+)?$", flags=re.I)
    for o in _try_iter(col):
        ident = _ident_str(getattr(o, "Identification", None)) or _ident_str(getattr(o, "Name", None))
        if not ident: continue
        if pat.match(_canon(ident)):
            ch = _anchor_children(o)
            return ch[0] if ch else o

    return created_handle


# =============================================================================
# Geometry helpers
# =============================================================================
def _ensure_point(g_i: Any, p: Point) -> Any:
    h = getattr(p, "plx_id", None)
    if h is not None:
        return h
    x, y, z = p.get_point()
    try:
        h = g_i.point(x, y, z)
    except TypeError:
        h = g_i.point((x, y, z))
    p.plx_id = h
    return h

def _ensure_surface(g_i: Any, polygon: Polygon3D) -> Any:
    """
    Ensure PLAXIS surface exists; assumes polygon points exist or will be created.
    Also writes back to polygon.plx_id.
    """
    pts = list(polygon.get_points())
    if len(pts) < 3:
        raise ValueError("Surface requires at least three points.")
    plx_pts = [_ensure_point(g_i, p) for p in pts]
    try:
        surf = g_i.surface(*plx_pts)
    except TypeError:
        coords: List[float] = []
        for p in pts:
            x, y, z = p.get_point()
            coords.extend([x, y, z])
        surf = g_i.surface(*coords)
    polygon.plx_id = surf
    return surf

def _assign_material(plx_obj: Any, mat_obj: Any) -> None:
    """
    Assign a (already-created) material handle to a structure.
    Tries a range of common property names.
    """
    if mat_obj is None:
        return
    h = getattr(mat_obj, "plx_id", None)
    if h is None:
        return
    candidates = (
        "Material", "BeamMaterial", "PlateMaterial",
        "N2NMaterial", "AnchorMaterial", "EmbeddedBeamMaterial",
    )
    for k in candidates:
        try:
            setattr(plx_obj, k, h)
            return
        except Exception:
            try:
                if hasattr(plx_obj, "setproperty"):
                    plx_obj.setproperty(k, h)
                    return
            except Exception:
                continue
    _set_many_props(plx_obj, {"Material": h})

# =============================================================================
# Interface helpers
# =============================================================================

def _set_interface_props(
        g_i: Any,
        *,
        parent_plate_h: Any,
        interface_h: Optional[Any],
        side_label: str,              # "PositiveInterface" / "NegativeInterface"
        cfg: PlateInterfaceBase,
    ) -> None:
        # 组装参数（与你截图字段一致）
        kv = {
            "MaterialMode": getattr(cfg.material_mode, "value", cfg.material_mode),
            "ApplyStrengthReduction": bool(cfg.apply_strength_reduction),
            "ActiveInFlow": bool(cfg.active_in_flow),
            "VirtualThicknessFactor": float(cfg.virtual_thickness_factor),
            # **(cfg.extra_props or {}),s
        }

        # 1) 优先对“界面句柄”设置
        if interface_h is not None:
            try:
                _set_many_props(interface_h, kv)
                return
            except Exception:
                pass  # 版本差异时退化

        # 2) 兜底：对“板.界面.属性”设置（与你截图命令一致）
        nested = {f"{side_label}.{k}": v for k, v in kv.items()}
        _set_many_props(parent_plate_h, nested)


# =============================================================================
# Anchor
# =============================================================================
class AnchorMapper:
    @staticmethod
    def create(g_i: Any, obj: "Anchor") -> Any:
        """
        Create a node-to-node Anchor from two points of obj.line.

        Keep it consistent with Beam:
          - Ensure both endpoints exist via _ensure_point()
          - Call tolerant constructor names with (ptA, ptB)
          - If PLAXIS returns [Line3D, Anchor], ALWAYS take the LAST as the structure
          - Do NOT delete helper line/points; optionally bind helper line back to obj.line.plx_id
          - Assign material/type based on obj.anchor_type (material object / enum / str / handle)
        """
        # 1) Endpoints
        pA, pB = obj.get_points()
        hA, hB = _ensure_point(g_i, pA), _ensure_point(g_i, pB)

        # 2) Create via tolerant entry names
        created_raw = _try_call(
            g_i,
            ("n2nanchor", "n2n_anchor", "node_to_node_anchor", "anchor", "cable", "create_anchor"),
            hA, hB
        )

        # 3) Extract handles: last = Anchor, first (if any) = helper Line3D
        line_h = None
        if isinstance(created_raw, (list, tuple)):
            if not created_raw:
                raise RuntimeError("PLAXIS returned empty result when creating Anchor.")
            if len(created_raw) >= 2:
                line_h = created_raw[0]
            anchor_h = created_raw[len(created_raw)-1]
        else:
            anchor_h = created_raw

        # 4) Naming
        _set_many_props(anchor_h, {"Name": getattr(obj, "name", None)})

        # 5) Material / type assignment
        t = getattr(obj, "anchor_type", None)
        assigned = False
        try:
            from src.plaxisproxy_excavation.materials.anchormaterial import (  # type: ignore
                ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor, AnchorType
            )
        except Exception:
            # Fallback guards if imports are not available at runtime
            ElasticAnchor = ElastoplasticAnchor = ElastoPlasticResidualAnchor = tuple()  # type: ignore
            from enum import Enum
            class _DummyEnum(Enum):  # type: ignore
                pass
            AnchorType = _DummyEnum  # type: ignore

        # 5.1 Material object → centralized assigner preferred
        if isinstance(t, (ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor)):
            try:
                _assign_material(anchor_h, t)
                assigned = True
            except Exception:
                mat_h = getattr(t, "plx_id", t)
                _set_many_props(anchor_h, {"Material": mat_h, "AnchorMaterial": mat_h, "Type": mat_h})
                assigned = True

        # 5.2 Enum-like
        if (not assigned) and isinstance(t, AnchorType):
            val = getattr(t, "value", str(t))
            _set_many_props(anchor_h, {"MaterialType": val, "Type": val, "AnchorType": val})
            assigned = True

        # 5.3 String label
        if (not assigned) and isinstance(t, str):
            _set_many_props(anchor_h, {"MaterialType": t, "Type": t, "AnchorType": t})
            assigned = True

        # 5.4 Direct handle fallback
        if (not assigned) and (t is not None):
            mat_h = getattr(t, "plx_id", t)
            _set_many_props(anchor_h, {"Material": mat_h, "AnchorMaterial": mat_h, "Type": mat_h})

        # 6) Bind runtime handles back to domain objects
        obj.plx_id = anchor_h
        if (line_h is not None) and hasattr(obj, "line") and (obj.line is not None):
            try:
                setattr(obj.line, "plx_id", line_h)
            except Exception:
                pass

        # 7) Logging (avoid dereferencing handles beyond necessity)
        _sync_final_name(obj, anchor_h)  # ✅ 新增
        obj.plx_id = _rebind_n2n_anchor_handle(g_i, obj, anchor_h)
        _log_create("Anchor", f"name={obj.name}", anchor_h)
        return anchor_h

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union["Anchor", Any]) -> bool:
        """
        Delete the Anchor structure only. Do NOT delete helper line/points here
        to mirror Beam behavior and avoid unintended cascading deletions.
        """
        obj = obj_or_handle if isinstance(obj_or_handle, Anchor) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("Anchor", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok


# =============================================================================
# Beam
# =============================================================================
class BeamMapper:
    @staticmethod
    def create(g_i: Any, obj: Beam) -> Any:
        pA, pB = obj.get_points()
        hA, hB = _ensure_point(g_i, pA), _ensure_point(g_i, pB)
        created = _try_call(g_i, ("beam", "createbeam", "line_beam"), hA, hB)
        created = _normalize_created_handle(created)

        _set_many_props(created, {"Name": getattr(obj, "name", None)})

        t = obj.beam_type
        if isinstance(t, (ElasticBeam, ElastoplasticBeam)):
            _assign_material(created, t)
        elif isinstance(t, BeamType):
            _set_many_props(created, {"MaterialType": t.value if hasattr(t, "value") else str(t)})

        obj.plx_id = created
        _log_create("Beam", f"name={obj.name}", created)
        return created

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[Beam, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, Beam) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("Beam", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

# =============================================================================
# Embedded Pile
# =============================================================================
class EmbeddedPileMapper:
    @staticmethod
    def create(g_i: Any, obj: EmbeddedPile) -> Any:
        pA, pB = obj.get_points()
        hA, hB = _ensure_point(g_i, pA), _ensure_point(g_i, pB)
        created = _try_call(g_i, ("embeddedbeam", "embedded_pile", "embeddedpile"), hA, hB)
        created = _normalize_created_handle(created)

        _set_many_props(created, {"Name": getattr(obj, "name", None)})

        t = obj.pile_type
        if isinstance(t, (ElasticPile, ElastoplasticPile)):
            _assign_material(created, t)
        else:
            _set_many_props(created, {"MaterialType": getattr(t, "value", t)})

        obj.plx_id = created
        _log_create("EmbeddedPile", f"name={obj.name}", created)
        return created

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[EmbeddedPile, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, EmbeddedPile) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("EmbeddedPile", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

# =============================================================================
# Retaining Wall (Plate)
# =============================================================================
class RetainingWallMapper:
    @staticmethod
    def create(g_i: Any, obj: RetainingWall,
               *,
               create_pos_interface: Optional[bool] = None,
               create_neg_interface: Optional[bool] = None,
               default_r_inter: Optional[float] = 0.67,
               default_active_in_flow: bool = False,
               default_virtual_thickness_factor_pos: float = 0.10,
               default_virtual_thickness_factor_neg: float = 0.11,
               default_apply_strength_reduction: bool = True
    ) -> Any:
        surf_h = getattr(obj.surface, "plx_id", None) or _ensure_surface(g_i, obj.surface)
        created = _try_call(g_i, ("plate", "createplate"), surf_h)
        created = _normalize_created_handle(created)

        _set_many_props(created, {"Name": getattr(obj, "name", None)})

        t = obj.plate_type
        if isinstance(t, ElasticPlate):
            _assign_material(created, t)
        else:
            _set_many_props(created, {"MaterialType": getattr(t, "value", t)})

        # ✅ 正确：plx_id 就是刚创建的句柄
        obj.plx_id = created

        # 创建界面
        pos_cfg = getattr(obj, "pos_interface", None)
        neg_cfg = getattr(obj, "neg_interface", None)

        if pos_cfg is None and create_pos_interface:
            pos_cfg = PositiveInterface(
                name=f"{obj.name}_PosInterface",
                active_in_flow=default_active_in_flow,
                virtual_thickness_factor=default_virtual_thickness_factor_pos,
                apply_strength_reduction=default_apply_strength_reduction,
                r_inter=default_r_inter,
            )
        if neg_cfg is None and create_neg_interface:
            neg_cfg = NegativeInterface(
                name=f"{obj.name}_NegInterface",
                active_in_flow=default_active_in_flow,
                virtual_thickness_factor=default_virtual_thickness_factor_neg,
                apply_strength_reduction=default_apply_strength_reduction,
                r_inter=default_r_inter,
            )

        # 逐侧创建 + 设置属性；收集 r_inter 候选
        r_candidates = []
        if pos_cfg:
            InterfaceMapper.create_positive(g_i, created, pos_cfg)
            if pos_cfg.r_inter is not None:
                r_candidates.append(pos_cfg.r_inter)

        if neg_cfg:
            InterfaceMapper.create_negative(g_i, created, neg_cfg)
            if neg_cfg.r_inter is not None:
                r_candidates.append(neg_cfg.r_inter)

        # 板级 R_inter（多数版本统一在板级；两侧都给时取更保守的较小值）
        r_use = min(r_candidates) if r_candidates else (default_r_inter if (pos_cfg or neg_cfg) else None)
        if r_use is not None:
            _set_many_props(created, {"Rinter": r_use})  # 若你版本键名不同，改为 {"R_inter": r_use} 或加键名映射

        # ✅ 回写最终存档名（WALL1 -> WALL_1）
        _sync_final_name(obj, created)

        _log_create("RetainingWall", f"name={obj.name}", created)
        return created

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[RetainingWall, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, RetainingWall) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("RetainingWall", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

class InterfaceMapper:
    @staticmethod
    def create_positive(g_i: Any, parent_plate_h: Any, obj: PositiveInterface) -> Optional[Any]:
        created = _try_call(g_i, ("posinterface", "positiveinterface", "CreatePositiveInterface"), parent_plate_h.Parent)
        ih = _normalize_created_handle(created)
        obj.plx_id = ih                    # ✅ 为界面对象写回 plx_id
        obj.parent_plate_id = parent_plate_h
        _set_interface_props(g_i, parent_plate_h=parent_plate_h, interface_h=ih, side_label="PositiveInterface", cfg=obj)
        return ih

    @staticmethod
    def create_negative(g_i: Any, parent_plate_h: Any, obj: NegativeInterface) -> Optional[Any]:
        created = _try_call(g_i, ("neginterface", "negativeinterface", "CreateNegativeInterface"), parent_plate_h.Parent)
        ih = _normalize_created_handle(created)
        obj.plx_id = ih
        obj.parent_plate_id = parent_plate_h
        _set_interface_props(g_i, parent_plate_h=parent_plate_h, interface_h=ih, side_label="NegativeInterface", cfg=obj)
        return ih


# =============================================================================
# Well
# =============================================================================
class WellMapper:
    @staticmethod
    def create(g_i: Any, obj: Well) -> Any:
        """
        Create a well from two end points of obj.line (TwoPointLineMixin).
        PLAXIS may return [Line3D, Well]; we pick the last as the actual Well.
        Only three parameters are mapped:
          - Behaviour/Type/Mode  <- obj.well_type
          - Q_well (|Q_well|)    <- obj.q_well
          - h_min                <- ONLY when Extraction
        """
        # 1) get endpoints from the domain object (no p1/p2 usage)
        pA, pB = obj.get_points()
        hA, hB = _ensure_point(g_i, pA), _ensure_point(g_i, pB)

        # 2) create in PLAXIS
        created_raw = _try_call(g_i, ("well", "flowwell", "gw_well", "well3d", "create_well"), hA, hB)

        # 3) pick handles: prefer the last as the well; keep first (line) if present
        line_h = None
        if isinstance(created_raw, (list,)):
            if not created_raw:
                raise RuntimeError("PLAXIS returned empty result when creating Well.")
            if len(created_raw) >= 2:
                line_h = created_raw[0]
            well_h = created_raw[-1]
        else:
            well_h = created_raw

        # 4) naming
        _set_many_props(well_h, {"Name": getattr(obj, "name", None)})

        # 5) behaviour/type
        behaviour_val = getattr(getattr(obj, "well_type", None), "value", getattr(obj, "well_type", None))
        _set_many_props(
            well_h,
            {
                "Behaviour": behaviour_val,     # UK spelling in many builds
            },
        )

        # 6) discharge (|Q_well|). If user didn't set, default 0.0 is ok.
        qv = float(getattr(obj, "q_well", 0.0))
        _set_many_props(
            well_h,
            {
                "Q": qv,
            },
        )

        # 7) h_min only for Extraction
        if str(behaviour_val).lower().endswith("extraction"):
            hv = float(getattr(obj, "h_min", 0.0))
            _set_many_props(
                well_h,
                {
                    "Hmin": hv,
                },
            )

        # 8) write back handles
        if line_h is not None and hasattr(obj, "line") and obj.line is not None:
            try:
                setattr(obj.line, "plx_id", line_h)
            except Exception:
                pass
        obj.plx_id = well_h

        _log_create("Well", f"name={obj.name} type={behaviour_val} Q={qv}", well_h, extra=(f" line={_format_handle(line_h)}" if line_h else ""))
        return well_h

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[Well, Any]) -> bool:
        """
        Delete the well object only (do not delete the auxiliary Line3D).
        """
        obj = obj_or_handle if isinstance(obj_or_handle, Well) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        _log_delete("Well", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok
    
# =============================================================================
# SoilBlock (best-effort volume creation)
# =============================================================================
class SoilBlockMapper:
    """Best-effort soil volume creation; bindings differ a lot across versions."""
    @staticmethod
    def _extrude_surface(g_i: Any, surf_h: Any, vec: Tuple[float, float, float]) -> Any:
        """
        Try common extrude entrypoints and signatures:
          extrude(surface, dx, dy, dz)  or  extrude(surface, (dx,dy,dz))
        """
        names = ("extrude", "extrusion", "extrude_surface", "extrudesurface")
        last_exc: Optional[Exception] = None
        for nm in names:
            fn = getattr(g_i, nm, None)
            if not callable(fn):
                continue
            # 1) surface, dx,dy,dz
            try:
                obj = fn(surf_h, vec[0], vec[1], vec[2])
                return _normalize_created_handle(obj)
            except Exception as e:
                last_exc = e
            # 2) surface, (dx,dy,dz)
            try:
                obj = fn(surf_h, tuple(vec))
                return _normalize_created_handle(obj)
            except Exception as e:
                last_exc = e
        if last_exc:
            raise last_exc
        raise RuntimeError("No extrude-like constructor found on g_i.")

    @staticmethod
    def create(g_i: Any, obj: SoilBlock) -> Any:
        geom = obj.geometry
        if geom is None:
            raise ValueError("SoilBlock geometry is None.")

        # ---------- A) Polygon3D -> Surface -> Extrude to volume ----------
        if isinstance(geom, Polygon3D):
            # 1) ensure surface exists
            surf_h = _first_non_none(getattr(geom, "plx_id", None), None)
            if surf_h is None:
                surf_h = _ensure_surface(g_i, geom)

            # 2) decide extrude vector
            vec = getattr(obj, "extrude_vec", None)
            if vec is not None:
                try:
                    dx, dy, dz = float(vec[0]), float(vec[1]), float(vec[2])
                    extrude_vec = (dx, dy, dz)
                except Exception:
                    raise TypeError("SoilBlock.extrude_vec must be a 3-tuple of numbers.")
            else:
                # fallback thickness: prefer height/thickness/depth, else default -5.0m in Z
                H = _first_non_none(
                    getattr(obj, "height", None),
                    getattr(obj, "thickness", None),
                    getattr(obj, "depth", None),
                    -5.0,
                )
                try:
                    H = float(H)
                except Exception:
                    H = -5.0
                # 默认向下挤出
                extrude_vec = (0.0, 0.0, H)

            # 3) extrude and assign material
            created = SoilBlockMapper._extrude_surface(g_i, surf_h, extrude_vec)
            _assign_material(created, getattr(obj, "material", None))
            setattr(obj, "plx_id", created)
            _sync_final_name(obj, created)   # ✅ 可选
            _log_create("SoilBlock", f"name={obj.name} extrude={extrude_vec}", created)
            return created

        # ---------- B) Already-closed volume (rare) -> try soil(...) ----------
        # 如果你的几何类型本身就是封闭体（例如自定义 Polyhedron 并能返回各包围面），
        # 你可以把它转换为一组 surface 句柄后传入 soil(...)。此处保留原来的 best-effort 调用。
        try:
            created = _try_call(g_i, ("soil", "create_soil", "volume", "solid"), geom)
            created = _normalize_created_handle(created)
            _assign_material(created, getattr(obj, "material", None))
            setattr(obj, "plx_id", created)
            _log_create("SoilBlock", f"name={obj.name} via=soil()", created)
            return created
        except Exception as e:
            raise RuntimeError(
                "SoilBlock creation failed. For Polygon3D please use extrusion; "
                "for closed volumes ensure you pass a set of bounding surfaces compatible with soil()."
            ) from e

    @staticmethod
    def update(g_i: Any, obj_or_handle: Any, *, 
               q_well: Optional[float] = None,
               h_min: Optional[float] = None,
               well_type: Optional[Any] = None) -> Any:
        """
        Update parameters on an existing PLAXIS Well handle.
        Prefer PhaseMapper.apply_well_overrides for per-phase logic.
        """
        h = getattr(obj_or_handle, "plx_id", None) if hasattr(obj_or_handle, "plx_id") else obj_or_handle
        if h is None:
            raise ValueError("Well handle is None.")
        if q_well is not None:
            _set_many_props(h, {"Qwell": float(q_well), "Q": float(q_well), "Discharge": float(q_well)})
        if h_min is not None:
            _set_many_props(h, {"Hmin": float(h_min), "HeadMin": float(h_min)})
        if well_type is not None:
            sval = getattr(well_type, "value", well_type)
            _set_many_props(h, {"WellType": str(sval), "Type": str(sval), "Behaviour": str(sval)})
        return h

    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[SoilBlock, Any]) -> bool:
        obj = obj_or_handle if isinstance(obj_or_handle, SoilBlock) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            setattr(obj, "plx_id", None)
        _log_delete("SoilBlock", f"name={getattr(obj,'name','raw')}", h, ok=ok)
        return ok