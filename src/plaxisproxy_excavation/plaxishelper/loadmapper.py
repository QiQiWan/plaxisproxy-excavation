# loadmapper.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# === 载荷与几何领域对象（按你的项目路径导入）===
from src.plaxisproxy_excavation.structures.load import (
    LoadStage, DistributionType,
    PointLoad, DynPointLoad,
    LineLoad, DynLineLoad,
    SurfaceLoad, DynSurfaceLoad,
    LoadMultiplier,
)
from src.plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D


# ----------------------------- low-level helpers -----------------------------
def _normalize_created_handle(created: Any) -> Any:
    if isinstance(created, (list, tuple)) and created:
        return created[0]
    return created

def _try_call(g_i: Any, names: Sequence[str], *args, **kwargs):
    last_err = None
    for n in names:
        fn = getattr(g_i, n, None)
        if not callable(fn):
            continue
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"None of {list(names)} worked. Last error: {last_err}")

def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
    """Prefer per-key setproperties(k, v); fallback to setattr/setproperty; skip None."""
    if not props:
        return
    for k, v in props.items():
        if v is None:
            continue
        try:
            if hasattr(plx_obj, "setproperties"):
                plx_obj.setproperties(k, v)
                continue
        except Exception:
            pass
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

def _try_delete_with_gi(g_i: Any, handle: Any) -> bool:
    if handle is None:
        return True
    for n in ("delete", "deleteobject", "Delete", "del"):
        fn = getattr(g_i, n, None)
        if not callable(fn):
            continue
        try:
            fn(handle)
            return True
        except Exception:
            continue
    return False

def _log_create(kind: str, msg: str, handle: Any) -> None:
    try:
        print(f"[CREATE] {kind}: {msg} -> {handle}")
    except Exception:
        pass

def _log_delete(kind: str, msg: str, handle: Any, ok: bool) -> None:
    try:
        print(f"[DELETE] {kind}: {msg} -> {'OK' if ok else 'FAIL'}")
    except Exception:
        pass


# ------------------------------ geometry ensure ------------------------------
def _ensure_point(g_i: Any, p: Point) -> Any:
    """Create point in PLAXIS if needed; return handle and write back plx_id."""
    if getattr(p, "plx_id", None) is not None:
        return p.plx_id
    # candidate constructors in various bindings
    created = _try_call(g_i, ("point", "createpoint", "Point", "point_3d"), p.x, p.y, p.z)
    created = _normalize_created_handle(created)
    try:
        p.plx_id = created
    except Exception:
        pass
    return created

def _extract_point_tuples_from_line(line: Line3D) -> List[Tuple[float, float, float]]:
    """best-effort extraction of 3D point tuples from Line3D/PointSet."""
    if hasattr(line, "as_tuple_list") and callable(getattr(line, "as_tuple_list")):
        return list(line.as_tuple_list())
    if hasattr(line, "points"):  # Line3D(PointSet([...])) pattern
        pts = getattr(line, "points")
        return [(float(pt.x), float(pt.y), float(pt.z)) for pt in pts]
    if hasattr(line, "pointset"):
        ps = getattr(line, "pointset")
        if hasattr(ps, "points"):
            return [(float(pt.x), float(pt.y), float(pt.z)) for pt in ps.points]
    raise TypeError("Cannot extract points from Line3D; implement as_tuple_list()/points on your class.")

def _ensure_line(g_i: Any, line: Line3D) -> Union[Any, List[Any]]:
    """Ensure a PLAXIS line (or polyline) exists. Returns handle (or list of segment handles)."""
    if getattr(line, "plx_id", None) is not None:
        return line.plx_id

    tuples = _extract_point_tuples_from_line(line)
    if len(tuples) < 2:
        raise ValueError("Line3D requires at least two points.")

    # 逐段创建，兼容 g_i.line(p1, p2) 风格
    handles: List[Any] = []
    for i in range(len(tuples) - 1):
        p1 = Point(*tuples[i])
        p2 = Point(*tuples[i + 1])
        h1 = _ensure_point(g_i, p1)
        h2 = _ensure_point(g_i, p2)
        seg_h = _try_call(g_i, ("line", "createline", "polyline", "Line"), h1, h2)
        handles.append(_normalize_created_handle(seg_h))

    # 如果只一段，返回单个句柄；否则返回段句柄列表
    ret: Union[Any, List[Any]] = handles[0] if len(handles) == 1 else handles
    try:
        line.plx_id = ret
    except Exception:
        pass
    return ret

def _extract_point_tuples_from_polygon(poly: Polygon3D) -> List[Tuple[float, float, float]]:
    """best-effort extraction of 3D point tuples for outer ring."""
    if hasattr(poly, "as_tuple_list") and callable(getattr(poly, "as_tuple_list")):
        return list(poly.as_tuple_list())
    if hasattr(poly, "outer"):
        outer = getattr(poly, "outer")
        if hasattr(outer, "points"):
            return [(float(pt.x), float(pt.y), float(pt.z)) for pt in outer.points]
    raise TypeError("Cannot extract points from Polygon3D; implement as_tuple_list()/outer.points.")

def _ensure_surface(g_i: Any, poly: Polygon3D) -> Any:
    """Ensure a PLAXIS polygon/surface exists and return its handle."""
    if getattr(poly, "plx_id", None) is not None:
        return poly.plx_id

    pts = _extract_point_tuples_from_polygon(poly)
    if len(pts) < 4:
        raise ValueError("Polygon3D requires at least 4 points (closed).")
    # 创建所有节点
    node_handles = [ _ensure_point(g_i, Point(*tp)) for tp in pts ]
    # 按不同绑定尝试：polygon(*pts) / polygon(list) / surface(...)
    created = None
    try:
        created = _try_call(g_i, ("polygon", "createpolygon", "surface", "createsurface"), *node_handles)
    except Exception:
        created = _try_call(g_i, ("polygon", "createpolygon", "surface", "createsurface"), node_handles)
    created = _normalize_created_handle(created)

    try:
        poly.plx_id = created
    except Exception:
        pass
    return created


# ================================== Mapper ===================================
class LoadMapper:
    """Create/delete point/line/surface loads with robust property mapping."""
    _POINT_CALLS = ("pointload", "point_force", "load_point", "nodeload")
    _LINE_CALLS  = ("lineload", "line_load",  "load_line")
    _SURF_CALLS  = ("surfaceload", "surface_load", "load_surface", "areaload")

    # ------------------------------- Point -----------------------------------
    @staticmethod
    def create_point(g_i: Any, obj: Union[PointLoad, DynPointLoad]) -> Any:
        hP = _ensure_point(g_i, obj.point)
        created = _normalize_created_handle(_try_call(g_i, LoadMapper._POINT_CALLS, hP))

        _set_many_props(created, {
            "Name": getattr(obj, "name", None),
            "Stage": getattr(getattr(obj, "stage", None), "value", None),
            "Distribution": getattr(getattr(obj, "distribution", None), "name", None),
            "Fx": getattr(obj, "Fx", None), "Fy": getattr(obj, "Fy", None), "Fz": getattr(obj, "Fz", None),
            "Mx": getattr(obj, "Mx", None), "My": getattr(obj, "My", None), "Mz": getattr(obj, "Mz", None),
        })

        # dynamic multipliers
        if isinstance(obj, DynPointLoad) and getattr(obj, "mult", None):
            for k, mul in obj.mult.items():
                lm_id = getattr(mul, "plx_id", None)
                if lm_id is None:
                    continue
                for key in (f"{k}Multiplier", f"Multiplier{k}", f"{k}_mult", f"{k}_Mul"):
                    _set_many_props(created, {key: lm_id})

        obj.plx_id = created
        _log_create("PointLoad", f"name={obj.name}", created)
        return created

    # -------------------------------- Line -----------------------------------
    @staticmethod
    def create_line(g_i: Any, obj: Union[LineLoad, DynLineLoad]) -> Any:
        hL = _ensure_line(g_i, obj.line)

        def _apply_props(handle: Any) -> None:
            base = {
                "Name": getattr(obj, "name", None),
                "Stage": getattr(getattr(obj, "stage", None), "value", None),
                "Distribution": getattr(getattr(obj, "distribution", None), "name", None),
                # q 映射到 Fx/Fy/Fz，并同时写 qx/qy/qz
                "Fx": getattr(obj, "qx", 0.0), "Fy": getattr(obj, "qy", 0.0), "Fz": getattr(obj, "qz", 0.0),
                "qx": getattr(obj, "qx", 0.0), "qy": getattr(obj, "qy", 0.0), "qz": getattr(obj, "qz", 0.0),
            }
            _set_many_props(handle, base)
            if getattr(obj, "distribution", None) == DistributionType.LINEAR:
                _set_many_props(handle, {
                    "Fx_end": getattr(obj, "qx_end", 0.0),
                    "Fy_end": getattr(obj, "qy_end", 0.0),
                    "Fz_end": getattr(obj, "qz_end", 0.0),
                    "qx_end": getattr(obj, "qx_end", 0.0),
                    "qy_end": getattr(obj, "qy_end", 0.0),
                    "qz_end": getattr(obj, "qz_end", 0.0),
                })

        # 尝试一次性创建；若抛异常则逐段创建已在 _ensure_line 完成
        try:
            created = _normalize_created_handle(_try_call(g_i, LoadMapper._LINE_CALLS, hL))
            targets = [created]
        except Exception:
            # 直接把已确保的线段句柄作为“已创建”
            targets = hL if isinstance(hL, list) else [hL]
            created = targets if len(targets) > 1 else targets[0]

        for h in targets:
            _apply_props(h)

        # 动力乘子
        if isinstance(obj, DynLineLoad) and getattr(obj, "mult", None):
            for comp in ("qx", "qy", "qz"):
                mul = obj.mult.get(comp)
                if mul is None:
                    continue
                lm_id = getattr(mul, "plx_id", None)
                if lm_id is None:
                    continue
                keys = (f"{comp}Multiplier", f"Multiplier{comp}", f"{comp}_mult", f"{comp}_Mul")
                alt = comp.replace("q", "F", 1)  # qx -> Fx
                alt_keys = (f"{alt}Multiplier", f"Multiplier{alt}", f"{alt}_mult", f"{alt}_Mul")
                for h in targets:
                    for k in (*keys, *alt_keys):
                        _set_many_props(h, {k: lm_id})

        obj.plx_id = created
        _log_create("LineLoad", f"name={obj.name} dist={obj.distribution.name}", created)
        return created

    # ------------------------------ Surface ----------------------------------
    @staticmethod
    def create_surface(g_i: Any, obj: Union[SurfaceLoad, DynSurfaceLoad]) -> Any:
        surf_h = getattr(obj.surface, "plx_id", None) or _ensure_surface(g_i, obj.surface)
        created = _normalize_created_handle(_try_call(g_i, LoadMapper._SURF_CALLS, surf_h))

        props: Dict[str, Any] = {
            "Name": getattr(obj, "name", None),
            "Stage": getattr(getattr(obj, "stage", None), "value", None),
            "Distribution": getattr(getattr(obj, "distribution", None), "name", None),
            # σ 同时写 SigmaX/Y/Z 与 Fx/Fy/Fz（不同绑定兼容）
            "Fx": getattr(obj, "sigmax", None),
            "Fy": getattr(obj, "sigmay", None),
            "Fz": getattr(obj, "sigmaz", None),
            "SigmaX": getattr(obj, "sigmax", None),
            "SigmaY": getattr(obj, "sigmay", None),
            "SigmaZ": getattr(obj, "sigmaz", None),
        }
        if getattr(obj, "distribution", None) == DistributionType.LINEAR:
            props.update({
                "Fx_end": getattr(obj, "sigmax_end", None),
                "Fy_end": getattr(obj, "sigmay_end", None),
                "Fz_end": getattr(obj, "sigmaz_end", None),
                "SigmaX_end": getattr(obj, "sigmax_end", None),
                "SigmaY_end": getattr(obj, "sigmay_end", None),
                "SigmaZ_end": getattr(obj, "sigmaz_end", None),
            })

        # gradients & reference point
        grad = getattr(obj, "grad", {}) or {}
        props.update({k: v for k, v in grad.items() if v is not None})
        refp = getattr(obj, "ref_point", None)
        if refp:
            props["RefPoint"] = refp

        _set_many_props(created, props)

        # 动力乘子（sigmax/sigmay/sigmaz + 兼容 Fx/Fy/Fz）
        if isinstance(obj, DynSurfaceLoad) and getattr(obj, "mult", None):
            for comp, mul in obj.mult.items():
                lm_id = getattr(mul, "plx_id", None)
                if lm_id is None:
                    continue
                for key in (f"{comp}Multiplier", f"Multiplier{comp}", f"{comp}_mult", f"{comp}_Mul"):
                    _set_many_props(created, {key: lm_id})
                alt = {"sigmax": "Fx", "sigmay": "Fy", "sigmaz": "Fz"}.get(comp)
                if alt:
                    for key in (f"{alt}Multiplier", f"Multiplier{alt}", f"{alt}_mult", f"{alt}_Mul"):
                        _set_many_props(created, {key: lm_id})

        obj.plx_id = created
        _log_create("SurfaceLoad", f"name={obj.name} dist={obj.distribution.name}", created)
        return created

    # ------------------------------ Delete -----------------------------------
    @staticmethod
    def delete(g_i: Any, obj_or_handle: Union[PointLoad, LineLoad, SurfaceLoad, DynPointLoad, DynLineLoad, DynSurfaceLoad, Any]) -> bool:
        obj_types = (PointLoad, LineLoad, SurfaceLoad, DynPointLoad, DynLineLoad, DynSurfaceLoad)
        obj = obj_or_handle if isinstance(obj_or_handle, obj_types) else None
        h = getattr(obj, "plx_id", None) if obj else obj_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj:
            obj.plx_id = None
        kind = obj.__class__.__name__ if obj else "Load"
        _log_delete(kind, f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok
