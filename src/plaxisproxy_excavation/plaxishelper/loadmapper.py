# =========================================
# Geometry-aware LoadMapper (unified Point/Line/Surface)
# - Static: ensure geometry via GeometryMapper, create a PLAXIS handle, then set magnitudes directly
# - Dynamic: DO NOT create a new object; ensure `base` exists, write base static values,
#            then override with dynamic object's own static values if given, and finally
#            attach multipliers using LoadMultiplierKey.value (e.g., "Multiplierx")
# - Deletion: static -> delete handle; dynamic -> clear multipliers on base handle
# (All comments in English.)
# =========================================
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..structures.load import (
    DistributionType, SignalType,
    LoadMultiplier, LoadMultiplierKey,
    PointLoad, DynPointLoad,
    LineLoad,  DynLineLoad,
    SurfaceLoad, DynSurfaceLoad,
)
from ..geometry import Point, Line3D, Polygon3D
from .geometrymapper import GeometryMapper


# ------------------------------
# Compact logging helpers
# ------------------------------
def _one_line(msg: str) -> str:
    return " ".join(str(msg).split())

def _h_str(h: Any) -> str:
    try:
        if hasattr(h, "Name"):
            return f"<{getattr(h, 'Name', 'load')}>"
        return f"<{str(h)}>"
    except Exception:
        return "<None>"

def _enum_str(v: Any) -> str:
    if hasattr(v, "name"): return v.name
    if hasattr(v, "value"): return str(v.value)
    return str(v)

def _log(stage: str, kind: str, msg: str, handle: Any = None, extra: str = "") -> None:
    s = f"[{stage}][{kind}] {msg}"
    if handle is not None: s += f" handle={_h_str(handle)}"
    if extra: s += f" {extra}"
    print(_one_line(s), flush=True)

def _log_create(kind: str, msg: str, handle: Any, extra: str = "") -> None:
    _log("CREATE", kind, msg, handle, extra)

def _log_attach(kind: str, msg: str, handle: Any, extra: str = "") -> None:
    _log("ATTACH", kind, msg, handle, extra)

def _log_delete(kind: str, msg: str, handle: Any = None, extra: str = "") -> None:
    _log("DELETE", kind, msg, handle, extra)

def _log_error(kind: str, msg: str, exc: Exception) -> None:
    _log("ERROR", kind, f"{msg} msg={exc}")


# ------------------------------
# Low-level utility
# ------------------------------
def _normalize(v: Any) -> Any:
    if isinstance(v, (list, tuple)) and v:
        return v[0]
    return v

def _try_call(g_i: Any, names: Iterable[str], *args, **kwargs) -> Any:
    last: Optional[Exception] = None
    for nm in names:
        fn = getattr(g_i, nm, None) or getattr(g_i, nm.lower(), None) or getattr(g_i, nm.title(), None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
    if last: raise last
    raise AttributeError(f"None of {tuple(names)} exists on g_i")

def _set_many(h: Any, props: Dict[str, Any]) -> None:
    for k, v in (props or {}).items():
        # Skip Nones, let explicit zeros/False pass through
        if v is None:
            continue
        try:
            if hasattr(h, "setproperties"):
                h.setproperties(k, v); continue
        except Exception: pass
        # try:
        #     setattr(h, k, v); continue
        # except Exception: pass
        # try:
        #     if hasattr(h, "setproperty"):
        #         h.setproperty(k, v)
        # except Exception: pass

def _unset_many(h: Any, keys: Iterable[str]) -> None:
    """Best-effort 'clear' for properties that hold object refs (e.g., multipliers)."""
    for k in keys:
        ok = False
        for val in (None, 0, "", False):
            try:
                setattr(h, k, val); ok = True; break
            except Exception: pass
            try:
                if hasattr(h, "setproperty"):
                    h.setproperty(k, val); ok = True; break
            except Exception: pass
        if not ok:
            for cname in (f"Clear{k}", f"clear_{k}", "Clear", "Reset"):
                fn = getattr(h, cname, None)
                if callable(fn):
                    try: fn(); break
                    except Exception: pass

def _dist_str(d: Optional[DistributionType]) -> str:
    if d == DistributionType.LINEAR: return "Linear"
    if d == DistributionType.UNIFORM: return "Constant"
    if d == DistributionType.X_ALIGNED_INC: return "X aligned increment"
    if d == DistributionType.Y_ALIGNED_INC: return "Y aligned increment"
    if d == DistributionType.Z_ALIGNED_INC: return "Z aligned increment"
    if d == DistributionType.VECTOR_ALIGNED_INC: return "Vector aligned increment"
    if d == DistributionType.FREE_INCREMENT: return "Free increments"
    if d == DistributionType.PERPENDICULAR: return "Perpendicular"
    if d == DistributionType.PERPENDICULAR_VERT_INC: return "Perpendicular, vertical increment"
    return "Constant"

def _delete_handle(g_i: Any, h: Any) -> None:
    """Try to delete a PLAXIS object handle."""
    for nm in ("Delete", "delete", "Remove", "remove", "DeleteObject"):
        fn = getattr(g_i, nm, None) or getattr(h, nm, None)
        if callable(fn):
            try:
                if fn is getattr(g_i, nm, None):
                    fn(h)
                else:
                    fn()
                return
            except Exception:
                continue
    try:
        if hasattr(h, "__del__"):
            h.__del__()  # type: ignore
    except Exception:
        pass


# ------------------------------
# PROPERTY WRITERS (single place)
# ------------------------------
def _write_point_static(tgt: Any, src: PointLoad) -> None:
    """Write point static values Fx/Fy/Fz and Mx/My/Mz."""
    _set_many(tgt, {
        "Fx": src.Fx, "Fy": src.Fy, "Fz": src.Fz,
        "Mx": src.Mx, "My": src.My, "Mz": src.Mz,
    })
    if getattr(src, "name", None):
        _set_many(tgt, {"Name": src.name})

def _write_line_static(tgt: Any, src: LineLoad) -> None:
    """Write line static values (UNIFORM -> q*_start only; LINEAR -> q*_start & q*_end)."""
    if getattr(src, "name", None):
        _set_many(tgt, {"Name": src.name})
    _set_many(tgt, {"Distribution": _dist_str(src.distribution)})
    if src.distribution == DistributionType.UNIFORM:
        _set_many(tgt, {"qx_start": src.qx, "qy_start": src.qy, "qz_start": src.qz})
    else:
        _set_many(tgt, {
            "qx_start": src.qx, "qy_start": src.qy, "qz_start": src.qz,
            "qx_end": src.qx_end, "qy_end": src.qy_end, "qz_end": src.qz_end,
        })

def _write_surface_static(tgt: Any, src: SurfaceLoad) -> None:
    """Write surface static values (sigx/sigy/sigz + gradients / refs / vector)."""
    if getattr(src, "name", None):
        _set_many(tgt, {"Name": src.name})
    _set_many(tgt, {"Distribution": _dist_str(src.distribution)})

    # base magnitudes
    if src.distribution != DistributionType.PERPENDICULAR:
        _set_many(tgt, {"sigx": src.sigmax, "sigy": src.sigmay, "sigz": src.sigmaz})
    else:
        _set_many(tgt, {"sign_ref": src.sigmaz})

    grad = getattr(src, "grad", {}) or {}
    rp   = getattr(src, "ref_point", None)

    if src.distribution in (DistributionType.X_ALIGNED_INC, DistributionType.FREE_INCREMENT):
        _set_many(tgt, {"sigx_inc_x": grad.get("gx_x"),
                        "sigy_inc_x": grad.get("gy_x"),
                        "sigz_inc_x": grad.get("gz_x")})
    if src.distribution in (DistributionType.Y_ALIGNED_INC, DistributionType.FREE_INCREMENT):
        _set_many(tgt, {"sigx_inc_y": grad.get("gx_y"),
                        "sigy_inc_y": grad.get("gy_y"),
                        "sigz_inc_y": grad.get("gz_y")})
    if src.distribution in (DistributionType.Z_ALIGNED_INC, DistributionType.FREE_INCREMENT):
        _set_many(tgt, {"sigx_inc_z": grad.get("gx_z"),
                        "sigy_inc_z": grad.get("gy_z"),
                        "sigz_inc_z": grad.get("gz_z")})

    if src.distribution in (DistributionType.X_ALIGNED_INC,
                            DistributionType.Y_ALIGNED_INC,
                            DistributionType.Z_ALIGNED_INC,
                            DistributionType.VECTOR_ALIGNED_INC,
                            DistributionType.FREE_INCREMENT):
        if isinstance(rp, (tuple, list)) and len(rp) == 3:
            _set_many(tgt, {"x_ref": rp[0], "y_ref": rp[1], "z_ref": rp[2]})

    if src.distribution == DistributionType.VECTOR_ALIGNED_INC:
        _set_many(tgt, {"sigx_inc_V": grad.get("gx_v"),
                        "sigy_inc_V": grad.get("gy_v"),
                        "sigz_inc_V": grad.get("gz_v")})
        vec = getattr(src, "vector", None)
        if isinstance(vec, (tuple, list)) and len(vec) == 3:
            _set_many(tgt, {"Vector_x": vec[0], "Vector_y": vec[1], "Vector_z": vec[2]})


# ------------------------------
# LoadMapper
# ------------------------------
class LoadMapper:
    """Unified creation & deletion for Point/Line/Surface loads.

    Static:
      - Ensure geometry via GeometryMapper
      - Create PLAXIS handle
      - Write magnitudes directly to the created handle

    Dynamic:
      - Ensure `base` exists (create if needed)
      - Write base static values to the base handle
      - Override with dynamic object's own static values if provided
      - Attach multipliers using LoadMultiplierKey.value (e.g., "Multiplierx")
    """

    _POINT_LOAD: Tuple[str, ...] = ("PointLoad", "point_force", "load_point", "nodeload")
    _LINE_LOAD:  Tuple[str, ...] = ("LineLoad", "lineload", "line_load", "load_line")
    _SURF_LOAD:  Tuple[str, ...] = ("SurfLoad", "surfaceload", "surface_load", "load_surface", "areaload")

    # ---------- PUBLIC API ----------
    @staticmethod
    def create(g_i: Any, obj: Union[PointLoad, DynPointLoad, LineLoad, DynLineLoad, SurfaceLoad, DynSurfaceLoad]) -> Any:
        if isinstance(obj, (SurfaceLoad, DynSurfaceLoad)):
            return LoadMapper.create_surface(g_i, obj)
        if isinstance(obj, (LineLoad, DynLineLoad)):
            return LoadMapper.create_line(g_i, obj)
        if isinstance(obj, (PointLoad, DynPointLoad)):
            return LoadMapper.create_point(g_i, obj)
        raise TypeError(f"Unsupported load type: {type(obj)}")

    # ---------- Point ----------
    @staticmethod
    def create_point(g_i: Any, obj: Union[PointLoad, DynPointLoad]) -> Any:
        try:
            # Dynamic -> attach to base; write base static, then dynamic static overrides, then multipliers
            if isinstance(obj, DynPointLoad):
                return LoadMapper._attach_dynamic_point(g_i, obj)

            # Static
            if not obj.point.plx_id:
                GeometryMapper.create_point(g_i, obj.point)
            hp = getattr(obj.point, "plx_id", None)
            if hp is None:
                raise RuntimeError("Failed to create/reuse PLAXIS point.")
            h = _normalize(_try_call(g_i, LoadMapper._POINT_LOAD, hp))

            _write_point_static(h, obj)
            obj.plx_id = h
            _log_create("Load:Point", f"name={obj.name} geom_id={getattr(obj.point,'id','N/A')}", h,
                        extra=f"F=({obj.Fx},{obj.Fy},{obj.Fz}) M=({obj.Mx},{obj.My},{obj.Mz})")
            return h
        except Exception as e:
            _log_error("Load:Point", f"name={getattr(obj,'name','N/A')}", e)
            raise

    # ---------- Line ----------
    @staticmethod
    def create_line(g_i: Any, obj: Union[LineLoad, DynLineLoad]) -> Any:
        try:
            if isinstance(obj, DynLineLoad):
                return LoadMapper._attach_dynamic_line(g_i, obj)

            # Static
            if not obj.line.plx_id:
                GeometryMapper.create_line(g_i, obj.line)
            lh = getattr(obj.line, "plx_id", None)
            if lh is None:
                raise RuntimeError("Failed to create/reuse PLAXIS line.")
            handles = lh if isinstance(lh, list) else [lh]
            created: List[Any] = []

            for seg in handles:
                h = _normalize(_try_call(g_i, LoadMapper._LINE_LOAD, seg))
                _write_line_static(h, obj)
                created.append(h)

            obj.plx_id = created[0] if len(created) == 1 else created
            _log_create("Load:Line", f"name={obj.name} geom_id={getattr(obj.line,'id','N/A')}", obj.plx_id,
                        extra=f"dist={_enum_str(obj.distribution)} q=({obj.qx},{obj.qy},{obj.qz}) "
                              f"q_end=({obj.qx_end},{obj.qy_end},{obj.qz_end})")
            return obj.plx_id
        except Exception as e:
            _log_error("Load:Line", f"name={getattr(obj,'name','N/A')}", e)
            raise

    # ---------- Surface ----------
    @staticmethod
    def create_surface(g_i: Any, obj: Union[SurfaceLoad, DynSurfaceLoad]) -> Any:
        try:
            if isinstance(obj, DynSurfaceLoad):
                return LoadMapper._attach_dynamic_surface(g_i, obj)

            # Static
            if not obj.surface.plx_id:
                GeometryMapper.create_surface(g_i, obj.surface)
            hs = getattr(obj.surface, "plx_id", None)
            if hs is None:
                raise RuntimeError("Failed to create/reuse PLAXIS surface.")
            h = _normalize(_try_call(g_i, LoadMapper._SURF_LOAD, hs))

            _write_surface_static(h, obj)

            obj.plx_id = h
            _log_create("Load:Surface", f"name={obj.name} geom_id={getattr(obj.surface,'id','N/A')}", h,
                        extra=f"dist={_enum_str(obj.distribution)} sig=({obj.sigmax},{obj.sigmay},{obj.sigmaz})")
            return h
        except Exception as e:
            _log_error("Load:Surface", f"name={getattr(obj,'name','N/A')}", e)
            raise

    # ---------- Dynamic attachment: Point ----------
    @staticmethod
    def _attach_dynamic_point(g_i: Any, dyn: DynPointLoad) -> Any:
        base = dyn.base
        # ensure base exists; if not create as static
        if getattr(base, "plx_id", None) is None:
            LoadMapper.create_point(g_i, base)
        h = getattr(base, "plx_id", None)
        if h is None:
            raise RuntimeError("Failed to ensure base point load in PLAXIS.")
        tgt = getattr(h, "PointLoad", None)
        if tgt is None:
            raise RuntimeError("The PointLoad property does not exist.")
        # 1) write base static values
        _write_point_static(h, base)
        # 2) override with dynamic object's own static values (if provided)
        _write_point_static(tgt, dyn)

        # 3) attach multipliers using LoadMultiplierKey.value
        LoadMapper._bind_multipliers(tgt, dyn.mult)

        _log_attach("DynPointLoad", f"base={getattr(base,'name','<static>')} geom_id={getattr(base.point,'id','N/A')}",
                    tgt)
        dyn.plx_id = tgt  # dynamic returns the same base handle
        return tgt

    # ---------- Dynamic attachment: Line ----------
    @staticmethod
    def _attach_dynamic_line(g_i: Any, dyn: DynLineLoad) -> Any:
        base = dyn.base
        # ensure base exists; if not, creat
        # e as static
        if getattr(base, "plx_id", None) is None:
            LoadMapper.create_line(g_i, base)
        h = getattr(base, "plx_id", None)
        if h is None:
            raise RuntimeError("Failed to ensure base line load in PLAXIS.")
        tgt = getattr(h, "LineLoad", None)
        if tgt is None:
            raise RuntimeError("The LineLoad property does not exist.")
        # base line may be one handle or list of per-segment handles (mirror and write to each)
        handles = h if isinstance(h, list) else [h]

        for h in handles:
            # 1) write base static values
            _write_line_static(h, base)
            # 2) override with dynamic object's own static values (if provided)
            _write_line_static(tgt, dyn)
            # 3) attach multipliers on each handle
            LoadMapper._bind_multipliers(tgt, dyn.mult)

        _log_attach("DynLineLoad", f"base={getattr(base,'name','<static>')} geom_id={getattr(base.line,'id','N/A')}",
                    tgt)
        dyn.plx_id = tgt
        return tgt

    # ---------- Dynamic attachment: Surface ----------
    @staticmethod
    def _attach_dynamic_surface(g_i: Any, dyn: DynSurfaceLoad) -> Any:
        base = dyn.base
        # ensure base exists; if not, create as static
        if getattr(base, "plx_id", None) is None:
            LoadMapper.create_surface(g_i, base)
        h = getattr(base, "plx_id", None)
        if h is None:
            raise RuntimeError("Failed to ensure base surface load in PLAXIS.")
        tgt = getattr(h, "SurfaceLoad", None)
        if tgt is None:
            raise RuntimeError("The SurfaceLoad property does not exist.")
        # 1) write base static values
        _write_surface_static(h, base)
        # 2) override with dynamic object's own static values (if provided)
        _write_surface_static(tgt, dyn)
        # 3) attach multipliers (LoadMultiplierKey.value -> e.g., "Multiplierx")
        LoadMapper._bind_multipliers(tgt, dyn.mult)

        _log_attach("DynSurfaceLoad", f"base={getattr(base,'name','<static>')} geom_id={getattr(base.surface,'id','N/A')}",
                    tgt)
        dyn.plx_id = tgt
        return tgt

    # ---------- Bind multipliers by key.value ----------
    @staticmethod
    def _bind_multipliers(tgt: Any, mult: Optional[Dict[LoadMultiplierKey, LoadMultiplier]]) -> None:
        if not mult:
            return
        for k, m in mult.items():
            if m is None:
                continue
            # ensure multiplier created
            if not getattr(m, "plx_id", None):
                LoadMultiplierMapper.create(tgt, m)  # fallback if caller mistakenly passed handle instead of g_i
            # If previous line used tgt instead of g_i, fix: we actually need g_i.
            # Safer approach:
            g_i = None
            try:
                g_i = getattr(tgt, "_g_i", None) or getattr(tgt, "g_i", None)
            except Exception:
                g_i = None
            if getattr(m, "plx_id", None) is None and g_i is not None:
                LoadMultiplierMapper.create(g_i, m)

            # property name is exactly key.value (enum stores API property, e.g. "Multiplierx")
            prop_name = k.value if hasattr(k, "value") else str(k)
            _set_many(tgt, {prop_name: getattr(m, "plx_id", None)})

    # ---------- Delete (static/dynamic) ----------
    @staticmethod
    def delete(g_i: Any, obj: Union[PointLoad, DynPointLoad, LineLoad, DynLineLoad, SurfaceLoad, DynSurfaceLoad, Any]) -> None:
        try:
            # raw handle -> delete directly
            if not hasattr(obj, "__class__") or (hasattr(obj, "Name") and not hasattr(obj, "base")):
                _delete_handle(g_i, obj)
                _log_delete("Load", "deleted by raw handle", obj)
                return

            # dynamic -> clear multipliers on base
            if isinstance(obj, (DynSurfaceLoad, DynLineLoad, DynPointLoad)):
                base = obj.base
                # Ensure base exists
                if getattr(base, "plx_id", None) is None:
                    LoadMapper.create(g_i, base)
                tgt = getattr(base, "plx_id", None)
                if tgt is None:
                    return
                # clear known multiplier props
                multiplier_keys = []
                for key in LoadMultiplierKey:
                    multiplier_keys.append(key.value)
                _unset_many(tgt, multiplier_keys)
                _log_delete("DynLoad", f"cleared multipliers on base={getattr(base,'name','<static>')}", tgt)
                return

            # static -> delete handle
            h = getattr(obj, "plx_id", None)
            if h is None:
                _log_delete("Load", f"name={getattr(obj,'name','N/A')} no handle to delete")
                return
            _delete_handle(g_i, h)
            _log_delete(f"Load:{obj.__class__.__name__}", f"name={getattr(obj,'name','N/A')}", h)
            obj.plx_id = None
        except Exception as e:
            _log_error("Load:Delete", f"name={getattr(obj,'name','N/A')}", e)

    @staticmethod
    def delete_point(g_i: Any, obj: Union[PointLoad, DynPointLoad, Any]) -> None:
        LoadMapper.delete(g_i, obj)

    @staticmethod
    def delete_line(g_i: Any, obj: Union[LineLoad, DynLineLoad, Any]) -> None:
        LoadMapper.delete(g_i, obj)

    @staticmethod
    def delete_surface(g_i: Any, obj: Union[SurfaceLoad, DynSurfaceLoad, Any]) -> None:
        LoadMapper.delete(g_i, obj)


# ------------------------------
# LoadMultiplier mapper
# ------------------------------
class LoadMultiplierMapper:
    """Create/configure LoadMultiplier (Harmonic or Table)."""
    _GENERIC   = ("multiplier", "loadmultiplier", "create_multiplier", "createmultiplier")
    _HARMONIC  = ("harmonicmultiplier", "create_harmonic_multiplier")
    _TABLE     = ("tablemultiplier", "create_table_multiplier")

    @staticmethod
    def create(g_i: Any, mul: LoadMultiplier) -> Any:
        if getattr(mul, "plx_id", None):
            return mul.plx_id

        # try specialized creators first
        h = None
        if mul.signal_type == SignalType.HARMONIC:
            try: h = _try_call(g_i, LoadMultiplierMapper._HARMONIC)
            except Exception: h = None
        elif mul.signal_type == SignalType.TABLE:
            try: h = _try_call(g_i, LoadMultiplierMapper._TABLE)
            except Exception: h = None
        if h is None:
            h = _try_call(g_i, LoadMultiplierMapper._GENERIC)

        h = _normalize(h)
        mul.plx_id = h

        # common props
        if mul.name:
            _set_many(h, {"Name": mul.name})
        _set_many(h, {"Signal": mul.signal_type.value})

        # write details
        if mul.signal_type == SignalType.HARMONIC:
            _set_many(h, {"Amplitude": mul.amplitude, "Phase": mul.phase, "Frequency": mul.frequency})
            # also try lowercase keys if binding expects them
            _set_many(h, {"amplitude": mul.amplitude, "phase": mul.phase, "frequency": mul.frequency})
        else:
            # clear any existing table, then populate
            cleared = False
            for clear in ("Clear", "ClearAll", "ClearTable", "Reset"):
                fn = getattr(h, clear, None)
                if callable(fn):
                    try: fn(); cleared = True
                    except Exception: pass
            add_point = getattr(h, "AddPoint", None) or getattr(h, "addpoint", None)
            if callable(add_point):
                for t, v in (mul.table_data or []):
                    try: add_point(float(t), float(v))
                    except Exception: pass
            elif not cleared:
                # fallback: try setting a flat list or a known table field
                flat: List[float] = []
                for t, v in (mul.table_data or []):
                    flat.extend([float(t), float(v)])
                for fld in ("Table", "Points", "Data"):
                    try:
                        setattr(h, fld, flat)
                        break
                    except Exception:
                        continue

        _log_create("LoadMultiplier", f"name={mul.name}", h, extra=f"type={_enum_str(mul.signal_type)}")
        return h
