# =========================================
# file: watertablemapper.py
# Mapper for WaterLevel / WaterLevelTable (PLAXIS 3D)
#
# - Geometry-aware: reuses GeometryMapper to create/reuse Point geometry
#   before building a UserWaterLevel surface.
# - Create:
#     * Flow mode (best-effort) -> create a UserWaterLevel using coordinates
#       of WaterLevel points (x, y, z).
#     * Set name/label when available.
# - Update:
#     * Try to move existing WL points via 'movepoint' (robust fallback to
#       delete+create if not supported in current build).
# - Delete:
#     * Remove an existing UserWaterLevel handle.
#
# Notes:
# * PLAXIS exposes water levels as "UserWaterLevel" objects in 3D; they are
#   defined by a 3D polygon-like list of points.
# * The remote scripting wrapper typically accepts Python tuples for points:
#     g_i.waterlevel( (x1,y1,z1), (x2,y2,z2), ... )
# * Property names and method names differ slightly per version; the mapper
#   tries multiple candidates for creation/rename/delete/move.
#
# All logs are concise one-liners to stdout.
# =========================================
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..geometry import Point
from ..core.plaxisobject import PlaxisObject
from ..components.watertable import WaterLevel, WaterLevelTable  # adjust the path if needed
from .geometrymapper import GeometryMapper


# ##############################
# Logging (compact)
# ##############################
def _one_line(msg: str) -> str:
    return " ".join(str(msg).split())

def _h_str(h: Any) -> str:
    try:
        if hasattr(h, "Name"):  # PLAXIS COM-like objects often have Name
            return f"<{getattr(h, 'Name', 'obj')}>"
        return f"<{str(h)}>"
    except Exception:
        return "<None>"

def _log(stage: str, kind: str, msg: str, handle: Any = None, extra: str = "") -> None:
    s = f"[{stage}][{kind}] {msg}"
    if handle is not None: s += f" handle={_h_str(handle)}"
    if extra: s += f" {extra}"
    print(_one_line(s), flush=True)

def _log_create(kind: str, msg: str, handle: Any, extra: str = "") -> None:
    _log("CREATE", kind, msg, handle, extra)

def _log_update(kind: str, msg: str, handle: Any, extra: str = "") -> None:
    _log("UPDATE", kind, msg, handle, extra)

def _log_delete(kind: str, msg: str, handle: Any = None, extra: str = "") -> None:
    _log("DELETE", kind, msg, handle, extra)

def _log_error(kind: str, msg: str, exc: Exception) -> None:
    _log("ERROR", kind, f"{msg} msg={exc}")


# ##############################
# Low-level helpers
# ##############################
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
        try:
            if hasattr(h, "setproperties"):
                h.setproperties(k, v); continue
        except Exception: pass
        try:
            setattr(h, k, v); continue
        except Exception: pass
        try:
            if hasattr(h, "setproperty"):
                h.setproperty(k, v)
        except Exception: pass

def _delete_handle(g_i: Any, h: Any) -> None:
    """Try to delete a PLAXIS object handle."""
    for nm in ("Delete", "delete", "Remove", "remove", "DeleteObject"):
        fn = getattr(g_i, nm, None) or getattr(h, nm, None)
        if callable(fn):
            try:
                if fn is getattr(g_i, nm, None):
                    fn(h)       # preferred: g_i.Delete(h)
                else:
                    fn()        # fallback: h.Delete()
                return
            except Exception:
                continue
    try:
        if hasattr(h, "__del__"):
            h.__del__()  # type: ignore
    except Exception:
        pass

def _ensure_flow_mode(g_i: Any) -> None:
    """Best-effort switch to Flow mode (3D)."""
    for nm in ("gotoflow", "GotoFlow", "goto_flow"):
        fn = getattr(g_i, nm, None)
        if callable(fn):
            try:
                fn(); return
            except Exception:
                pass
    # If not available, continue silently; some versions don't require this.


# ##############################
# Water table mapper
# ##############################
class WaterTableMapper:
    """
    Create / update / delete water levels (UserWaterLevel) in PLAXIS 3D.

    Workflow:
    - Create each WaterLevel point's geometry via GeometryMapper (for consistency & logs).
    - Call waterlevel(...) with tuples (x,y,z) to build a UserWaterLevel.
    - Store handle on WaterLevelTable.plx_id.
    """

    # candidates for creation
    _CREATE_WL: Tuple[str, ...] = ("waterlevel", "addwaterpoint", "create_waterlevel", "userwaterlevel", "WaterLevel")
    # candidates for renaming
    _RENAME_KEYS: Tuple[str, ...] = ("Name", "Identification", "Label")
    # candidates for movepoint (3D variant usually accepts index + (x y z))
    _MOVEPOINT: Tuple[str, ...] = ("movepoint", "MovePoint", "move_point")

    # ########## point (optional helper) ##########
    @staticmethod
    def create_level_point(g_i: Any, lvl: WaterLevel) -> Any:
        """Ensure PLAXIS point exists for this WaterLevel (not strictly required by waterlevel())."""
        try:
            if getattr(lvl, "plx_id", None) is None:
                GeometryMapper.create_point(g_i, lvl)  # WaterLevel inherits Point
                _log_create("WaterLevelPoint",
                            f"label={getattr(lvl,'label',None)} t={getattr(lvl,'time',None)} "
                            f"xyz=({lvl.x},{lvl.y},{lvl.z})",
                            getattr(lvl, "plx_id", None))
            return getattr(lvl, "plx_id", None)
        except Exception as e:
            _log_error("WaterLevelPoint", f"label={getattr(lvl,'label',None)}", e)
            raise

    # ########## create table ##########
    @staticmethod
    def create_table(g_i: Any, tbl: WaterLevelTable, *, goto_flow: bool = False) -> Any:
        """
        Create a new UserWaterLevel from the points in `tbl.levels`.
        Stores the returned handle as `tbl.plx_id` and sets its name/label if provided.
        """
        try:
            if goto_flow:
                _ensure_flow_mode(g_i)

            levels = list(tbl.levels or [])
            if len(levels) < 3:
                raise ValueError("WaterLevelTable requires at least 3 points to define a 3D surface.")

            # Ensure geometry for each level point (for logging/consistency)
            for lvl in levels:
                # WaterTableMapper.create_level_point(g_i, lvl)

                # Build tuple list for the command: ((x,y,z), (x,y,z), ...)
                pt = tuple((float(lvl.x), float(lvl.y), float(lvl.z), 0))

                # Create UserWaterLevel object
                h = _normalize(_try_call(g_i, WaterTableMapper._CREATE_WL, *pt))

            # Rename if label present
            label = getattr(tbl, "label", None)
            if label:
                for key in WaterTableMapper._RENAME_KEYS:
                    try:
                        _set_many(h, {key: label})
                        break
                    except Exception:
                        continue

            tbl.plx_id = h
            _log_create("WaterLevelTable",
                        f"label={label} npts={len(levels)}",
                        h,
                        extra=f"first=({levels[0].x},{levels[0].y},{levels[0].z})")
            return h
        except Exception as e:
            _log_error("WaterLevelTable", f"label={getattr(tbl,'label',None)}", e)
            raise

    # ########## update table (move points or rebuild) ##########
    @staticmethod
    def update_table(g_i: Any, tbl: WaterLevelTable, *, rebuild_if_needed: bool = True) -> Any:
        """
        Try to move the existing WL points to match current `tbl.levels`.
        If 'movepoint' is not available, optionally delete + re-create.
        """
        try:
            h = getattr(tbl, "plx_id", None)
            if h is None:
                # Not created yet -> create
                return WaterTableMapper.create_table(g_i, tbl)

            levels = list(tbl.levels or [])
            if len(levels) < 3:
                raise ValueError("WaterLevelTable requires at least 3 points to update.")

            # Try movepoint API
            moved = False
            move_fn = None
            for nm in WaterTableMapper._MOVEPOINT:
                move_fn = getattr(g_i, nm, None) or getattr(h, nm, None)
                if callable(move_fn):
                    break

            if callable(move_fn):
                for idx, lvl in enumerate(levels):
                    # ensure geometry point (optional)
                    WaterTableMapper.create_level_point(g_i, lvl)
                    # movepoint <WL> <index> (x y z)
                    try:
                        move_fn(h, idx, (float(lvl.x), float(lvl.y), float(lvl.z))) \
                            if move_fn is getattr(g_i, getattr(move_fn, "__name__", ""), None) \
                            else move_fn(idx, (float(lvl.x), float(lvl.y), float(lvl.z)))
                        moved = True
                    except Exception:
                        # try alternative calling convention
                        try:
                            move_fn(h, idx, float(lvl.x), float(lvl.y), float(lvl.z))
                            moved = True
                        except Exception:
                            pass

            if moved:
                _log_update("WaterLevelTable", f"moved {len(levels)} points", h)
                # ensure label is up to date
                label = getattr(tbl, "label", None)
                if label:
                    for key in WaterTableMapper._RENAME_KEYS:
                        try:
                            _set_many(h, {key: label}); break
                        except Exception: pass
                return h

            # Fallback -> rebuild
            if rebuild_if_needed:
                _log_update("WaterLevelTable", "movepoint not available -> rebuild", h)
                WaterTableMapper.delete_table(g_i, tbl)
                return WaterTableMapper.create_table(g_i, tbl)

            _log_update("WaterLevelTable", "no changes applied", h)
            return h

        except Exception as e:
            _log_error("WaterLevelTable", f"label={getattr(tbl,'label',None)}", e)
            raise

    # ########## delete ##########
    @staticmethod
    def delete_table(g_i: Any, tbl: WaterLevelTable) -> None:
        try:
            h = getattr(tbl, "plx_id", None)
            if h is None:
                _log_delete("WaterLevelTable", f"label={getattr(tbl,'label',None)} no handle")
                return
            _delete_handle(g_i, h)
            _log_delete("WaterLevelTable", f"label={getattr(tbl,'label',None)}", h)
            tbl.plx_id = None
        except Exception as e:
            _log_error("WaterLevelTable", f"label={getattr(tbl,'label',None)}", e)

    # ########## convenience: create or update ##########
    @staticmethod
    def create_or_update(g_i: Any, tbl: WaterLevelTable) -> Any:
        """Create if needed, else update."""
        return WaterTableMapper.update_table(g_i, tbl) if getattr(tbl, "plx_id", None) else WaterTableMapper.create_table(g_i, tbl)

    # ########## optional: set as global water level (best-effort) ##########
    @staticmethod
    def set_global(g_i: Any, tbl: Union[WaterLevelTable, Any]) -> None:
        """
        Try to assign this water level as 'Global Water Level'.
        This is version-dependent; we attempt several targets/properties.
        """
        h = getattr(tbl, "plx_id", tbl)
        label = getattr(tbl, "label", getattr(h, "Name", None))

        tried = 0
        ok = False
        # Possible containers to hold a global setting
        containers = [g_i, getattr(g_i, "Project", None), getattr(g_i, "Model", None), getattr(g_i, "flow", None)]
        props = ("GlobalWaterLevel", "Global water level", "GlobalWaterlevel", "GeneralPhreaticLevel", "PhreaticLevel")

        for tgt in containers:
            if tgt is None: continue
            for p in props:
                tried += 1
                try:
                    if hasattr(tgt, "setproperties"):
                        tgt.setproperties(p, h); ok = True; break
                    setattr(tgt, p, h); ok = True; break
                except Exception:
                    try:
                        if hasattr(tgt, "setproperty"):
                            tgt.setproperty(p, h); ok = True; break
                    except Exception:
                        pass
            if ok: break

        _log_update("WaterLevelTable:SetGlobal",
                    f"label={label} tried={tried}",
                    h,
                    extra=("OK" if ok else "FAILED"))
