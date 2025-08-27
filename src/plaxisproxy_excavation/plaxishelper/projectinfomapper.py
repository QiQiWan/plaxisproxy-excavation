from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Union, Tuple, List

# === import your domain model ===
from ..components.projectinformation import ProjectInformation

# -----------------------------------------------------------------------------
# Logging (single-line, English)
# -----------------------------------------------------------------------------
def _format_handle(h: Any) -> str:
    if h is None:
        return "None"
    for k in ("Id", "ID", "id", "Guid", "GUID", "guid", "Name", "Identification", "Title"):
        try:
            v = getattr(h, k, None)
            if v is not None:
                if hasattr(v, "value"):
                    v = v.value
                return f"{k}={v}"
        except Exception:
            continue
    s = str(h).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= 120 else (s[:117] + "...")

def _one_line(msg: str) -> str:
    return " ".join(str(msg).split())

def _log_create(kind: str, desc: str, handle: Any) -> None:
    print(_one_line(f"[CREATE][{kind}] {desc} handle={_format_handle(handle)}"), flush=True)

def _log_delete(kind: str, desc: str, handle: Any, ok: bool) -> None:
    print(_one_line(f"[DELETE][{kind}] {desc} handle={_format_handle(handle)} result={'OK' if ok else 'FAIL'}"), flush=True)

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _normalize_created_handle(created: Any) -> Any:
    if isinstance(created, (list, tuple)) and created:
        return created[0]
    return created

def _try_call(g_i: Any, names: Sequence[str], *args, **kwargs) -> Any:
    """
    Call the first available g_i.<name>(*args, **kwargs).
    This pattern absorbs API differences between PLAXIS bindings.
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

def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
    """
    Best-effort property setting with:
      1) bulk setproperties(k1, v1, k2, v2, ...)
      2) setattr(obj, key, value)
      3) setproperty(key, value)
    Skips None values.
    """
    if not props:
        return
    filtered = {k: v for k, v in props.items() if v is not None}

    # Bulk (fast path)
    if hasattr(plx_obj, "setproperties"):
        try:
            kv: List[Any] = []
            for k, v in filtered.items():
                kv.extend([k, v])
            if kv:
                plx_obj.setproperties(*kv)  # type: ignore[arg-type]
                return
        except Exception:
            pass

    # Per-key
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

# -----------------------------------------------------------------------------
# SoilContour helper
# -----------------------------------------------------------------------------
def _apply_soil_contour(g_i: Any, xmin: float, xmax: float, ymin: float, ymax: float) -> Any:
    """
    Define the soil contour as a rectangle :
      - SoilContour.initializerectangular(xmin, ymin, xmax, ymax)   

    Returns: a handle or the SoilContour container itself (best effort).
    Raises: RuntimeError if no approach works.
    """
    # Strategy B: property-based via SoilContour object
    sc = getattr(g_i, "SoilContour", None)
    if sc is not None:
        try:
            sc.initializerectangular(xmin, ymin, xmax, ymax)
        except Exception:
    # If nothing worked:
            raise RuntimeError("Failed to apply SoilContour via known entrypoints.")

# -----------------------------------------------------------------------------
# Mapper
# -----------------------------------------------------------------------------
class ProjectInformationMapper:
    """
    Apply project metadata and units into the PLAXIS model, and define SoilContour
    rectangle using (xmin, xmax, ymin, ymax).

    Behavior:
      - `create(g_i, proj)` resolves a project-like container (e.g., g_i.project),
        writes units and metadata, then sets SoilContour via robust strategies.
        The resolved container (or g_i as fallback) is stored in `proj.plx_id`.

      - `delete(g_i, proj)` only clears the local reference `proj.plx_id` and
        prints a single-line log (no actual project deletion here).
    """

    _PROJECT_HANDLES = ("project", "get_project", "setproject")

    @staticmethod
    def _resolve_project_container(g_i: Any) -> Any:
        for nm in ProjectInformationMapper._PROJECT_HANDLES:
            try:
                obj = getattr(g_i, nm, None)
                if callable(obj):
                    h = obj()
                    if h is not None:
                        return h
                elif obj is not None:
                    return obj
            except Exception:
                continue
        return g_i

    @staticmethod
    def create(g_i: Any, proj: ProjectInformation) -> Any:
        # 1) container
        container = ProjectInformationMapper._resolve_project_container(g_i)

        # 2) units
        units_payload = {
            "UnitLength": proj.length_unit.value,
            "UnitForce": proj.force_unit.value,
            # "StressUnit": proj.stress_unit.value,
            "UnitTime": proj.time_unit.value,
        }

        # 3) metadata (no Xmin/Xmax/Ymin/Ymax here anymore)
        meta_payload = {
            "Title": proj.title,
            "Company": proj.company,
            "Comments": proj.comment,
            # "Model": proj.model,
            # "Element": proj.element,
            # Directory / file name aliases
            # "ProjectDirectory": proj.dir,
            # "Directory": proj.dir,
            # "Path": proj.dir,
            # "FileName": proj.file_name,
            # "ProjectFile": proj.file_name,
            # Water unit weight aliases
            "WaterWeight": proj.gamma_water,
            # "Gravity": proj.gamma_water / 10,
            # "GWUnitWeight": proj.gamma_water,
            # "UnitWeightWater": proj.gamma_water,
        }

        _set_many_props(container, units_payload | meta_payload)

        # 4) SoilContour rectangle
        sc_handle = _apply_soil_contour(g_i, proj.x_min, proj.x_max, proj.y_min, proj.y_max)

        # 5) backref + log
        proj.plx_id = container
        _log_create("Project", f"title={proj.title} soilcontour=rect({proj.x_min},{proj.y_min})-({proj.x_max},{proj.y_max})", container)
        return container

    # ----------------------------- resets ---------------------------------
    @staticmethod
    def resetsoilcontour(
        g_i: Any,
        rect: Optional[Tuple[float, float, float, float]] = None,
        padding: float = 0.0,
        default_rect: Tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0),
    ) -> Any:
        """
        Reset the SoilContour and (re)define it as a rectangle.

        Args:
            g_i:       PLAXIS interface.
            rect:      (xmin, xmax, ymin, ymax). If None, use default_rect.
            padding:   Expand rectangle by this margin on all sides (>= 0).
            default_rect:
                       Used when rect is None. Format is (xmin, xmax, ymin, ymax).

        Returns:
            SoilContour handle/container (best effort). Raises if all attempts fail.
        """
        # 0) Try direct reset entrypoints on g_i (best effort; may not exist)
        for nm in ("reset_soilcontour", "resetsoilcontour", "clear_soilcontour", "clearsoilcontour"):
            fn = getattr(g_i, nm, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    pass

        # 1) Try method on SoilContour object
        sc = getattr(g_i, "SoilContour", None)
        if sc is not None:
            for meth in ("Reset", "Clear", "Delete", "Remove"):
                f = getattr(sc, meth, None)
                if callable(f):
                    try:
                        f()
                        break
                    except Exception:
                        continue
            # In case properties can "clear" by empty assignment (some bindings)
            for key in ("Polygon", "SoilContourPolygon", "Coordinates", "XY", "Points"):
                try:
                    setattr(sc, key, [])
                except Exception:
                    try:
                        if hasattr(sc, "setproperty"):
                            sc.setproperty(key, [])
                    except Exception:
                        pass

        # 2) Decide rectangle and apply
        if rect is None:
            rect = default_rect
        xmin, xmax, ymin, ymax = rect

        if padding and padding > 0.0:
            xmin -= padding
            ymin -= padding
            xmax += padding
            ymax += padding

        handle = _apply_soil_contour(g_i, xmin, xmax, ymin, ymax)

        # 3) One-line log
        print(_one_line(f"[RESET][SoilContour] rect=({xmin},{ymin})-({xmax},{ymax}) handle={_format_handle(handle)}"), flush=True)
        return handle

    @staticmethod
    def resetsoilcontour_from_project(
        g_i: Any,
        proj: "ProjectInformation",
        padding: float = 0.0
    ) -> Any:
        """
        Convenience wrapper: reset SoilContour using bounds from a ProjectInformation object.

        Args:
            g_i:      PLAXIS interface.
            proj:     ProjectInformation with x_min/x_max/y_min/y_max.
            padding:  Extra margin on all sides.

        Returns:
            SoilContour handle/container.
        """
        rect = (proj.x_min, proj.x_max, proj.y_min, proj.y_max)
        return ProjectInformationMapper.resetsoilcontour(g_i, rect=rect, padding=padding)

    @staticmethod
    def delete(g_i: Any, proj_or_handle: Union[ProjectInformation, Any]) -> bool:
        proj = proj_or_handle if isinstance(proj_or_handle, ProjectInformation) else None
        h = getattr(proj, "plx_id", None) if proj else proj_or_handle
        ok = True
        if proj is not None:
            proj.plx_id = None
        _log_delete("Project", f"title={getattr(proj, 'title', 'raw')}", h, ok)
        return ok
