from __future__ import annotations
from typing import Any, Iterable, Iterator, List, Sequence, Tuple, Union, Optional, overload, Literal, cast
from ..geometry import Point, PointSet, Line3D, Polygon3D


# ----------------------------- logging helpers ------------------------------
def _enum_to_str(v: Any) -> str:
    try:
        if hasattr(v, "name"):
            return str(v.name)
        if hasattr(v, "value"):
            return str(v.value)
        return str(v)
    except Exception:
        return str(v)

def _get_attr_value(obj: Any, key: str) -> Optional[str]:
    try:
        v = getattr(obj, key, None)
        if v is None:
            return None
        if hasattr(v, "value"):
            v = v.value
        return str(v)
    except Exception:
        return None

def _format_handle(h: Any) -> str:
    if h is None:
        return "None"
    for k in ("Id", "ID", "id", "Guid", "GUID", "guid", "Name", "MaterialName", "Identification"):
        val = _get_attr_value(h, k)
        if val:
            return f"{k}={val}"
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

def _try_delete_with_gi(g_i: Any, plx_obj: Any) -> bool:
    """Try a few common deletion entrypoints on g_i or the object itself."""
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


class GeometryMapper:
    """
    Static utility class that maps geometric primitives to PLAXIS via g_i.
    It provides point/line/surface creation using only g_i.point / g_i.line / g_i.surface,
    and supports deletion with plx_id reset on the domain object.
    """

    # ----------------------------- public: points -----------------------------
    @staticmethod
    def create_point(g_i: Any, point: Point) -> Union[Any, bool]:
        """Create a PLAXIS point and attach handle to point.plx_id."""
        try:
            x, y, z = point.get_point()
            try:
                plx_id = g_i.point(x, y, z)
            except TypeError:
                plx_id = g_i.point((x, y, z))
            setattr(point, "plx_id", plx_id)
            _log_create("Point", f"id={getattr(point, 'id', 'N/A')}", plx_id)
            return plx_id
        except Exception as e:
            print(_one_line(f"[ERROR][Point] create id={getattr(point,'id','N/A')} msg={e}"))
            return False

    @staticmethod
    def delete_point(g_i: Any, obj: Union[Point, Any]) -> bool:
        """
        Delete a PLAXIS point by Point or raw handle.
        On success, reset Point.plx_id = None.
        """
        is_point = isinstance(obj, Point)
        handle = getattr(obj, "plx_id", None) if is_point else obj
        if handle is None:
            _log_delete("Point", f"id={getattr(obj,'id','N/A')}", handle, ok=False, extra="reason=missing_handle")
            return False

        ok = _try_delete_with_gi(g_i, handle)
        if ok and is_point:
            setattr(obj, "plx_id", None)
        desc = f"id={getattr(obj,'id','N/A')}" if is_point else "raw_handle"
        _log_delete("Point", desc, handle, ok=ok)
        return ok

    @staticmethod
    def create_points(
        g_i: Any,
        data: Union[
            PointSet,
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
        ],
        stop_on_error: bool = False,
    ) -> List[Optional[Any]]:
        """Batch create points. Returns list of handles (None for failures)."""
        results: List[Optional[Any]] = []
        ok_count = 0
        for idx, p in enumerate(GeometryMapper._iter_points(data)):
            try:
                x, y, z = p.get_point()
                try:
                    plx_id = g_i.point(x, y, z)
                except TypeError:
                    plx_id = g_i.point((x, y, z))
                setattr(p, "plx_id", plx_id)
                results.append(plx_id)
                ok_count += 1
            except Exception as e:
                msg = f"[ERROR][Point] batch_index={idx} msg={e}"
                if stop_on_error:
                    raise RuntimeError(msg) from e
                print(_one_line(msg))
                results.append(None)
        _log_create("PointBatch", f"count={len(results)}", results[0] if results else None, extra=f"ok={ok_count} fail={len(results)-ok_count}")
        return results

    # ----------------------------- public: lines ------------------------------
    @staticmethod
    def create_line(
        g_i: Any,
        data: Union[
            # points-based inputs
            PointSet,
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
            Tuple[Point, Point],
            Tuple[Tuple[float, float, float], Tuple[float, float, float]],
            # line-based inputs
            Line3D,
            Sequence[Line3D],
            Iterable[Line3D],
        ],
        name: Optional[str] = None,
        stop_on_error: bool = False,
    ) -> Union[Any, List[Any], Line3D, List[Line3D], None]:
        """
        Create PLAXIS line(s) with g_i.line. Attach plx_id to Line3D where applicable.
        """
        try:
            # --------- B) Line3D input (single or multiple) ---------
            if isinstance(data, Line3D) or (
                isinstance(data, (list, tuple)) and len(data) > 0 and all(isinstance(x, Line3D) for x in data)
            ):
                line_objs: List[Line3D] = [data] if isinstance(data, Line3D) else list(data)
                for ln in line_objs:
                    pts = list(ln.get_points())
                    if len(pts) < 2:
                        raise ValueError("Line3D must contain at least two points.")
                    seg_pairs = GeometryMapper._segments_from_points(pts)

                    created_ids: List[Any] = []
                    for i, (pA, pB) in enumerate(seg_pairs):
                        seg_id = GeometryMapper._create_gi_line_between_points(g_i, pA, pB)
                        if i == 0 and name and hasattr(seg_id, "Name"):
                            try:
                                seg_id.Name = name
                            except Exception:
                                pass
                        created_ids.append(seg_id)

                    # Attach to Line3D.plx_id (single or list)
                    setattr(ln, "plx_id", created_ids[0] if len(created_ids) == 1 else created_ids)
                    seg_info = f"segments={len(created_ids)}"
                    _log_create("Line", f"id={getattr(ln,'id','N/A')} {seg_info}", created_ids[0] if created_ids else None)
                return line_objs[0] if isinstance(data, Line3D) else line_objs

            # --------- A) Points input (two or more points) ----------
            if isinstance(data, (tuple, list)) and len(data) == 2 and (
                isinstance(data[0], Point) or isinstance(data[0], (tuple, list))
            ):
                pts = [GeometryMapper._as_point(data[0]),
                       GeometryMapper._as_point(data[1])]
            elif isinstance(data, PointSet):
                pts = data.get_points()
            else:
                iter_src: Iterable[Union[Point, Tuple[float, float, float]]] = cast(Iterable[Any], data)
                pts = [GeometryMapper._as_point(p) for p in GeometryMapper._iter_points(iter_src)]

            if len(pts) < 2:
                raise ValueError("A line requires at least two points.")

            seg_pairs = GeometryMapper._segments_from_points(pts)
            created_ids: List[Any] = []
            for i, (pA, pB) in enumerate(seg_pairs):
                seg_id = GeometryMapper._create_gi_line_between_points(g_i, pA, pB)
                if i == 0 and name and hasattr(seg_id, "Name"):
                    try:
                        seg_id.Name = name
                    except Exception:
                        pass
                created_ids.append(seg_id)

            _log_create("Line", f"from_points segments={len(created_ids)}", created_ids[0] if created_ids else None)
            return created_ids[0] if len(created_ids) == 1 else created_ids

        except Exception as e:
            msg = f"[ERROR][Line] create msg={e}"
            if stop_on_error:
                raise RuntimeError(msg) from e
            print(_one_line(msg))
            return None

    @staticmethod
    def delete_line(g_i: Any, obj: Union[Line3D, Any, List[Any]]) -> bool:
        """
        Delete a PLAXIS line by Line3D or raw handle(s).
        If obj is Line3D and deletion success for all underlying handles,
        reset Line3D.plx_id = None; if partially failed, keep remaining handles.
        """
        if isinstance(obj, Line3D):
            handles: List[Any] = []
            h = getattr(obj, "plx_id", None)
            if isinstance(h, list):
                handles.extend(h)
            elif h is not None:
                handles.append(h)

            if not handles:
                _log_delete("Line", f"id={getattr(obj,'id','N/A')}", None, ok=False, extra="reason=missing_handle")
                return False

            oks = []
            for hd in handles:
                ok = _try_delete_with_gi(g_i, hd)
                oks.append(ok)
                _log_delete("Line", f"id={getattr(obj,'id','N/A')}", hd, ok=ok)

            if all(oks):
                setattr(obj, "plx_id", None)
                return True
            else:
                # keep any undeleted handle(s)
                remaining = [hd for hd, ok in zip(handles, oks) if not ok]
                setattr(obj, "plx_id", remaining if remaining else None)
                return False

        # raw handle or list of handles
        if isinstance(obj, list):
            oks = []
            for hd in obj:
                ok = _try_delete_with_gi(g_i, hd)
                oks.append(ok)
                _log_delete("Line", "raw_handle", hd, ok=ok)
            return all(oks)
        else:
            ok = _try_delete_with_gi(g_i, obj)
            _log_delete("Line", "raw_handle", obj, ok=ok)
            return ok

    # ---------------------------- public: surfaces ----------------------------
    @overload
    @staticmethod
    def create_surface(
        g_i: Any,
        data: Union[
            Polygon3D,
            PointSet,
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
        ],
        name: Optional[str] = None,
        auto_close: bool = True,
        stop_on_error: bool = False,
        return_polygon: Literal[True] = True,
    ) -> Tuple[Any, Polygon3D]: ...
    @overload
    @staticmethod
    def create_surface(
        g_i: Any,
        data: Union[
            Polygon3D,
            PointSet,
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
        ],
        name: Optional[str] = None,
        auto_close: bool = True,
        stop_on_error: bool = False,
        return_polygon: Literal[False] = False,
    ) -> Union[Any, bool]: ...

    @staticmethod
    def create_surface(
        g_i: Any,
        data: Union[
            Polygon3D,
            PointSet,
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
        ],
        name: Optional[str] = None,
        auto_close: bool = True,
        stop_on_error: bool = False,
        return_polygon: bool = False,
    ) -> Union[Any, bool, Tuple[Any, Polygon3D]]:
        """Create a PLAXIS surface, attach handle to Polygon3D.plx_id."""
        try:
            polygon = GeometryMapper.coerce_to_polygon3d(data, auto_close=auto_close)

            pts = list(polygon.get_points())
            if len(pts) < 3:
                raise ValueError("A surface requires at least three points.")

            plx_pts = [GeometryMapper._ensure_plaxis_point(g_i, p) for p in pts]

            # Try signatures: (handles...) or flat numeric coords
            try:
                surf_id = g_i.surface(*plx_pts)
            except TypeError:
                coords: List[float] = []
                for p in pts:
                    x, y, z = GeometryMapper._to_float_xyz(p)
                    coords.extend([x, y, z])
                surf_id = g_i.surface(*coords)

            try:
                setattr(polygon, "plx_id", surf_id)
            except Exception:
                pass

            if name and hasattr(surf_id, "Name"):
                try:
                    surf_id.Name = name
                except Exception:
                    pass

            _log_create("Surface", f"id={getattr(polygon,'id','N/A')} npts={len(pts)}", surf_id)

            return (surf_id, polygon) if return_polygon else surf_id

        except Exception as e:
            msg = f"[ERROR][Surface] create msg={e}"
            if stop_on_error:
                raise RuntimeError(msg) from e
            print(_one_line(msg))
            return False

    @staticmethod
    def delete_surface(g_i: Any, obj: Union[Polygon3D, Any]) -> bool:
        """
        Delete a PLAXIS surface by Polygon3D or raw handle.
        On success, reset Polygon3D.plx_id = None.
        """
        is_poly = isinstance(obj, Polygon3D)
        handle = getattr(obj, "plx_id", None) if is_poly else obj
        if handle is None:
            _log_delete("Surface", f"id={getattr(obj,'id','N/A')}", handle, ok=False, extra="reason=missing_handle")
            return False

        ok = _try_delete_with_gi(g_i, handle)
        if ok and is_poly:
            setattr(obj, "plx_id", None)
        desc = f"id={getattr(obj,'id','N/A')}" if is_poly else "raw_handle"
        _log_delete("Surface", desc, handle, ok=ok)
        return ok

    # ----------------------------- helpers (static) ---------------------------
    @staticmethod
    def _as_point(obj: Any) -> Point:
        """Normalize any supported input into a Point instance."""
        if isinstance(obj, Point):
            return obj
        if all(hasattr(obj, a) for a in ("x", "y", "z")):
            return Point(float(getattr(obj, "x")), float(getattr(obj, "y")), float(getattr(obj, "z")))
        try:
            x, y, z = obj  # type: ignore[misc]
            return Point(float(x), float(y), float(z))
        except Exception as exc:
            raise TypeError(
                "Unsupported point-like input. Expected Point, (x,y,z), or object with x/y/z."
            ) from exc

    @staticmethod
    def _iter_points(
        data: Union[PointSet, Iterable[Point], Iterable[Tuple[float, float, float]]]
    ) -> Iterator[Point]:
        """Yield Points from supported containers."""
        if isinstance(data, PointSet):
            for p in data.get_points():
                yield GeometryMapper._as_point(p)
            return
        for item in data:
            yield GeometryMapper._as_point(item)

    @staticmethod
    def _to_float_xyz(p: Point) -> Tuple[float, float, float]:
        """Return a numeric (x, y, z) triple from a Point."""
        x, y, z = p.get_point()
        return float(x), float(y), float(z)

    @staticmethod
    def _ensure_plaxis_point(g_i: Any, p: Point) -> Any:
        """
        Ensure the point exists in PLAXIS and return its plx_id.
        If the Point already has plx_id, reuse it; otherwise create it.
        """
        existing = getattr(p, "plx_id", None)
        if existing is not None:
            return existing
        x, y, z = GeometryMapper._to_float_xyz(p)
        try:
            plx_id = g_i.point(x, y, z)
        except TypeError:
            plx_id = g_i.point((x, y, z))
        setattr(p, "plx_id", plx_id)
        return plx_id

    @staticmethod
    def _segments_from_points(pts: Sequence[Point]) -> List[Tuple[Point, Point]]:
        """Return consecutive point pairs as segments."""
        if len(pts) < 2:
            raise ValueError("A line requires at least two points.")
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    @staticmethod
    def _create_gi_line_between_points(g_i: Any, p1: Point, p2: Point) -> Any:
        """
        Create a PLAXIS line between two Points using g_i.line only.
        Tries (handle, handle) first, then falls back to numeric coordinates.
        """
        h1 = GeometryMapper._ensure_plaxis_point(g_i, p1)
        h2 = GeometryMapper._ensure_plaxis_point(g_i, p2)
        try:
            return g_i.line(h1, h2)  # common signature
        except TypeError:
            x1, y1, z1 = GeometryMapper._to_float_xyz(p1)
            x2, y2, z2 = GeometryMapper._to_float_xyz(p2)
            return g_i.line(x1, y1, z1, x2, y2, z2)

    @staticmethod
    def coerce_to_polygon3d(
        data: Union[
            Polygon3D,
            PointSet,
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
        ],
        auto_close: bool = True,
    ) -> Polygon3D:
        """Coerce any supported input into a Polygon3D."""
        if isinstance(data, Polygon3D):
            poly = data
        else:
            pts = [GeometryMapper._as_point(p) for p in GeometryMapper._iter_points(cast(Iterable[Any], data))]
            # auto-close BEFORE constructing Polygon3D (if requested)
            if pts and (pts[0].x, pts[0].y, pts[0].z) != (pts[-1].x, pts[-1].y, pts[-1].z) and auto_close:
                pts = pts + [pts[0]]
            poly = Polygon3D.from_points(PointSet(pts))

        # Check closure with best-effort
        try:
            is_closed = poly._is_closed()  # type: ignore[attr-defined]
        except AttributeError:
            pts2 = list(poly.get_points())
            is_closed = (pts2[0].x, pts2[0].y, pts2[0].z) == (pts2[-1].x, pts2[-1].y, pts2[-1].z)

        if not is_closed and auto_close:
            pts2 = list(poly.get_points())
            if pts2 and (pts2[0].x, pts2[0].y, pts2[0].z) != (pts2[-1].x, pts2[-1].y, pts2[-1].z):
                pts2 = pts2 + [pts2[0]]
            poly = Polygon3D.from_points(PointSet(pts2))

        return poly
