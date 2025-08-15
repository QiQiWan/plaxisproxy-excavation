from __future__ import annotations
from typing import Any, Iterable, Iterator, List, Sequence, Tuple, Union, Optional
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D

class GeometryMapper:
    """
    Static utility class that maps geometric primitives to PLAXIS via g_i.
    It provides point/line/surface creation using only g_i.point / g_i.line / g_i.surface.
    """

    # ----------------------------- public: points -----------------------------
    @staticmethod
    def create_point(g_i: Any, point: Point) -> Any:
        """
        Create a single PLAXIS point and attach the returned plx_id to the Point.

        Returns:
          - The PLAXIS point handle (plx_id) on success;
          - False on failure (and prints an error).
        """
        try:
            x, y, z = point.get_point()
            try:
                plx_id = g_i.point(x, y, z)       # common signature with numeric args
            except TypeError:
                plx_id = g_i.point((x, y, z))     # some bindings accept a tuple
            point.plx_id = plx_id
            return plx_id
        except Exception as e:
            print(f"[Error] Failed to create point {getattr(point, 'id', '')}: {e}")
            return False

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
    ) -> List[Any]:
        """
        Create multiple PLAXIS points from a PointSet or any iterable of Point/(x,y,z).

        Args:
          data:
            - PointSet
            - Iterable/Sequence of Point
            - Iterable/Sequence of (x, y, z) coordinates
          stop_on_error:
            - If True, stop at the first failure and raise the exception.
            - If False (default), continue and append None for the failed entry.

        Returns:
          List of PLAXIS point handles (plx_id). Failed entries are None.
        """
        results: List[Any] = []
        for idx, p in enumerate(GeometryMapper._iter_points(data)):
            try:
                x, y, z = p.get_point()
                try:
                    plx_id = g_i.point(x, y, z)
                except TypeError:
                    plx_id = g_i.point((x, y, z))
                p.plx_id = plx_id
                results.append(plx_id)
            except Exception as e:
                msg = f"[Error] Failed to create point at index {idx}: {e}"
                if stop_on_error:
                    raise RuntimeError(msg) from e
                print(msg)
                results.append(None)
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
        Create PLAXIS line(s) with g_i.line according to input type.

        Behavior
        --------
        A) Points input (two or more points):
           - Convert to consecutive segments (Line3D conceptually).
           - For each segment, call g_i.line( ... ) to create a line in PLAXIS.
           - Return:
               * single handle   if exactly two points were provided;
               * list of handles if >=3 points (piecewise polyline).

        B) Line3D input (single or multiple):
           - For each Line3D, create line(s) based on its endpoints:
               * If the Line3D has exactly 2 points -> one PLAXIS line.
               * If >2 points -> split to segments and create multiple lines.
           - Assign returned handle(s) to line.plx_id.
           - Return the original Line3D (single) or list of Line3D (multiple)
             with plx_id populated.

        Notes
        -----
        - Only g_i.line is used (no polyline fallback).
        - If 'name' is provided and the returned PLAXIS object exposes .Name,
          we set it on the first created segment (or the single line).
        """
        try:
            # --------- B) Line3D input (single or multiple) ---------
            if isinstance(data, Line3D) or (
                isinstance(data, (list, tuple)) and data and all(isinstance(x, Line3D) for x in data)
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

                    ln.plx_id = created_ids[0] if len(created_ids) == 1 else created_ids  # type: ignore[attr-defined]
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
                pts = [GeometryMapper._as_point(p) for p in GeometryMapper._iter_points(data)]  # type: ignore

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

            return created_ids[0] if len(created_ids) == 1 else created_ids

        except Exception as e:
            msg = f"[Error] Failed to create line(s): {e}"
            if stop_on_error:
                raise RuntimeError(msg) from e
            print(msg)
            return None

    # ---------------------------- public: surfaces ----------------------------
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
    ) -> Any:
        """
        Create a PLAXIS surface using g_i.surface only.
        If input is not a Polygon3D, it is first coerced into one and (optionally) returned.

        Args:
          data: Polygon3D OR point-like container to be converted into Polygon3D.
          name: optional name (set if PLAXIS object exposes .Name).
          auto_close: auto-close the ring during coercion if needed.
          validate_ring: pass-through to coerce_to_polygon3d(...).
          stop_on_error: strict mode toggle.
          return_polygon: if True, return (surf_id, polygon).

        Returns:
          - PLAXIS surface handle, or
          - (surf_id, polygon) if return_polygon=True, or
          - False on failure.
        """
        try:
            polygon = GeometryMapper.coerce_to_polygon3d(
                data, auto_close=auto_close
            )

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

            # Attach back to Polygon3D if it supports it
            try:
                polygon.plx_id = surf_id  # type: ignore[attr-defined]
            except Exception:
                pass

            if name and hasattr(surf_id, "Name"):
                try:
                    surf_id.Name = name
                except Exception:
                    pass

            if return_polygon:
                return surf_id, polygon
            return surf_id

        except Exception as e:
            msg = f("[Error] Failed to create surface: {e}")
            if stop_on_error:
                raise RuntimeError(msg) from e
            print(msg)
            return False

    # ----------------------------- helpers (static) ---------------------------
    @staticmethod
    def _as_point(obj: Any) -> Point:
        """
        Normalize any supported input into a Point instance.

        Supported inputs:
          - Point
          - (x, y, z) tuple/list
          - object with attributes x, y, z

        Raises:
          TypeError if the input cannot be converted.
        """
        if isinstance(obj, Point):
            return obj
        if all(hasattr(obj, a) for a in ("x", "y", "z")):
            return Point(float(obj.x), float(obj.y), float(obj.z))
        try:
            x, y, z = obj
            return Point(float(x), float(y), float(z))
        except Exception as exc:
            raise TypeError(
                "Unsupported point-like input. Expected Point, (x,y,z), or object with x/y/z."
            ) from exc

    @staticmethod
    def _iter_points(
        data: Union[PointSet, Iterable[Point], Iterable[Tuple[float, float, float]]]
    ) -> Iterator[Point]:
        """
        Yield Points from supported containers:
          - PointSet
          - Iterable[Point]
          - Iterable[(x, y, z)]
        """
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
        if getattr(p, "plx_id", None) is not None:
            return p.plx_id
        x, y, z = GeometryMapper._to_float_xyz(p)
        try:
            plx_id = g_i.point(x, y, z)     # numeric signature
        except TypeError:
            plx_id = g_i.point((x, y, z))   # tuple signature
        p.plx_id = plx_id
        return plx_id

    @staticmethod
    def _segments_from_points(pts: List[Point]) -> List[Tuple[Point, Point]]:
        """
        Given >=2 points, return consecutive point pairs as segments.
        2 points  -> 1 segment: [(p0, p1)]
        N>=3 pts  -> N-1 segments: [(p0,p1), (p1,p2), ...]
        """
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
        """
        Coerce any supported input into a Polygon3D.

        Behavior:
          - If data is already a Polygon3D, use it directly.
          - Otherwise, normalize to a list of Points -> build PointSet -> build Polygon3D.
          - Use Polygon3D's own _is_closed() and validate_ring() for closure & ring validation.
        """
        if isinstance(data, Polygon3D):
            poly = data
        else:
            pts = [GeometryMapper._as_point(p) for p in GeometryMapper._iter_points(data)]
            # auto-close BEFORE constructing Polygon3D (if requested)
            if pts and (pts[0].x, pts[0].y, pts[0].z) != (pts[-1].x, pts[-1].y, pts[-1].z) and auto_close:
                pts = pts + [pts[0]]
            poly = Polygon3D.from_points(PointSet(pts))

        # Use Polygon3D's own methods
        try:
            is_closed = poly._is_closed()
        except AttributeError:
            # Robust fallback if private name differs
            pts2 = list(poly.get_points())
            is_closed = (pts2[0].x, pts2[0].y, pts2[0].z) == (pts2[-1].x, pts2[-1].y, pts2[-1].z)

        if not is_closed and auto_close:
            pts2 = list(poly.get_points())
            if pts2 and (pts2[0].x, pts2[0].y, pts2[0].z) != (pts2[-1].x, pts2[-1].y, pts2[-1].z):
                pts2 = pts2 + [pts2[0]]
            poly = Polygon3D.from_points(PointSet(pts2))


        return poly
