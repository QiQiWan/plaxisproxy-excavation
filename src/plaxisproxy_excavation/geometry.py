from __future__ import annotations
import uuid
import math
from typing import List, Iterator, Optional, Tuple, Dict, Any, Sequence
from .core.plaxisobject import PlaxisObject


"""plx.geometry - core geometric primitives with optional Shapely support
--------------------------------------------------------------------------
This module defines lightweight 3-D Point / Line / Polygon containers that are
sufficient for exchanging data with the Plaxis API while remaining free of any
heavy GIS dependency.  When the *Shapely* package is available, richer planar
operations (area, perimeter, validity checks, boolean ops) are automatically
enabled; otherwise, minimal fall-backs keep the library functional.
"""

# ---------------------------------------------------------------------------
# Optional Shapely backend ---------------------------------------------------
# ---------------------------------------------------------------------------
try:
    from shapely.geometry import Point as _ShpPoint, LineString as _ShpLine, Polygon as _ShpPolygon
    from shapely.ops import unary_union as _shp_union, polygonize as _shp_polygonize
    _SHAPELY_AVAILABLE = True
except ModuleNotFoundError:
    _SHAPELY_AVAILABLE = False
    _ShpPoint, _ShpLine, _ShpPolygon = None, None, None
    _shp_union, _shp_polygonize = None, None


class GeometryBase(PlaxisObject):
    
    def __init__(self, name: str = "") -> None:
        super().__init__(name=name)

    @staticmethod
    def _uuid_to_str(u: uuid.UUID | None) -> str | None:
        """Converts a UUID object to a string, handling None values."""
        return str(u) if u is not None else None

    @staticmethod
    def _str_to_uuid(s: str | None) -> uuid.UUID | None:
        """Converts a string to a UUID object, handling None values."""
        return uuid.UUID(s) if s is not None and s != '' else None

# ---------------------------------------------------------------------------
# Core Primitives -----------------------------------------------------------
# ---------------------------------------------------------------------------
class Point(GeometryBase):
    """
    Immutable three-dimensional point.

    This class provides a lightweight, hashable representation of a point in 3D
    space, with optional interoperability with the Shapely library for 2D
    planar operations.
    """

    __slots__ = ("_x", "_y", "_z")

    def __init__(self, x: float, y: float, z: float):
        super().__init__()
        self._x = float(x)
        self._y = float(y)
        self._z = float(z)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_point(self) -> Tuple[float, float, float]:
        """Return the coordinates as a tuple."""
        return (self._x, self._y, self._z)
    
    def distance_to(self, other: "Point") -> float:
        """Calculate Euclidean distance to another Point."""
        if not isinstance(other, Point):
            raise TypeError("Can only calculate distance to another Point instance.")
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )

    def to_shapely(self):
        """Return *shapely.geometry.Point* projected to the XY plane."""
        if not _SHAPELY_AVAILABLE:
            raise RuntimeError("Shapely is not installed - pip install shapely")
        from shapely.geometry import Point as _ShpPoint
        return _ShpPoint(self._x, self._y)

    # ------------------------------------------------------------------
    # Dunder / Properties
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "id": self._uuid_to_str(getattr(self, "_id", None))
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Point":
        pt = cls(data["x"], data["y"], data["z"])
        if "id" in data:
            pt.id = cls._str_to_uuid(data["id"])         # type: ignore[attr-defined]
        return pt


    def __repr__(self):
        # Updated repr format
        return f"<plx.geometry.Point x={self._x:.3f}, y={self._y:.3f}, z={self._z:.3f}>"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (
            math.isclose(self._x, other._x, abs_tol=1e-9) and
            math.isclose(self._y, other._y, abs_tol=1e-9) and
            math.isclose(self._z, other._z, abs_tol=1e-9)
        )
    
    def __hash__(self):
        return hash((self._x, self._y, self._z))

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def z(self) -> float:
        return self._z


class PointSet(GeometryBase):
    """
    Mutable ordered collection of :class:`Point`.

    This class serves as a container for ordered lists of Point objects,
    providing basic mutation and query operations.
    """

    __slots__ = ("_points",)

    def __init__(self, points: Optional[List[Point]] = None):
        super().__init__()
        self._points: List[Point] = points if points is not None else []

    # -------------------- mutation -----------------------------------
    def add_point(self, x: float, y: float, z: float) -> None:
        """Add a new point to the set from coordinates."""
        self._points.append(Point(x, y, z))

    # -------------------- queries ------------------------------------
    def get_points(self) -> List[Point]:
        """Return the internal list of Point objects."""
        return self._points

    def is_closed(self) -> bool:
        """Check if the point set forms a closed loop."""
        return len(self._points) >= 2 and self._points[0] == self._points[-1]

    def to_shapely(self):
        """Return *shapely.geometry.LineString* projected to XY plane."""
        if not _SHAPELY_AVAILABLE:
            raise RuntimeError("Shapely is required for this operation. Please install shapely.")
        from shapely.geometry import LineString as _ShpLine
        return _ShpLine([(p.x, p.y) for p in self._points])

    # ---------------- container magic --------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {"points": [p.to_dict() for p in self._points]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointSet":
        pts = [Point.from_dict(p) for p in data["points"]]
        return cls(pts)
    
    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[Point]:
        return iter(self._points)

    def __getitem__(self, idx) -> Point:
        return self._points[idx]

    def __repr__(self) -> str:
        # Updated repr format
        return f"<plx.geometry.PointSet n_points={len(self)} closed={self.is_closed()}>"


# ---------------------------------------------------------------------------
# Line3D --------------------------------------------------------------------
# ---------------------------------------------------------------------------
class Line3D(GeometryBase):
    """
    Polyline in 3-D space.

    This class represents a sequence of connected Point objects and provides
    various geometric properties and methods.
    """

    __slots__ = ("_id", "_plx_id", "_point_set")

    def __init__(self, point_set: PointSet):
        super().__init__()
        if not isinstance(point_set, PointSet):
            raise TypeError("Line3D must be initialized with a PointSet instance.")
        self._point_set = point_set

    # --------------- delegation ---------------------------------------
    def add_point(self, x: float, y: float, z: float) -> None:
        """Add a point to the underlying PointSet."""
        self._point_set.add_point(x, y, z)

    def get_points(self) -> List[Point]:
        """Return the list of points that define the line."""
        return self._point_set.get_points()

    # --------------- geometric tests ---------------------------------
    def is_closed(self) -> bool:
        """Check if the line is a closed loop."""
        return self._point_set.is_closed()

    def is_valid_ring(self) -> bool:
        """
        Check if the line forms a valid closed ring.
        Requires >= 3 unique points and a closed loop.
        """
        pts = self.get_points()
        if len(pts) < 4 or not self.is_closed():
            return False
        # Check for at least 3 unique points (excluding the closing point)
        return len({(p.x, p.y, p.z) for p in pts[:-1]}) >= 3

    def as_tuple_list(self) -> List[Tuple[float, float, float]]:
        """
        Return a list of (x, y, z) float tuples for this Line3D.

        This is tolerant to different point representations:
        - plx.geometry.Point objects
        - objects exposing get_point() -> (x, y, z)
        - objects with attributes x, y, z
        - any iterable of length 3

        Raises
        ------
        TypeError: if a point is of unsupported type or coordinates are non-numeric.
        ValueError: if a point does not provide exactly 3 coordinates.
        """
        tuples: List[Tuple[float, float, float]] = []
        for p in self._point_set.get_points():
            if isinstance(p, Point):
                x, y, z = p.x, p.y, p.z
            # elif hasattr(p, "get_point") and callable(p.get_point):
            #     x, y, z = p.get_point()
            elif hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
                x, y, z = p.x, p.y, p.z
            else:
                try:
                    x, y, z = tuple(p)
                except Exception as exc:
                    raise TypeError("Unsupported point type in Line3D.") from exc

            if (x, y, z) is None or (len((x, y, z)) != 3):
                raise ValueError("Each point must provide exactly 3 coordinates (x, y, z).")

            try:
                fx, fy, fz = float(x), float(y), float(z)
            except Exception as exc:
                raise TypeError("Point coordinates must be numeric.") from exc

            tuples.append((fx, fy, fz))

        return tuples

    def to_shapely(self):
        """Return *shapely.geometry.LineString* projected to XY plane."""
        if not _SHAPELY_AVAILABLE:
            raise RuntimeError("Shapely is required for this operation. Please install shapely.")
        from shapely.geometry import LineString as _ShpLine
        return _ShpLine([(p.x, p.y) for p in self._point_set])

    # --------------- helpers -----------------------------------------
    @property
    def length(self) -> float:
        """
        Calculate the total length of the polyline.
        This now works for polylines with any number of points.
        """
        total_length = 0.0
        for i in range(len(self._point_set) - 1):
            total_length += self._point_set[i].distance_to(self._point_set[i+1])
        return total_length

    def is_vertical(self) -> bool:
        """Check if the line is vertical (constant X and Y)."""
        points = self._point_set.get_points() # Use get_points() for explicit list access
        if len(points) < 2:
            return True # A single point can be considered vertical.
        
        first_point = points[0]
        for p in points[1:]:
            if not (math.isclose(p.x, first_point.x, abs_tol=1e-9) and
                    math.isclose(p.y, first_point.y, abs_tol=1e-9)):
                return False
        return True

    def xy_location(self) -> Tuple[float, float]:
        """
        Return the (x, y) location of a vertical line.
        Raises ValueError if the line is not vertical.
        """
        if not self.is_vertical():
            raise ValueError("Line is not vertical (x and y coordinates are not constant).")
        return (self._point_set[0].x, self._point_set[0].y)

    # --------------- dunder ------------------------------------------
    def __len__(self) -> int:
        return len(self._point_set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "plx_id": self._plx_id,
            "points": [p.to_dict() for p in self._point_set.get_points()]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Line3D":
        pts = [Point.from_dict(p) for p in data["points"]]
        line = cls(PointSet(pts))
        # line.id = data["id"]
        line.plx_id = data.get("plx_id")
        return line

    def __iter__(self) -> Iterator[Point]:
        return iter(self._point_set)

    def __getitem__(self, idx) -> Point:
        return self._point_set[idx]
    
    def __eq__(self, other):
        if not isinstance(other, Line3D):
            return NotImplemented
        # Compare points to check for equality
        return self.get_points() == other.get_points()
    
    def __repr__(self) -> str:
        # Updated repr format
        return f"<plx.geometry.Line3D id={self._id} points={len(self)} closed={self.is_closed()}>"


# ---------------------------------------------------------------------------
# Polygon3D ------------------------------------------------------------------
# ---------------------------------------------------------------------------
class Polygon3D(GeometryBase):
    """
    Planar polygon (outer + optional inner rings).

    This class represents a 3D polygon defined by one or more closed `Line3D`
    rings, with the first line being the outer boundary and subsequent lines
    representing inner holes.
    """

    __slots__ = ("_id", "_plx_id", "_lines", "_point_set")

    def __init__(self, lines: Optional[List[Line3D]] = None, require_horizontal: bool = False):

        super().__init__()
        self._lines: List[Line3D] = lines if lines else []

        if not self._lines:
            return
        
        self.update_points()

        outer_points = self._lines[0].as_tuple_list() if hasattr(
            self._lines[0], "as_tuple_list") else list(self._lines[0])

        if len(outer_points) < 3:
            raise ValueError("Polygon3D requires at least 3 points for the outer ring.")

        if not self._is_coplanar(outer_points):
            raise ValueError("Polygon3D rings must be coplanar.")
        
        # Validation for all rings
        for i, ln in enumerate(self._lines):
            if not ln.is_valid_ring():
                raise ValueError(f"Ring at index {i} is invalid - must be closed with ≥3 unique planar points.")
        
        # Check if all rings are co-planar 
        for idx, ln in enumerate(self._lines):
            self._validate_ring(ln, idx)
        self._check_coplanar_general()

        # Coplanarity verification (second modes)
        if require_horizontal:
            self._check_constant_z()
        else:
            self._check_coplanar_general()

    def _check_coplanar(self) -> None:
        """Ensure all rings lie on the same plane. Raise ValueError if degenerate or non-coplanar."""
        if not getattr(self, "_lines", None):
            return

        outer = self._lines[0].get_points()
        if len(outer) < 3:
            raise ValueError("Polygon3D outer ring must have at least 3 points.")

        tol = 1e-9

        def vsub(a, b):
            return (a.x - b.x, a.y - b.y, a.z - b.z)

        def cross(u, v):
            return (u[1]*v[2] - u[2]*v[1],
                    u[2]*v[0] - u[0]*v[2],
                    u[0]*v[1] - u[1]*v[0])

        def dot_u_p(u, p):
            return u[0]*p.x + u[1]*p.y + u[2]*p.z

        def norm(u):
            return (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) ** 0.5

        # 1) Find any three points that are not collinear in the outer ring to determine the normal vector.
        p0 = outer[0]
        normal = None
        for i in range(1, len(outer) - 1):
            v1 = vsub(outer[i], p0)
            for j in range(i + 1, len(outer)):
                v2 = vsub(outer[j], p0)
                n = cross(v1, v2)
                if norm(n) > tol:
                    normal = n
                    break
            if normal is not None:
                break

        if normal is None:
            # The outer ring is completely collinear / degenerated
            raise ValueError("Polygon3D outer ring is degenerate (all points are collinear).")

        D = dot_u_p(normal, p0)

        # 2) Verify that all rings and all points are within this plane.
        for ln in self._lines:
            for p in ln.get_points():
                if not math.isclose(dot_u_p(normal, p), D, abs_tol=1e-8):
                    raise ValueError("Polygon3D rings must be coplanar.")
                
    def as_tuple_list(self) -> List[Tuple[float, float, float]]:
        """Return all vertex coordinates as a list of (x, y, z) tuples."""
        if not self._lines:
            return []
        return [p.get_point() for p in self.get_points()]
    
    def _ring_core_points(self, ln) -> List["Point"]:
        """Return the list of core points after removing the repeated ending points."""
        pts = ln.get_points()
        # Assuming that Line3D.is_closed() ensures that the starting and ending points coincide
        return pts[:-1] if len(pts) >= 2 else pts
    
    def _validate_ring(self, ln, idx: int, tol: float = 1e-12) -> None:
        """The ring must be closed, have at least 3 unique points, and not be collinear. Failure will result in raising a ValueError."""
        pts = ln.get_points()
        if not ln.is_closed():
            raise ValueError(f"Ring {idx} must be closed.")
        core = self._ring_core_points(ln)
        if len(core) < 3:
            raise ValueError(f"Ring {idx} must contain at least 3 points.")
        # The sole 3D point
        uniq = {(p.x, p.y, p.z) for p in core}
        if len(uniq) < 3:
            raise ValueError(f"Ring {idx} is degenerate (<3 unique points).")
        # Non-collinear: Attempt to find any three points that are not collinear.
        if self._non_collinear_triplet(core, tol) is None:
            raise ValueError(f"Ring {idx} is collinear and cannot form a polygon.")

    def _non_collinear_triplet(self, pts: List["Point"], tol: float = 1e-12) -> Tuple[int, int, int] | None:
        """Search for any three points within the set that are not collinear, and return their index triple; if no such points exist, return None."""
        def vsub(a, b): return (a.x - b.x, a.y - b.y, a.z - b.z)
        def cross(u, v): return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        def norm(u): return (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) ** 0.5

        n = len(pts)
        for i in range(n-2):
            for j in range(i+1, n-1):
                v1 = vsub(pts[j], pts[i])
                for k in range(j+1, n):
                    v2 = vsub(pts[k], pts[i])
                    nvec = cross(v1, v2)
                    if norm(nvec) > tol:
                        return (i, j, k)
        return None

    def _fit_plane_from_outer(self, tol: float = 1e-12):
        """
        Use the outer ring to find three non-collinear points to fit a plane, and return (p0, nvec).
        If the outer ring is degenerate (no non-collinear points can be found), raise a ValueError.
        """
        outer_core = self._ring_core_points(self._lines[0])
        idxs = self._non_collinear_triplet(outer_core, tol)
        if idxs is None:
            raise ValueError("Polygon3D outer ring is degenerate (collinear points).")
        i, j, k = idxs
        p0, p1, p2 = outer_core[i], outer_core[j], outer_core[k]

        # 法向
        def vsub(a, b): return (a.x - b.x, a.y - b.y, a.z - b.z)
        def cross(u, v): return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        nvec = cross(vsub(p1, p0), vsub(p2, p0))
        return p0, nvec

    def _point_plane_distance(self, p: "Point", p0: "Point", nvec) -> float:
        """The distance from the point (p0, nvec) to the plane; nvec does not need to be normalized."""
        nx, ny, nz = nvec
        denom = (nx*nx + ny*ny + nz*nz) ** 0.5
        if denom == 0.0:
            return float("inf")
        return abs((p.x - p0.x) * nx + (p.y - p0.y) * ny + (p.z - p0.z) * nz) / denom

    def _check_coplanar_general(self, tol_dist: float = 1e-9) -> None:
        """
        Use the outer ring fitting plane to check whether the distance from all points on all rings to the plane is <= tol_dist.
        If any point exceeds this limit, raise a ValueError. Supports any inclined plane.
        """
        p0, nvec = self._fit_plane_from_outer()
        for ln in self._lines:
            for p in ln.get_points():
                if self._point_plane_distance(p, p0, nvec) > tol_dist:
                    raise ValueError("Polygon3D rings must be coplanar.")

    def _check_constant_z(self, abs_tol: float = 1e-8) -> None:
        if not self._lines:
            return
        ref_z = self._lines[0].get_points()[0].z
        for ln in self._lines:
            for p in ln.get_points():
                if not math.isclose(p.z, ref_z, abs_tol=abs_tol):
                    raise ValueError("Polygon3D rings must be horizontal (constant z).")

    def _is_closed(self, tol: float = 1e-9, outer_only: bool = False) -> bool:
        """
        Return True if the polygon's ring(s) are closed within a given tolerance.

        By default, checks the outer ring and all inner rings. If you only need
        to verify the outer boundary, set `outer_only=True`.

        A ring is considered closed if its first and last vertices coincide
        in 3D within `tol`.

        Args:
            tol (float): Absolute tolerance for coordinate comparison.
            outer_only (bool): If True, only check the outer ring; otherwise all rings.

        Returns:
            bool: True if closed under the chosen scope, else False.
        """
        if not self._lines:
            return False

        def _ring_closed(ln: Line3D) -> bool:
            pts = ln.get_points()
            if len(pts) < 2:
                return False
            a, b = pts[0], pts[-1]
            return (
                math.isclose(a.x, b.x, abs_tol=tol) and
                math.isclose(a.y, b.y, abs_tol=tol) and
                math.isclose(a.z, b.z, abs_tol=tol)
            )

        if outer_only:
            return _ring_closed(self._lines[0])

        # check outer + all inner rings
        for ln in self._lines:
            if not _ring_closed(ln):
                return False
        return True

    @classmethod
    def from_points(cls, point_set: PointSet) -> "Polygon3D":
        if len(point_set) < 3:
            raise ValueError("Need at least 3 points for a polygon boundary.")

        if not point_set.is_closed():
            pts = point_set.get_points()
            closed_ps = PointSet(pts + [Point(pts[0].x, pts[0].y, pts[0].z)])
            return cls([Line3D(closed_ps)])

        # 已闭合直接用
        return cls([Line3D(point_set)])
    
    @classmethod
    def from_rectangle(cls, x_min, y_min, x_max, y_max, z=0) -> "Polygon3D":
        if x_min > x_max or y_min > y_max:
            raise ValueError("x_min and y_min should be less than x_max and y_max.")
        closed_ps = PointSet([
            Point(x_min, y_min, z),
            Point(x_min, y_max, z),
            Point(x_max, y_max, z),
            Point(x_max, y_min, z),
            Point(x_min, y_min, z)
        ])
        return cls([Line3D(closed_ps)])
        
    # ---------------- mutation ---------------------------------------
    def add_hole(self, line: Line3D) -> None:
        """Add an inner hole (closed ring) to the polygon."""
        if not line.is_valid_ring():
            raise ValueError("Hole to add is an invalid ring.")

        # 外环参考 z
        if not self._lines:
            raise ValueError("Polygon has no outer ring to compare hole plane.")
        ref_z = self._lines[0].get_points()[0].z

        # 孔洞必须与外环处于同一水平面（常量 z）
        for p in line.get_points():
            if not math.isclose(p.z, ref_z, abs_tol=1e-8):
                raise ValueError("Hole ring must lie on the same horizontal plane as the outer ring.")

        self._lines.append(line)
        self.update_points()

    def extrude(self, z1: float, z2: float) -> Cube:
        """
        Extrude the current planar polygon vertically between z1 and z2,
        returning a Cube that bounds the prism. This assumes the polygon is
        horizontal and rectangular.
        """
        if not self._is_closed():
            raise ValueError("Cannot extrude an open polygon.")

        # 取平面范围
        xs = [p.x for p in self.get_points()]
        ys = [p.y for p in self.get_points()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        from .geometry import Point, Cube
        p0 = Point(x_min, y_min, min(z1, z2))
        p1 = Point(x_max, y_max, max(z1, z2))
        return Cube(p0, p1)

    # ---------------- shapely interop --------------------------------
    def to_shapely(self):
        """Return a *shapely.geometry.Polygon* projected to XY plane."""
        if not _SHAPELY_AVAILABLE:
            raise RuntimeError("Shapely is required for this operation. Please install shapely.")
        outer = self._lines[0].to_shapely()
        inners = [ln.to_shapely() for ln in self._lines[1:]]
        from shapely.geometry import Polygon as _ShpPolygon
        return _ShpPolygon(outer, [inner.coords for inner in inners])

    # ---------------- geometric properties ---------------------------
    def area(self) -> float:
        """
        Calculate the area of the polygon.
        Uses Shapely if available, otherwise falls back to a shoelace algorithm on the outer ring.
        """
        if _SHAPELY_AVAILABLE:
            return self.to_shapely().area
        # fallback shoelace on outer ring
        pts = self._lines[0].get_points()
        s = sum(pts[i].x*pts[i+1].y-pts[i+1].x*pts[i].y for i in range(len(pts)-1))
        return abs(s)*0.5

    def perimeter(self) -> float:
        """Calculate the perimeter of the outer ring."""
        return self._lines[0].length

    def is_valid(self) -> bool:
        """Check if the polygon is geometrically valid."""
        if _SHAPELY_AVAILABLE:
            return self.to_shapely().is_valid
        return all(ln.is_valid_ring() for ln in self._lines)
    
    def _vector_sub(self, a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

    def _cross(self, a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    def _dot(self, a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def _norm(self, a):
        return math.sqrt(self._dot(a, a))

    def _is_coplanar(self, points, tol=1e-8):
        # 取前三个不共线点
        if len(points) < 3:
            return False
        p0 = points[0]
        # 找到非共线的三个点
        n = None
        for i in range(1, len(points)-1):
            v1 = self._vector_sub(points[i], p0)
            for j in range(i+1, len(points)):
                v2 = self._vector_sub(points[j], p0)
                n_candidate = self._cross(v1, v2)
                if self._norm(n_candidate) > tol:
                    n = n_candidate
                    break
            if n is not None:
                break
        if n is None:
            # All points are collinear, regarded as an invalid polygonal face.
            return False
        # Calculate the distance from all points to the plane (using point multiplication for approximation)
        for k in range(1, len(points)):
            vk = self._vector_sub(points[k], p0)
            if abs(self._dot(n, vk)) > tol:
                return False
        return True

    # ---------------- container --------------------------------------
    def get_lines(self) -> List[Line3D]:
        """Return the list of Line3D rings that define the polygon."""
        return self._lines

    def get_points(self) -> List[Point]:
        """Return a flat list of all points from all rings."""
        pts: List[Point] = []
        for ln in self._lines:
            pts.extend(ln.get_points())
        return pts
    
    def update_points(self) -> None:
        """Update the point set in the Polygon3D"""
        self._point_set = PointSet(self.get_points())

    @property
    def outer_ring(self) -> Line3D:
        """Return the outer boundary Line3D."""
        if not self._lines:
            raise IndexError("Polygon has no outer ring.")
        return self._lines[0]

    @property
    def inner_rings(self) -> List[Line3D]:
        """Return a list of inner hole Line3D rings."""
        return self._lines[1:]

    def __len__(self) -> int:
        return len(self._lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "plx_id": self._plx_id,
            "rings": [ln.to_dict() for ln in self._lines]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Polygon3D":
        rings = [Line3D.from_dict(r) for r in data["rings"]]
        poly = cls(rings)
        poly.id = data["id"]
        poly.plx_id = data.get("plx_id")
        return poly

    def __iter__(self) -> Iterator[Line3D]:
        return iter(self._lines)

    def __getitem__(self, idx) -> Line3D:
        return self._lines[idx]

    def __repr__(self) -> str:
        # Updated repr format
        area = f"{self.area():.3f}" if _SHAPELY_AVAILABLE else "?"
        return f"<plx.geometry.Polygon3D rings={len(self)} area={area}>"


# ---------------------------------------------------------------------------
# Volume Models (Plaxis 3D Compatible) --------------------------------------
# ---------------------------------------------------------------------------
class Volume(GeometryBase):
    """
    Abstract base class for a 3D geometric volume.
    Represents a closed body in 3D space.
    """

    def __init__(self):
        super().__init__()
        self._faces = []

    def get_faces(self) -> List["Polygon3D"]:
        """Return the list of Polygon3D faces that define the volume."""
        return self._faces
    
    def volume(self) -> float:
        """Calculate the volume of the body."""
        raise NotImplementedError("Subclass must implement the volume() method.")
    
    def centroid(self) -> Point:
        total_volume = self.volume()
        # If the volume is almost zero, then return the average coordinates of all vertices.
        if math.isclose(total_volume, 0.0, abs_tol=1e-9):
            all_points = [p for face in self._faces for p in face.get_all_points()]
            avg_x = sum(p.x for p in all_points) / len(all_points)
            avg_y = sum(p.y for p in all_points) / len(all_points)
            avg_z = sum(p.z for p in all_points) / len(all_points)
            return Point(avg_x, avg_y, avg_z)
        # When the volume is non-zero, more complex centroid algorithms can be implemented here. For now, the vertex average value is used as an approximation.
        all_points = [p for face in self._faces for p in face.get_all_points()]
        avg_x = sum(p.x for p in all_points) / len(all_points)
        avg_y = sum(p.y for p in all_points) / len(all_points)
        avg_z = sum(p.z for p in all_points) / len(all_points)
        return Point(avg_x, avg_y, avg_z)
    
    def __repr__(self) -> str:
        # Updated repr format
        return f"<plx.geometry.{self.__class__.__name__} id={self._id}>"

class Cube(Volume):
    """
    A simple cuboid volume defined by two opposing corners.
    """
    __slots__ = ("_min_point", "_max_point")

    def __init__(self, min_point: Point, max_point: Point):
        super().__init__()
        if not isinstance(min_point, Point) or not isinstance(max_point, Point):
            raise TypeError("Cube must be initialized with two Point instances.")
        self._min_point = min_point
        self._max_point = max_point
        # A cube is a simple volume, we don't necessarily need to generate faces
        # unless it's required for rendering or complex operations.

    @classmethod
    def from_center_and_size(cls, center: Point, dx: float, dy: float, dz: float):
        """Create a Cube from a center point and its dimensions."""
        min_p = Point(center.x - dx/2, center.y - dy/2, center.z - dz/2)
        max_p = Point(center.x + dx/2, center.y + dy/2, center.z + dz/2)
        return cls(min_p, max_p)

    def volume(self) -> float:
        """Calculate the volume of the cube."""
        dx = self._max_point.x - self._min_point.x
        dy = self._max_point.y - self._min_point.y
        dz = self._max_point.z - self._min_point.z
        return abs(dx * dy * dz)

    def centroid(self) -> Point:
        """Calculate the centroid (center point) of the cube."""
        cx = (self._min_point.x + self._max_point.x) / 2
        cy = (self._min_point.y + self._max_point.y) / 2
        cz = (self._min_point.z + self._max_point.z) / 2
        return Point(cx, cy, cz)

    @property
    def min_point(self) -> Point:
        return self._min_point

    @property
    def max_point(self) -> Point:
        return self._max_point

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Cube",
            "id": self._uuid_to_str(getattr(self, "_id", None)),
            "plx_id": self._plx_id,
            "min_point": self._min_point.to_dict(),
            "max_point": self._max_point.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cube":
        cube = cls(
            Point.from_dict(data["min_point"]),
            Point.from_dict(data["max_point"])
        )
        cube.id = data["id"]
        cube.plx_id = data.get("plx_id")
        return cube

    def __repr__(self) -> str:
        # Updated repr format
        return f"<plx.geometry.Cube id={self._id} min={self._min_point.get_point()} max={self._max_point.get_point()}>"


class Polyhedron(Volume):
    """
    A general volume represented by a collection of closed Polygon3D faces.
    This class is highly compatible with Plaxis 3D's definition of a volume.
    """
    def __init__(self, faces: List[Polygon3D]):
        super().__init__()
        if not isinstance(faces, list) or not all(isinstance(f, Polygon3D) for f in faces):
            raise TypeError("Polyhedron must be initialized with a list of Polygon3D faces.")
        
        # Validation: Check if the faces form a closed, watertight manifold
        # This is complex, so we'll just check if faces exist and are valid polygons.
        # A more advanced implementation would check for connectivity and closure.
        if not faces:
             raise ValueError("Polyhedron must have at least one face.")
        
        self._faces = faces

    # --- Volume calculation using Divergence Theorem (Gauss's theorem) ---
    def volume(self) -> float:
        """
        Calculates the volume of the polyhedron using the divergence theorem.
        Assumes faces are consistently oriented (outward-pointing normals).
        """
        total_volume = 0.0
        for face in self._faces:
            # We assume the polygon is planar, so we can use its first three points
            # to define the normal vector
            p1, p2, p3 = face[0][0], face[0][1], face[0][2]
            
            # Normal vector components
            normal_x = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y)
            normal_y = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z)
            normal_z = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
            
            # Average point of the face (centroid)
            face_centroid_x = sum(p.x for p in face.get_points()) / len(face.get_points())
            face_centroid_y = sum(p.y for p in face.get_points()) / len(face.get_points())
            face_centroid_z = sum(p.z for p in face.get_points()) / len(face.get_points())
            
            # The contribution of this face to the total volume
            total_volume += (
                face_centroid_x * normal_x + 
                face_centroid_y * normal_y + 
                face_centroid_z * normal_z
            ) * face.area()

        return abs(total_volume) / 3.0

    def centroid(self) -> Point:
        """
        Calculates the centroid of the polyhedron.
        Assumes faces are consistently oriented and the volume calculation is correct.
        """
        # This is a complex calculation, especially for concave polyhedra.
        # A simple method involves summing weighted face centroids, but requires
        # careful handling of tetrahedra volumes.
        # Let's provide a basic implementation for convex polyhedra.
        
        total_volume = self.volume()
        if math.isclose(total_volume, 0.0, abs_tol=1e-9):
            all_points = [p for face in self._faces for p in face.get_points()]
            avg_x = sum(p.x for p in all_points) / len(all_points)
            avg_y = sum(p.y for p in all_points) / len(all_points)
            avg_z = sum(p.z for p in all_points) / len(all_points)
            return Point(avg_x, avg_y, avg_z)
        
        # Placeholder for a full centroid calculation.
        all_points = [p for face in self._faces for p in face.get_points()]
        avg_x = sum(p.x for p in all_points) / len(all_points)
        avg_y = sum(p.y for p in all_points) / len(all_points)
        avg_z = sum(p.z for p in all_points) / len(all_points)
        
        return Point(avg_x, avg_y, avg_z)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Polyhedron",
            "id": self._id,
            "plx_id": self._plx_id,
            "faces": [f.to_dict() for f in self._faces]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Polyhedron":
        faces = [Polygon3D.from_dict(f) for f in data["faces"]]
        poly = cls(faces)
        poly.id = data["id"]
        poly.plx_id = data.get("plx_id")
        return poly

    def __repr__(self) -> str:
        # Updated repr format
        vol = f"{self.volume():.3f}" if self._faces else "?"
        return f"<plx.geometry.Polyhedron id={self._id} n_faces={len(self._faces)} volume={vol}>"

# ---------------------------------------------------------------------------
# Utility - planar ring closure check (foundation pit walls) -----------------
# ---------------------------------------------------------------------------
def rings_close_to_footprint(wall_faces: List[Polygon3D], footprint: Polygon3D, tol: float = 1e-3) -> bool:
    """Return *True* if the XY union of wall top edges encloses the footprint within *tol* m²."""
    if not _SHAPELY_AVAILABLE:
        raise RuntimeError("Shapely is required for this operation. Please install shapely.")

    from shapely.geometry import LineString as _ShpLine, Polygon as _ShpPolygon
    from shapely.ops import unary_union as _shp_union, polygonize as _shp_polygonize
    # Collect top rims (assume first ring of each face is full profile; take first two points for top edge)
    rims = [_ShpLine([(pt.x, pt.y) for pt in face.get_lines()[0].get_points()]) for face in wall_faces]
    merged = _shp_union(rims)
    polys = list(_shp_polygonize(merged))
    if not polys:
        return False
    wall_poly = max(polys, key=lambda p: p.area)
    return wall_poly.symmetric_difference(footprint.to_shapely()).area < tol
