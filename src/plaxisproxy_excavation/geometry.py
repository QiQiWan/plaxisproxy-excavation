from __future__ import annotations
import uuid
import math
from typing import List, Iterator, Optional, Tuple, Dict, Any


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


class GeometryBase:
    
    _id: uuid.UUID = uuid.uuid4()
    _plx_id: Optional[str] = None

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value):
        self._id = value

    @property
    def plx_id(self):
        return self._plx_id
    
    @plx_id.setter
    def plx_id(self, value):
        self.plx_id = value

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
        if not isinstance(point_set, PointSet):
            raise TypeError("Line3D must be initialized with a PointSet instance.")
        self._id = uuid.uuid4()
        self._plx_id = None
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
        return len({(p.x, p.y) for p in pts[:-1]}) >= 3

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
            "id": self._uuid_to_str(self._id),
            "plx_id": self._plx_id,
            "points": [p.to_dict() for p in self._point_set.get_points()]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Line3D":
        pts = [Point.from_dict(p) for p in data["points"]]
        line = cls(PointSet(pts))
        line.id = cls._str_to_uuid(data["id"])   
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

    def __init__(self, lines: Optional[List[Line3D]] = None):
        self._lines: List[Line3D] = lines if lines else []

        if not self._lines:
            return
        
        self.update_points()

        # Validation for all rings
        for i, ln in enumerate(self._lines):
            if not ln.is_valid_ring():
                raise ValueError(f"Ring at index {i} is invalid - must be closed with ≥3 unique planar points.")
        
        # Check if all rings are co-planar
        self._check_coplanar()

    def _check_coplanar(self) -> None:
        """Internal check to ensure all rings are on the same plane."""
        if len(self._lines) == 0:
            return
        
        # Use the first 3 points of the outer ring to define a plane
        p1, p2, p3 = self._lines[0][0], self._lines[0][1], self._lines[0][2]
        
        # Calculate the normal vector of the plane
        vec_a = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        vec_b = (p3.x - p1.x, p3.y - p1.y, p3.z - p1.z)
        normal = (
            vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1],
            vec_a[2] * vec_b[0] - vec_a[0] * vec_b[2],
            vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]
        )
        
        # Equation of the plane: A*x + B*y + C*z = D
        D = normal[0] * p1.x + normal[1] * p1.y + normal[2] * p1.z
        
        # Check all points in all rings against this plane
        for ln in self._lines:
            for p in ln:
                if not math.isclose(normal[0] * p.x + normal[1] * p.y + normal[2] * p.z, D, abs_tol=1e-9):
                    raise ValueError("All points in a Polygon3D must be co-planar.")
                
    def as_tuple_list(self) -> List[Tuple[float, float, float]]:
        """Return all vertex coordinates as a list of (x, y, z) tuples."""
        if not self._lines:
            return []
        return [p.get_point() for p in self.get_all_points()]

    @classmethod
    def from_points(cls, point_set: PointSet):
        """Create a Polygon3D from a single PointSet forming the outer boundary."""
        if len(point_set) < 3:
            raise ValueError("Need at least 3 points for a polygon boundary.")
        if not point_set.is_closed():
            pts = point_set.get_points()
            point_set.add_point(pts[0].x, pts[0].y, pts[0].z)
        return cls([Line3D(point_set)])

    # ---------------- mutation ---------------------------------------
    def add_hole(self, line: Line3D) -> None:
        """Add an inner hole (closed ring) to the polygon."""
        if not line.is_valid_ring():
            raise ValueError("Hole to add is an invalid ring.")
        self._lines.append(line)
        self.update_points()

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

    # ---------------- container --------------------------------------
    def get_lines(self) -> List[Line3D]:
        """Return the list of Line3D rings that define the polygon."""
        return self._lines

    def get_all_points(self) -> List[Point]:
        """Return a flat list of all points from all rings."""
        pts: List[Point] = []
        for ln in self._lines:
            pts.extend(ln.get_points())
        return pts
    
    def update_points(self) -> None:
        """Update the point set in the Polygon3D"""
        self._point_set = PointSet(self.get_all_points())

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
            "id": self._uuid_to_str(self._id),
            "plx_id": self._plx_id,
            "rings": [ln.to_dict() for ln in self._lines]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Polygon3D":
        rings = [Line3D.from_dict(r) for r in data["rings"]]
        poly = cls(rings)
        poly.id = cls._str_to_uuid(data["id"])
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
    _id: uuid.UUID
    _plx_id: Optional[str]
    _faces: List["Polygon3D"]

    def __init__(self):
        self._id = uuid.uuid4()
        self._plx_id = None
        self._faces = []

    def get_faces(self) -> List["Polygon3D"]:
        """Return the list of Polygon3D faces that define the volume."""
        return self._faces
    
    def volume(self) -> float:
        """Calculate the volume of the body."""
        raise NotImplementedError("Subclass must implement the volume() method.")
    
    def centroid(self) -> Point:
        """Calculate the geometric centroid of the body."""
        raise NotImplementedError("Subclass must implement the centroid() method.")
    
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
            "id": self._uuid_to_str(self._id),
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
        cube.id = cls._str_to_uuid(data["id"])
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
            face_centroid_x = sum(p.x for p in face.get_all_points()) / len(face.get_all_points())
            face_centroid_y = sum(p.y for p in face.get_all_points()) / len(face.get_all_points())
            face_centroid_z = sum(p.z for p in face.get_all_points()) / len(face.get_all_points())
            
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
            raise ValueError("Cannot calculate centroid of a zero-volume polyhedron.")
        
        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
        
        # Pick an arbitrary origin point (e.g., the first point of the first face)
        origin = self._faces[0][0][0]
        
        for face in self._faces:
            # Create tetrahedra from origin to each face
            face_centroid = Point(
                sum(p.x for p in face.get_all_points()) / len(face.get_all_points()),
                sum(p.y for p in face.get_all_points()) / len(face.get_all_points()),
                sum(p.z for p in face.get_all_points()) / len(face.get_all_points())
            )
            
            # Tetrahedra centroid
            tetra_centroid_x = (origin.x + face_centroid.x) / 4
            tetra_centroid_y = (origin.y + face_centroid.y) / 4
            tetra_centroid_z = (origin.z + face_centroid.z) / 4
            
            # Need to get a tetrahedron volume using a cross-product based method
            # This is complex. Let's simplify for now to illustrate the concept.
            # The most robust method for general polyhedra is Green's Theorem-based,
            # but this would require a major rewrite.
            
            # For a placeholder, we can just return a simple average of all points,
            # but this is not a true centroid for non-uniform shapes.
            # A more robust implementation is beyond the scope of a simple example,
            # so we'll leave it as a conceptual placeholder.
            pass
            
        # Placeholder for a full centroid calculation.
        all_points = [p for face in self._faces for p in face.get_all_points()]
        avg_x = sum(p.x for p in all_points) / len(all_points)
        avg_y = sum(p.y for p in all_points) / len(all_points)
        avg_z = sum(p.z for p in all_points) / len(all_points)
        
        return Point(avg_x, avg_y, avg_z)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Polyhedron",
            "id": self._uuid_to_str(self._id),
            "plx_id": self._plx_id,
            "faces": [f.to_dict() for f in self._faces]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Polyhedron":
        faces = [Polygon3D.from_dict(f) for f in data["faces"]]
        poly = cls(faces)
        poly.id = cls._str_to_uuid(data["id"])
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
