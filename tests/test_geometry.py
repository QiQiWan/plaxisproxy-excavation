import math
import uuid
import unittest
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D, Polygon3D, Cube, Polyhedron, Volume

class TestGeometry(unittest.TestCase):
    def test_point_basic(self):
        """Point creation and basic properties (x, y, z)."""
        p = Point(1.2, 3.4, -5.6)
        # Coordinates should be stored as float and accessible via properties
        self.assertAlmostEqual(p.x, 1.2)
        self.assertAlmostEqual(p.y, 3.4)
        self.assertAlmostEqual(p.z, -5.6)
        self.assertEqual(p.get_point(), (1.2, 3.4, -5.6))
        # __repr__ contains coordinates to 3 decimal places
        rep = repr(p)
        self.assertIn("x=1.200", rep)
        self.assertIn("y=3.400", rep)
        self.assertIn("z=-5.600", rep)

    def test_point_distance_and_equality(self):
        """Point.distance_to computes correct Euclidean distance; equality uses tolerance."""
        p1 = Point(0, 0, 0)
        p2 = Point(3, 4, 0)
        # Distance between (0,0,0) and (3,4,0) should be 5.0
        self.assertAlmostEqual(p1.distance_to(p2), 5.0)
        # Non-Point argument raises TypeError
        with self.assertRaises(TypeError):
            p1.distance_to((3,4,0))
        # Points that differ by small tolerance are equal
        p3 = Point(0.0, 0.0, 0.0)
        p4 = Point(0.0, 0.0, 1e-10)  # very small difference
        self.assertTrue(p3 == p4)
        # Hash of equal points should be the same
        self.assertEqual(hash(p3), hash(Point(0.0, 0.0, 0.0)))

    def test_point_to_shapely_not_available(self):
        """Point.to_shapely raises RuntimeError if Shapely not available."""
        # Force shapely unavailable if not already
        import plaxisproxy_excavation.geometry as geom_mod
        geom_mod._SHAPELY_AVAILABLE = False
        p = Point(1, 2, 3)
        with self.assertRaises(RuntimeError):
            p.to_shapely()

    def test_point_serialization(self):
        """Point to_dict and from_dict preserve data (including optional id)."""
        p = Point(7, 8, 9)
        p_dict = p.to_dict()
        # Should contain coordinates and id (as str or None)
        self.assertEqual(p_dict["x"], 7.0)
        self.assertEqual(p_dict["y"], 8.0)
        self.assertEqual(p_dict["z"], 9.0)
        # id in dict is string
        self.assertIsInstance(p_dict.get("id"), (str, type(None)))
        # Create new point from dict and verify equality (coordinates and id)
        p2 = Point.from_dict(p_dict)
        self.assertEqual(p2.get_point(), (7.0, 8.0, 9.0))
        self.assertEqual(p2, p)  # __eq__ checks coordinates with tolerance
        # If id was present, the new point should carry it
        if p_dict.get("id"):
            self.assertEqual(str(p2.id), p_dict["id"])

    def test_pointset_operations(self):
        """PointSet add_point and is_closed logic."""
        pts = PointSet()
        pts.add_point(0, 0, 0)
        pts.add_point(1, 0, 0)
        self.assertEqual(len(pts), 2)
        # Initially not closed since first != last
        self.assertFalse(pts.is_closed())
        pts.add_point(0, 0, 0)  # add a point identical to first to close loop
        self.assertTrue(pts.is_closed())
        # get_points returns list of Point objects
        points_list = pts.get_points()
        self.assertIsInstance(points_list, list)
        self.assertIsInstance(points_list[0], Point)
        # Iteration and indexing
        coords = [tuple(p.get_point()) for p in pts]
        self.assertEqual(coords[0], (0.0, 0.0, 0.0))
        self.assertEqual(coords[1], (1.0, 0.0, 0.0))
        self.assertEqual(coords[2], (0.0, 0.0, 0.0))  # closed loop back to start
        self.assertEqual(pts[1].get_point(), (1.0, 0.0, 0.0))
        # __repr__ indicates number of points and closed status
        rep = repr(pts)
        self.assertIn("n_points=3", rep)
        self.assertIn("closed=True", rep)

    def test_pointset_to_shapely_no_shapely(self):
        """PointSet.to_shapely raises if Shapely not installed."""
        import plaxisproxy_excavation.geometry as geom_mod
        geom_mod._SHAPELY_AVAILABLE = False
        pts = PointSet([Point(0,0,0), Point(1,0,0)])
        with self.assertRaises(RuntimeError):
            pts.to_shapely()

    def test_pointset_serialization(self):
        """PointSet to_dict and from_dict preserve contained points."""
        pts = PointSet([Point(1,1,1), Point(2,2,2)])
        d = pts.to_dict()
        # 'points' key with list of point dicts
        self.assertIn("points", d)
        self.assertEqual(len(d["points"]), 2)
        # Reconstruct
        pts2 = PointSet.from_dict(d)
        self.assertEqual(len(pts2), 2)
        self.assertEqual(pts2[0].get_point(), (1.0, 1.0, 1.0))
        self.assertEqual(pts2[1].get_point(), (2.0, 2.0, 2.0))

    def test_line3d_basic_and_length(self):
        """Line3D initialization, length calculation and closed ring check."""
        ps = PointSet([Point(0,0,0), Point(3,4,0)])
        line = Line3D(ps)
        # Underlying PointSet should be used
        self.assertEqual(line.get_points(), ps.get_points())
        # add_point delegates to PointSet
        line.add_point(3, 4, 1)
        self.assertEqual(len(line), 3)
        # length: distance from (0,0,0)->(3,4,0) plus (3,4,0)->(3,4,1)
        self.assertAlmostEqual(line.length, 5.0 + 1.0)  # 5 (from first two) + 1 (vertical segment)
        # is_closed: closed if first == last
        self.assertFalse(line.is_closed())
        # make closed by adding a point identical to start
        line.add_point(0, 0, 0)
        self.assertTrue(line.is_closed())
        # is_valid_ring: requires closed and at least 3 unique planar points
        self.assertFalse(line.is_valid_ring())  # currently 4 points but only 2 unique (vertical line)
        # Create a valid closed triangle ring
        tri_pts = PointSet([Point(0,0,0), Point(1,0,0), Point(0,1,0)])
        tri_pts.add_point(0,0,0)  # close it
        ring = Line3D(tri_pts)
        self.assertTrue(ring.is_closed())
        self.assertTrue(ring.is_valid_ring())  # triangle with three unique points

    def test_line3d_vertical_and_xy_location(self):
        """Line3D vertical check and xy_location property."""
        # Vertical line: X and Y constant
        pts_vertical = PointSet([Point(2,2,0), Point(2,2,5)])
        line_vert = Line3D(pts_vertical)
        self.assertTrue(line_vert.is_vertical())
        self.assertEqual(line_vert.xy_location(), (2.0, 2.0))
        # Non-vertical line should raise ValueError for xy_location
        pts_nonvert = PointSet([Point(0,0,0), Point(1,0,1)])
        line_nonvert = Line3D(pts_nonvert)
        self.assertFalse(line_nonvert.is_vertical())
        with self.assertRaises(ValueError):
            line_nonvert.xy_location()

    def test_line3d_equality_and_repr(self):
        """Line3D equality and representation."""
        pts1 = PointSet([Point(0,0,0), Point(1,0,0)])
        pts1.add_point(0,0,0)  # close
        pts2 = PointSet([Point(0,0,0), Point(1,0,0)])
        pts2.add_point(0,0,0)  # close
        line_a = Line3D(pts1)
        line_b = Line3D(pts2)
        self.assertEqual(line_a, line_b)
        # __repr__ includes id and number of points and closed status
        rep = repr(line_a)
        self.assertIn("Line3D", rep)
        self.assertIn("points=3", rep)
        self.assertIn("closed=True", rep)
        # __iter__ and __getitem__
        coords = [p.get_point() for p in line_a]
        self.assertEqual(coords[0], (0.0, 0.0, 0.0))
        self.assertEqual(line_a[1].get_point(), (1.0, 0.0, 0.0))

    def test_polygon3d_validation_and_properties(self):
        """Polygon3D ring validation, co-planarity and geometric properties."""
        # Prepare a valid outer ring (triangle closed)
        pts = PointSet([Point(0,0,0), Point(1,0,0), Point(0,1,0)])
        pts.add_point(0,0,0)
        outer_line = Line3D(pts)
        # Valid polygon with one ring
        poly = Polygon3D([outer_line])
        # area: triangle area = 0.5
        self.assertAlmostEqual(poly.area(), 0.5, places=6)
        # perimeter equals outer ring length
        self.assertAlmostEqual(poly.perimeter(), outer_line.length)
        self.assertTrue(poly.is_valid())
        # outer_ring and inner_rings properties
        self.assertIs(poly.outer_ring, outer_line)
        self.assertEqual(poly.inner_rings, [])
        # __len__, __iter__, __getitem__
        self.assertEqual(len(poly), 1)
        for ring in poly:  # iteration yields rings
            self.assertIsInstance(ring, Line3D)
        self.assertIs(poly[0], outer_line)
        # __repr__ contains rings count and area (or '?' if shapely not available)
        rep = repr(poly)
        self.assertIn("rings=1", rep)
        self.assertIn("area=", rep)

        # Add an invalid hole (not closed) should raise ValueError
        open_pts = PointSet([Point(0,0,0), Point(0.5,0.5,0.0)])  # not closed
        open_line = Line3D(open_pts)
        with self.assertRaises(ValueError):
            poly.add_hole(open_line)
        # Add a valid hole
        hole_pts = PointSet([Point(0.2,0.2,0), Point(0.3,0.2,0), Point(0.25,0.3,0)])
        hole_pts.add_point(0.2,0.2,0)  # close the small triangle
        hole_line = Line3D(hole_pts)
        poly.add_hole(hole_line)
        self.assertEqual(len(poly.get_lines()), 2)
        # After adding hole, update_points should refresh _point_set
        all_points = poly.get_all_points()
        self.assertEqual(len(all_points), len(poly.outer_ring.get_points()) + len(hole_line.get_points()))
        # Invalid polygon: outer ring not closed
        bad_pts = PointSet([Point(0,0,0), Point(1,0,0), Point(0,1,1)])  # not planar
        bad_line = Line3D(bad_pts)
        # Closing the ring
        bad_pts.add_point(0,0,0)
        with self.assertRaises(ValueError):
            Polygon3D([bad_line])  # not co-planar, should raise

    def test_polygon3d_from_points(self):
        """Polygon3D.from_points should create a closed polygon or raise if insufficient points."""
        ps = PointSet([Point(0,0,0), Point(1,0,0)])
        # Less than 3 points should error
        with self.assertRaises(ValueError):
            Polygon3D.from_points(ps)
        # Provide exactly 3 points (triangle), not closed -> method should close it
        tri_pts = PointSet([Point(0,0,0), Point(1,0,0), Point(0,1,0)])
        poly = Polygon3D.from_points(tri_pts)
        # Polygon should have one ring that is closed
        self.assertEqual(len(poly), 1)
        self.assertTrue(poly.outer_ring.is_closed())
        # The automatically added closing point should equal the first point
        ring_points = poly.outer_ring.get_points()
        self.assertEqual(ring_points[0], ring_points[-1])

    def test_polygon3d_to_shapely_no_shapely(self):
        """Polygon3D.to_shapely raises if Shapely not installed."""
        import plaxisproxy_excavation.geometry as geom_mod
        geom_mod._SHAPELY_AVAILABLE = False
        pts = PointSet([Point(0,0,0), Point(1,0,0), Point(0,1,0)])
        pts.add_point(0,0,0)
        poly = Polygon3D([Line3D(pts)])
        with self.assertRaises(RuntimeError):
            poly.to_shapely()

    def test_polygon3d_serialization(self):
        """Polygon3D to_dict and from_dict round-trip."""
        pts = PointSet([Point(0,0,0), Point(1,0,0), Point(0,1,0)])
        pts.add_point(0,0,0)
        poly = Polygon3D([Line3D(pts)])
        d = poly.to_dict()
        # Should contain rings list with dicts
        self.assertIn("rings", d)
        self.assertEqual(len(d["rings"]), 1)
        poly2 = Polygon3D.from_dict(d)
        # Outer ring points should match original
        orig_coords = [p.get_point() for p in poly.outer_ring.get_points()]
        new_coords = [p.get_point() for p in poly2.outer_ring.get_points()]
        self.assertEqual(orig_coords, new_coords)
        # Should preserve id if present
        if d.get("id"):
            self.assertEqual(uuid.UUID(d["id"]), poly2.id)

    def test_volume_and_subclasses(self):
        """Volume base class and Cube/Polyhedron functionality."""
        # Volume is abstract: volume() and centroid() not implemented
        vol = Volume()
        with self.assertRaises(NotImplementedError):
            vol.volume()
        with self.assertRaises(NotImplementedError):
            vol.centroid()
        # __repr__ includes class name and id
        rep = repr(vol)
        self.assertTrue(rep.startswith("<plx.geometry.Volume id="))

        # Cube initialization type checking
        with self.assertRaises(TypeError):
            Cube("not a Point", Point(1,1,1))
        with self.assertRaises(TypeError):
            Cube(Point(0,0,0), "not a Point")
        # Valid cube
        p_min = Point(0,0,0)
        p_max = Point(2,2,2)
        cube = Cube(p_min, p_max)
        # Volume = 2*2*2 = 8
        self.assertAlmostEqual(cube.volume(), 8.0)
        # Centroid = midpoint -> (1,1,1)
        center = cube.centroid()
        self.assertIsInstance(center, Point)
        self.assertEqual(center.get_point(), (1.0, 1.0, 1.0))
        # Properties min_point and max_point
        self.assertEqual(cube.min_point.get_point(), (0.0,0.0,0.0))
        self.assertEqual(cube.max_point.get_point(), (2.0,2.0,2.0))
        # from_center_and_size produces correct min/max
        center_pt = Point(0,0,0)
        cube2 = Cube.from_center_and_size(center_pt, 2, 4, 6)
        # half lengths: dx/2=1, dy/2=2, dz/2=3 -> min = (-1,-2,-3), max = (1,2,3)
        self.assertEqual(cube2.min_point.get_point(), (-1.0,-2.0,-3.0))
        self.assertEqual(cube2.max_point.get_point(), (1.0,2.0,3.0))
        # Cube to_dict/from_dict
        d = cube.to_dict()
        self.assertEqual(d["type"], "Cube")
        cube3 = Cube.from_dict(d)
        self.assertEqual(cube3.min_point.get_point(), cube.min_point.get_point())
        self.assertEqual(cube3.max_point.get_point(), cube.max_point.get_point())
        if d.get("id"):
            self.assertEqual(uuid.UUID(d["id"]), cube3.id)
        # __repr__ includes min and max coordinates
        rep_cube = repr(cube)
        self.assertIn("min=(0.0, 0.0, 0.0)", rep_cube)
        self.assertIn("max=(2.0, 2.0, 2.0)", rep_cube)

        # Polyhedron initialization errors and volume/centroid basics
        with self.assertRaises(TypeError):
            Polyhedron("not a list")
        with self.assertRaises(ValueError):
            Polyhedron([])  # empty list
        # Create a simple Polyhedron with one face (triangle) 
        face_pts = PointSet([Point(0,0,0), Point(1,0,0), Point(0,1,0)])
        face_pts.add_point(0,0,0)
        face = Polygon3D([Line3D(face_pts)])
        polyh = Polyhedron([face])
        # Volume should be a float (likely zero since only one face)
        vol_val = polyh.volume()
        self.assertIsInstance(vol_val, float)
        self.assertGreaterEqual(vol_val, 0.0)
        # Centroid should return a Point
        centroid = polyh.centroid()
        self.assertIsInstance(centroid, Point)
