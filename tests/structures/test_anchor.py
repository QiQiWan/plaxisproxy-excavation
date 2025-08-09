import unittest
from plaxisproxy_excavation.structures.anchor import Anchor
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D

class TestAnchor(unittest.TestCase):
    def setUp(self):
        # Create a valid Line3D with exactly two points for use in tests
        pts = PointSet([Point(0, 0, 0), Point(0, 0, 1)])
        self.line2 = Line3D(pts)

    def test_valid_anchor(self):
        """Anchor initialization with a valid 2-point line."""
        anchor = Anchor("Anchor1", self.line2, anchor_type="TypeA")
        # The stored line and type should match inputs
        self.assertIs(anchor.line, self.line2)
        self.assertEqual(anchor.anchor_type, "TypeA")
        # get_points should return the list of the two Point objects
        points = anchor.get_points()
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].get_point(), (0.0, 0.0, 0.0))
        self.assertEqual(points[1].get_point(), (0.0, 0.0, 1.0))

    def test_invalid_line_length(self):
        """Anchor line must have exactly two points; otherwise ValueError is raised."""
        # Line with only 1 point -> should raise ValueError
        line1 = Line3D(PointSet([Point(0, 0, 0)]))
        with self.assertRaises(ValueError):
            Anchor("A2", line1, anchor_type="TypeA")
        # Line with 3 points -> should raise ValueError
        line3 = Line3D(PointSet([Point(0, 0, 0), Point(1, 1, 1), Point(2, 2, 2)]))
        with self.assertRaises(ValueError):
            Anchor("A3", line3, anchor_type="TypeA")

    def test_repr(self):
        """__repr__ returns a constant string for Anchor objects."""
        anchor = Anchor("AnchorRepr", self.line2, anchor_type="TypeB")
        self.assertEqual(repr(anchor), "<plx.structures.anchor>")

if __name__ == '__main__':
    unittest.main()
