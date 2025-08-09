import unittest
from plaxisproxy_excavation.structures.embeddedpile import EmbeddedPile
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D

class TestEmbeddedPile(unittest.TestCase):
    def setUp(self):
        # Line3D with two points for a valid embedded pile
        pts = PointSet([Point(0, 0, 0), Point(0, 0, 5)])
        self.line2 = Line3D(pts)

    def test_valid_embedded_pile(self):
        """EmbeddedPile initialization with correct Line3D and pile_type."""
        ep = EmbeddedPile("Pile1", self.line2, pile_type="PileTypeA")
        self.assertIs(ep.line, self.line2)
        self.assertEqual(ep.pile_type, "PileTypeA")
        # get_points should return the two points
        points = ep.get_points()
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].get_point(), (0.0, 0.0, 0.0))
        self.assertEqual(points[1].get_point(), (0.0, 0.0, 5.0))
        # length() should return the distance between the two points (5.0 here)
        self.assertAlmostEqual(ep.length(), 5.0, places=6)

    def test_invalid_line(self):
        """Line must be Line3D with exactly two points; otherwise ValueError is raised."""
        not_line = [Point(0, 0, 0), Point(1, 1, 1)]  # not a Line3D instance
        with self.assertRaises(ValueError):
            EmbeddedPile("Pile2", not_line, pile_type="PileTypeA")
        # Line3D with wrong number of points
        line1 = Line3D(PointSet([Point(0, 0, 0)]))
        with self.assertRaises(ValueError):
            EmbeddedPile("Pile3", line1, pile_type="PileTypeA")

    def test_repr(self):
        """__repr__ output should include the name and pile type."""
        ep = EmbeddedPile("PileRepr", self.line2, pile_type="TypeB")
        repr_str = repr(ep)
        # The repr should contain class name, provided name and type
        self.assertIn("EmbeddedPile", repr_str)
        self.assertIn("name='PileRepr'", repr_str)
        self.assertIn("type='TypeB'", repr_str)

if __name__ == '__main__':
    unittest.main()
