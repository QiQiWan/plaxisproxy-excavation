import unittest
from plaxisproxy_excavation.structures.beam import Beam
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D

class TestBeam(unittest.TestCase):
    def setUp(self):
        # Create a 2-point Line3D (beam span)
        pts = PointSet([Point(0, 0, 0), Point(1, 0, 0)])
        self.line2 = Line3D(pts)

    def test_valid_beam(self):
        """Beam initialization with valid 2-point line."""
        beam = Beam("Beam1", self.line2, beam_type="TypeX")
        # Properties should match initialization
        self.assertIs(beam.line, self.line2)
        self.assertEqual(beam.beam_type, "TypeX")
        # get_points returns the two Point objects in the line
        pts = beam.get_points()
        self.assertEqual(len(pts), 2)
        self.assertEqual(pts[0].get_point(), (0.0, 0.0, 0.0))
        self.assertEqual(pts[1].get_point(), (1.0, 0.0, 0.0))

    def test_invalid_line_length(self):
        """Beam line must have exactly two points; otherwise ValueError is raised."""
        line1 = Line3D(PointSet([Point(1, 1, 1)]))  # single point
        with self.assertRaises(ValueError):
            Beam("B2", line1, beam_type="TypeX")
        line3 = Line3D(PointSet([Point(0, 0, 0), Point(0, 1, 0), Point(0, 2, 0)]))  # three points
        with self.assertRaises(ValueError):
            Beam("B3", line3, beam_type="TypeX")

    def test_repr(self):
        """__repr__ returns the expected constant string for Beam."""
        beam = Beam("BeamRepr", self.line2, beam_type="TypeY")
        self.assertEqual(repr(beam), "<plx.structures.beam>")

if __name__ == '__main__':
    unittest.main()
