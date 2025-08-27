import unittest
from plaxisproxy_excavation.structures.well import Well, WellType
from plaxisproxy_excavation.geometry import Point, PointSet, Line3D

class TestWell(unittest.TestCase):
    def setUp(self):
        # Create a vertical Line3D (top and bottom points share x,y)
        self.top = Point(1, 2, 10)
        self.bottom = Point(1, 2, 0)
        self.line2 = Line3D(PointSet([self.top, self.bottom]))

    def test_valid_well(self):
        """Well initialization with valid inputs (vertical 2-point line and WellType)."""
        well = Well("Well1", self.line2, WellType.Extraction, h_min=5.0)
        # Check initial properties
        self.assertEqual(well.well_type, WellType.Extraction)
        self.assertEqual(well.h_min, 5.0)
        # pos should be (x, y) since line is vertical
        self.assertEqual(well.pos, (1.0, 2.0))
        # line property returns the internal Line3D
        self.assertIs(well.line, self.line2)

    def test_invalid_initial_args(self):
        """Invalid line length or well_type should raise ValueError."""
        short_line = Line3D(PointSet([Point(0, 0, 0)]))  # only one point
        with self.assertRaises(ValueError):
            Well("W1", short_line, WellType.Infiltration)
        # invalid well_type (not a WellType Enum member)
        with self.assertRaises(ValueError):
            Well("W2", self.line2, well_type="NotAWellType")

    def test_move_updates_line_and_position(self):
        """The move() method should shift the well's line and update its position."""
        well = Well("WellMove", self.line2, WellType.Infiltration)
        orig_points = well.get_points()
        # Move the well by given offsets
        well.move(dx=3.0, dy=-2.0, dz=1.0)
        new_points = well.get_points()
        self.assertEqual(len(new_points), 2)
        # Check that each point moved correctly
        self.assertAlmostEqual(new_points[0].x, orig_points[0].x + 3.0)
        self.assertAlmostEqual(new_points[0].y, orig_points[0].y - 2.0)
        self.assertAlmostEqual(new_points[0].z, orig_points[0].z + 1.0)
        self.assertAlmostEqual(new_points[1].x, orig_points[1].x + 3.0)
        self.assertAlmostEqual(new_points[1].y, orig_points[1].y - 2.0)
        self.assertAlmostEqual(new_points[1].z, orig_points[1].z + 1.0)
        # pos (x,y) should update to new coordinates
        self.assertEqual(well.pos, (orig_points[0].x + 3.0, orig_points[0].y - 2.0))

    def test_line_setter_validation(self):
        """Setting well.line should update pos or raise error if invalid line."""
        well = Well("WellLineSet", self.line2, WellType.Extraction)
        # Replace with a new valid 2-point vertical line
        new_line = Line3D(PointSet([Point(1, 2, 5), Point(1, 2, 1)]))
        well.line = new_line
        self.assertIs(well.line, new_line)
        # pos should update (same x,y as new_line)
        self.assertEqual(well.pos, (1.0, 2.0))
        # Setting a line with invalid length should raise ValueError
        bad_line = Line3D(PointSet([Point(0, 0, 0)]))
        with self.assertRaises(ValueError):
            well.line = bad_line

    def test_set_x_and_y(self):
        """Setting Well.x or Well.y updates the line and pos accordingly."""
        well = Well("WellPos", self.line2, WellType.Infiltration)
        # Change x coordinate
        old_y = well.pos[1]
        well.x = 5.0
        self.assertEqual(well.x, 5.0)
        self.assertEqual(well.y, old_y)
        # After setting x, both points' x should be updated
        pts = well.get_points()
        self.assertTrue(all(abs(p.x - 5.0) < 1e-9 for p in pts))
        # Change y coordinate
        old_x = well.pos[0]
        well.y = -3.0
        self.assertEqual(well.y, -3.0)
        self.assertEqual(well.x, old_x)
        pts2 = well.get_points()
        self.assertTrue(all(abs(p.y + 3.0) < 1e-9 for p in pts2))

    def test_well_type_setter(self):
        """Setting well_type only allows valid WellType values."""
        well = Well("WellTypeTest", self.line2, WellType.Extraction)
        # Change to a valid WellType
        well.well_type = WellType.Infiltration
        self.assertEqual(well.well_type, WellType.Infiltration)
        # Invalid assignment should raise ValueError
        with self.assertRaises(ValueError):
            well.well_type = "InvalidType"

    def test_h_min_setter(self):
        """h_min property should be settable with any float."""
        well = Well("WellHmin", self.line2, WellType.Extraction, h_min=1.0)
        well.h_min = 10.5
        self.assertEqual(well.h_min, 10.5)

    def test_repr(self):
        """__repr__ returns constant string for Well."""
        well = Well("WellRepr", self.line2, WellType.Extraction)
        self.assertEqual(
            repr(well), 
            "<plx.structures.Well name='WellRepr' type='Extraction' h_min=0.0>"
            )

if __name__ == '__main__':
    unittest.main()
