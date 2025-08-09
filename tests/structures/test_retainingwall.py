import unittest
from plaxisproxy_excavation.structures.retainingwall import RetainingWall
from plaxisproxy_excavation.geometry import Polygon3D

class TestRetainingWall(unittest.TestCase):
    def setUp(self):
        # Use an empty Polygon3D instance for the surface
        self.surface = Polygon3D()

    def test_valid_retainingwall(self):
        """RetainingWall initialization with correct Polygon3D surface."""
        rw = RetainingWall("Wall1", self.surface, plate_type="TypeP")
        self.assertIs(rw.surface, self.surface)
        self.assertEqual(rw.plate_type, "TypeP")

    def test_invalid_surface_type(self):
        """Surface must be a Polygon3D instance; otherwise TypeError is raised."""
        with self.assertRaises(TypeError):
            RetainingWall("Wall2", "not_a_polygon", plate_type="TypeP")

    def test_repr(self):
        """__repr__ output should include the name and plate type."""
        rw = RetainingWall("WallX", self.surface, plate_type="TypeQ")
        repr_str = repr(rw)
        self.assertIn("RetainingWall", repr_str)
        self.assertIn("name='WallX'", repr_str)
        self.assertIn("type='TypeQ'", repr_str)

if __name__ == '__main__':
    unittest.main()
