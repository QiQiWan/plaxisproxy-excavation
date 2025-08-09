import unittest
from plaxisproxy_excavation.structures.soilblock import SoilBlock

# Define a dummy material with a 'name' attribute for testing
class DummyMaterial:
    def __init__(self, name):
        self.name = name

class TestSoilBlock(unittest.TestCase):
    def test_initial_properties(self):
        """SoilBlock should store provided material and geometry on init."""
        mat = DummyMaterial("SoilTypeA")
        geom = [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]  # geometry as list of coordinates
        sb = SoilBlock("Block1", comment="Test block", material=mat, geometry=geom)
        self.assertEqual(sb.name, "Block1")
        self.assertEqual(sb.comment, "Test block")
        self.assertIs(sb.material, mat)
        self.assertEqual(sb.geometry, geom)

    def test_setters(self):
        """set_material and set_geometry should update the properties."""
        sb = SoilBlock("Block2")
        mat2 = DummyMaterial("SoilTypeB")
        sb.set_material(mat2)
        self.assertIs(sb.material, mat2)
        geom2 = [(0.0, 0.0, 0.0)]
        sb.set_geometry(geom2)
        self.assertEqual(sb.geometry, geom2)

    def test_from_dict_core(self):
        """_from_dict_core should create a SoilBlock with data from dictionary."""
        mat = DummyMaterial("MatX")
        geom = [(1.0, 1.0, 1.0)]
        data = {"name": "BlockX", "comment": "C", "material": mat, "geometry": geom}
        sb = SoilBlock._from_dict_core(data)
        self.assertIsInstance(sb, SoilBlock)
        self.assertEqual(sb.name, "BlockX")
        self.assertEqual(sb.comment, "C")
        self.assertIs(sb.material, mat)
        self.assertEqual(sb.geometry, geom)

    def test_repr_output(self):
        """__repr__ should reflect material name, geometry status, and sync state."""
        mat = DummyMaterial("MatY")
        sb = SoilBlock("BlockY", material=mat)
        # Not synced (plx_volume_id None) -> 'unsynced'; no geometry -> 'geom=None'
        repr_str = repr(sb)
        self.assertIn(f"mat='{mat.name}'", repr_str)
        self.assertIn("geom=None", repr_str)
        self.assertIn("unsynced", repr_str)
        # If geometry is set and plx_volume_id is provided, they should reflect in repr
        sb.set_geometry([(0.0, 0.0, 0.0)])
        sb._plx_volume_id = "123"
        repr_str2 = repr(sb)
        self.assertIn("geom=set", repr_str2)
        self.assertIn("123", repr_str2)

if __name__ == '__main__':
    unittest.main()
