import unittest
from plaxisproxy_excavation.materials.basematerial import BaseMaterial

class TestBaseMaterial(unittest.TestCase):
    def test_properties_and_repr(self):
        """BaseMaterial stores initialization parameters correctly and __repr__ works."""
        material = BaseMaterial("TestMaterial", "DummyType", "Some comment", gamma=10.0, E=200.0, nu=0.3)
        self.assertEqual(material.type, "DummyType")
        self.assertEqual(material.comment, "Some comment")
        self.assertEqual(material.gamma, 10.0)
        self.assertEqual(material.E, 200.0)
        self.assertEqual(material.nu, 0.3)
        self.assertEqual(repr(material), "<plx.materials.BaseMaterial>")

    def test_invalid_name_raises(self):
        """Invalid names should trigger a validation error of the PlaxisObject."""
        with self.assertRaises(ValueError):
            BaseMaterial("", "DummyType", "No name", gamma=1.0, E=1.0, nu=0.1)

if __name__ == '__main__':
    unittest.main()
