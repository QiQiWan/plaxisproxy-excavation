import unittest
from plaxisproxy_excavation.structures.basestructure import BaseStructure

class TestBaseStructure(unittest.TestCase):
    def test_init_and_properties(self):
        """Test BaseStructure initialization and basic properties."""
        bs = BaseStructure("TestName")
        # name and comment should be set correctly
        self.assertEqual(bs.name, "TestName")
        self.assertEqual(bs.comment, "")
        # id should be a non-empty string (UUID)
        self.assertTrue(isinstance(bs.id, str) and bs.id)

        # Providing an explicit comment
        bs2 = BaseStructure("Name2", comment="Some comment")
        self.assertEqual(bs2.comment, "Some comment")

    def test_invalid_name(self):
        """Name must be a non-empty string, otherwise ValueError is raised."""
        with self.assertRaises(ValueError):
            BaseStructure("")  # empty name not allowed
        with self.assertRaises(ValueError):
            BaseStructure(123)  # name must be a string

    def test_repr(self):
        """__repr__ should return the expected string format."""
        bs = BaseStructure("X")
        expected = "<plx.structures.BaseStructure name='X'>"
        self.assertEqual(repr(bs), expected)

if __name__ == '__main__':
    unittest.main()
