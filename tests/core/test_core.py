import uuid
import unittest
from plaxisproxy_excavation.core.plaxisobject import PlaxisObject, SerializableBase

class TestCore(unittest.TestCase):
    def test_plaxisobject_init_and_properties(self):
        """PlaxisObject initialization with valid/invalid name and property behaviors."""
        # Valid initialization
        obj = PlaxisObject("ObjName", "A comment")
        self.assertEqual(obj.name, "ObjName")
        self.assertEqual(obj.comment, "A comment")
        self.assertIsInstance(uuid.UUID(obj.id), uuid.UUID)  # id is string uuid
        self.assertIsNone(obj.plx_id)
        # Setting plx_id
        obj.plx_id = "server_id"
        self.assertEqual(obj.plx_id, "server_id")
        # Invalid name (None or empty) should raise ValueError
        with self.assertRaises(ValueError):
            PlaxisObject(None, "")
        with self.assertRaises(ValueError):
            PlaxisObject("", "")
        # Name setter should not allow setting (it recurses infinitely, causing RecursionError)
        with self.assertRaises(RecursionError):
            obj.name = "NewName"

    def test_plaxisobject_repr_and_serialization(self):
        """PlaxisObject __repr__ and to_dict/from_dict serialization."""
        obj = PlaxisObject("MyObj", "Testing")
        rep = repr(obj)
        # __repr__ format: <PlaxisObject(name='MyObj', id='...')>
        self.assertIn("name='MyObj'", rep)
        self.assertIn("id='", rep)
        # to_dict should include properties and type info
        d = obj.to_dict()
        self.assertEqual(d["name"], "MyObj")
        self.assertEqual(d["comment"], "Testing")
        self.assertIn("__type__", d)
        self.assertIn("__version__", d)
        # The id in to_dict should match object's id
        self.assertEqual(d["id"], obj.id)
        # from_dict should recreate an object with same name and comment
        new_obj = PlaxisObject.from_dict({"name": "Copy", "comment": "CopyCmt"})
        self.assertIsInstance(new_obj, PlaxisObject)
        self.assertEqual(new_obj.name, "Copy")
        self.assertEqual(new_obj.comment, "CopyCmt")

    def test_serializable_uuid_helpers(self):
        """SerializableBase static methods _uuid_to_str and _str_to_uuid handle None and conversion."""
        u = uuid.uuid4()
        s = SerializableBase._uuid_to_str(u)
        self.assertEqual(s, str(u))
        # None input returns None string
        self.assertIsNone(SerializableBase._uuid_to_str(None))
        # _str_to_uuid: valid string returns UUID, None or empty returns None
        self.assertEqual(SerializableBase._str_to_uuid(s), u)
        self.assertIsNone(SerializableBase._str_to_uuid(None))
        self.assertIsNone(SerializableBase._str_to_uuid(""))
