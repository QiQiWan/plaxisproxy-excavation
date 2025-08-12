
import uuid
import unittest
from plaxisproxy_excavation.components.curvepoint import CurvePoint, NodePoint, StressPoint

class TestCurvePoints(unittest.TestCase):
    def test_curvepoint_id_and_plxid(self):
        """CurvePoint inherits Point and has unique id and plx_id property."""
        cp = CurvePoint(1.0, 2.0, 3.0, label="CP1")
        # id should be a UUID
        self.assertIsInstance(cp.id, uuid.UUID)
        # plx_id initially None, setting updates property
        self.assertIsNone(cp.plx_id)
        dummy_plx = "DummyPlaxisID"
        cp.plx_id = dummy_plx
        self.assertEqual(cp.plx_id, dummy_plx)
        # label property
        self.assertEqual(cp.label, "CP1")
        # datafrom and pre_calc default
        self.assertIsNone(cp.datafrom)
        self.assertFalse(cp.pre_calc)
        # __repr__ includes label and coordinates
        rep = repr(cp)
        self.assertIn("CurvePoint", rep)
        self.assertIn("CP1", rep)
        self.assertIn("(1.000, 2.000, 3.000)", rep)

    def test_nodepoint_and_stresspoint_specific(self):
        """NodePoint and StressPoint store their specific identifiers and repr includes them."""
        np = NodePoint(0.0, 0.0, 0.0, label="NodeA", node_id=123)
        self.assertIsInstance(np, CurvePoint)
        self.assertEqual(np.node_id, 123)
        # __repr__ should include node_id and label
        rep_np = repr(np)
        self.assertIn("NodePoint", rep_np)
        self.assertIn("node_id=123", rep_np)
        self.assertIn("NodeA", rep_np)
        sp = StressPoint(1.0, 1.0, 1.0, label=None, element_id=7, local_index=2)
        self.assertIsInstance(sp, CurvePoint)
        self.assertEqual(sp.element_id, 7)
        self.assertEqual(sp.local_index, 2)
        # __repr__ should include element_id and local_index
        rep_sp = repr(sp)
        self.assertIn("StressPoint", rep_sp)
        self.assertIn("elem_id=7", rep_sp)
        self.assertIn("idx=2", rep_sp)
        # If label is None, label_info should not appear
        self.assertNotIn("''", rep_sp)  # no empty label string in repr
