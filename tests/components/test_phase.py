import unittest
from types import SimpleNamespace
from plaxisproxy_excavation.components.phase import StageSettings, ConstructionStage, CalcType, LoadType, PhaseType

class TestPhaseComponents(unittest.TestCase):
    def test_stagesettings_post_init_and_factory(self):
        """StageSettings adds calc_type to settings and factory methods produce correct defaults."""
        # __post_init__ should insert calc_type if missing
        settings = StageSettings(calc_type=CalcType.Plastic, settings={})
        self.assertIn("calc_type", settings.settings)
        self.assertEqual(settings.settings["calc_type"], CalcType.Plastic.value)
        # Factory method for Plastic settings
        s = StageSettings.create_plastic_settings(p_stop=2.5, load_type=LoadType.StageConstruction.value)
        self.assertIsInstance(s, StageSettings)
        # Ensure calc_type and p_stop are set correctly
        self.assertEqual(s.settings["calc_type"], CalcType.Plastic.value)
        self.assertEqual(s.settings["p_stop"], 2.5)
        # Defaults present
        self.assertIn("Arc_length_control", s.settings)
        self.assertTrue(s.settings["Arc_length_control"])

    def test_constructionstage_activation_and_properties(self):
        """ConstructionStage activation/deactivation and property behaviors."""
        stage = ConstructionStage("Stage1", comment="Test stage")
        # Default phase_type is OTHER and duration 0.0
        self.assertEqual(stage.phase_type, PhaseType.OTHER)
        self.assertEqual(stage.duration, 0.0)
        # Water table property
        wt = SimpleNamespace(label="WT1")
        stage.water_table = wt
        self.assertIs(stage.water_table, wt)
        # Notes property (set via constructor optional)
        self.assertIsNone(stage.notes)
        stage2 = ConstructionStage("Stage2", comment="", notes="Note A")
        self.assertEqual(stage2.notes, "Note A")
        # Activate and deactivate
        objA = SimpleNamespace(name="StructA")
        objB = "StructB"
        stage.activate(objA)
        stage.deactivate(objB)
        self.assertIn(objA, stage._activate)
        self.assertIn("StructB", stage._deactivate)
        # Excavation and other soil actions
        soilX = SimpleNamespace(name="SoilX")
        stage.excavate_block(soilX)
        stage.freeze_block(soilX)
        stage.thaw_block(soilX)
        stage.backfill_block(soilX)
        self.assertIn(soilX, stage._excavate)
        self.assertIn(soilX, stage._freeze)
        self.assertIn(soilX, stage._thaw)
        self.assertIn(soilX, stage._backfill)
        # Summary output reflects these lists (names)
        summ = stage.summary()
        self.assertEqual(summ["name"], "Stage1")
        self.assertIn("StructA", summ["activate"])
        self.assertIn("StructB", summ["deactivate"])
        self.assertIn("SoilX", summ["excavate"])
        self.assertIn("SoilX", summ["freeze"])
        self.assertIn("SoilX", summ["thaw"])
        self.assertIn("SoilX", summ["backfill"])
        self.assertEqual(summ["water_table"], "WT1")
        self.assertIsNone(summ["notes"])
        # to_dict output contains similar data and metadata
        d = stage.to_dict()
        self.assertEqual(d["name"], "Stage1")
        self.assertEqual(d["comment"], "Test stage")
        self.assertEqual(d["phase_type"], PhaseType.OTHER.value)
        self.assertIn("activate", d)
        self.assertIn("excavate", d)
        self.assertIn("plx_id", d)
        # Activate list in to_dict should use name or string if available
        self.assertIn("StructA", d["activate"])
        self.assertIn("StructB", d["deactivate"])
        self.assertIn("SoilX", d["excavate"])
        self.assertEqual(d["water_table"], "WT1")
        # from_dict should reconstruct basic fields correctly
        stage_dict = stage.to_dict()
        stage_copy = ConstructionStage.from_dict(stage_dict)
        self.assertEqual(stage_copy.name, "Stage1")
        self.assertEqual(stage_copy.comment, "Test stage")
        self.assertEqual(stage_copy.phase_type, PhaseType.OTHER)
        self.assertEqual(stage_copy.duration, 0.0)
        # Activation/deactivation lists may contain placeholders or names; ensure expected items present
        self.assertTrue(any(getattr(x, "name", str(x)) == "StructA" for x in stage_copy._activate))
        self.assertTrue(any(getattr(x, "name", str(x)) == "StructB" for x in stage_copy._deactivate))
