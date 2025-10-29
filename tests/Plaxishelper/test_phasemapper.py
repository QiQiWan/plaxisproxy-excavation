import unittest
from types import SimpleNamespace
from plaxisproxy_excavation.plaxishelper.phasemapper import PhaseMapper

class TestPhaseMapper(unittest.TestCase):
    def test_create_phases_empty(self):
        """PhaseMapper.create_phases returns immediately if phases list is empty."""
        dummy_g_i = SimpleNamespace(Phases=[SimpleNamespace()])  # initial phase exists
        # Should not throw or do anything
        PhaseMapper.create_phases(dummy_g_i, [])
        # No new attributes should be added to dummy_g_i
        self.assertFalse(hasattr(dummy_g_i, "phase_created"))

    def test_create_phases_full(self):
        """PhaseMapper.create_phases creates phases and sets activities and excavations."""
        # Dummy g_i with Phases[0] and phase method
        initial_phase = SimpleNamespace(Name="InitialPhase")
        created_phases = []
        def dummy_phase(prev_phase):
            # Create and return a new dummy phase
            new_phase = SimpleNamespace(Name=None)
            created_phases.append(new_phase)
            return new_phase
        dummy_g_i = SimpleNamespace(Phases=[initial_phase], phase=lambda prev: dummy_phase(prev), 
                                    activate=lambda obj, ph: setattr(obj, "activated_in", ph),
                                    deactivate=lambda obj, ph: setattr(obj, "deactivated_in", ph))
        # Prepare a ConstructionStage-like object
        cs = SimpleNamespace(name="Phase1", settings=SimpleNamespace(settings={"foo": 1.23}), 
                             _activate=[SimpleNamespace(name="ActObj", plx_id=None)], 
                             _deactivate=[SimpleNamespace(name="DeactObj", plx_id=None)], 
                             _excavate=[SimpleNamespace(name="SoilX", plx_id=SimpleNamespace(Active=True))],
                             plx_id=None)
        PhaseMapper.create_phases(dummy_g_i, [cs])
        # A new phase should have been created and assigned a Name
        self.assertEqual(len(created_phases), 1)
        new_phase = created_phases[0]
        self.assertEqual(new_phase.Name, "Phase1")
        # The settings dict 'foo' should be applied if attribute exists (if not present on dummy, it's skipped harmlessly)
        # Activation/deactivation: since plx_id of ActObj and DeactObj were None and dummy_g_i has attributes by name?
        # Our dummy objects had name but no corresponding attr on g_i, so they should trigger warnings (no exception).
        # Excavation: SoilX has plx_id set, g_i.deactivate should have been called
        soil_obj = cs._excavate[0].plx_id
        # SoilX plx_id should have been deactivated either via g_i.deactivate or fallback Active False
        self.assertFalse(hasattr(soil_obj, "Active") and soil_obj.Active)
        # plx_id of cs should be set to new phase
        self.assertIs(cs.plx_id, new_phase)
