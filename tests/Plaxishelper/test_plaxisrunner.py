import builtins
import unittest
from io import StringIO
from types import SimpleNamespace
from plaxisproxy_excavation.plaxishelper.plaxisrunner import PlaxisRunner

class DummyServer:
    def __init__(self, connection, proxy_factory, tokenizer):
        DummyServer.init_called = True
    def get_global_input(self):
        # Return a dummy global input object with minimal interface
        return SimpleNamespace(new=lambda: setattr(self, "new_called", True),
                                Phases=[SimpleNamespace(Name="InitialPhase", ShouldCalculate=True)],
                                gotomesh=lambda: None,
                                generatemesh=lambda: None,
                                gotostages=lambda: None,
                                calculate=lambda: None,
                                save=lambda path=None: setattr(self, "save_path", path) if path else setattr(self, "saved", True))

class DummyConnection:
    def __init__(self, host, port, password=None):
        DummyConnection.called = True

class DummyProxyFactory:
    pass

class DummyTokenizer:
    pass

class TestPlaxisRunner(unittest.TestCase):
    def setUp(self):
        # Monkeypatch the third_party imports in PlaxisRunner to use dummy classes
        PlaxisRunner.Server = DummyServer
        PlaxisRunner.HttpConnection = DummyConnection
        PlaxisRunner.PlxProxyFactory = DummyProxyFactory
        PlaxisRunner.PlxTokenizer = DummyTokenizer

    def test_connect_success_and_new_project(self):
        """connect() returns True on successful connections and optionally starts new project."""
        runner = PlaxisRunner(input_port=10000, output_port=10001, password="secret")
        success = runner.connect(start_new_project=True)
        self.assertTrue(success)
        # DummyServer should have been initialized for both input and output
        self.assertTrue(hasattr(DummyServer, "init_called") and DummyServer.init_called)
        # After connect, global objects g_i and g_o should be set
        self.assertIsNotNone(runner.g_i)
        self.assertIsNotNone(runner.g_o)
        # Starting new project should have called g_i.new()
        self.assertTrue(hasattr(runner.g_i, "new_called") and runner.g_i.new_called)

    def test_connect_failure(self):
        """connect() handles ConnectionRefusedError and returns False."""
        # Patch DummyConnection to raise ConnectionRefusedError
        def failing_conn(host, port, password=None):
            raise ConnectionRefusedError()
        PlaxisRunner.HttpConnection = failing_conn
        runner = PlaxisRunner(1234, 5678, "pwd")
        success = runner.connect()
        self.assertFalse(success)

    def test_build_model_calls_mapper(self):
        """build_model uses PlaxisMapper to run model creation when connected."""
        runner = PlaxisRunner(10000, 10001, "pwd")
        # Simulate connected state by providing a dummy g_i
        runner.g_i = SimpleNamespace()
        # Monkeypatch PlaxisMapper within PlaxisRunner to track invocation
        class DummyMapper:
            def __init__(self, gi):
                DummyMapper.initialized_with = gi
            def run(self, model):
                DummyMapper.run_called_with = model
        PlaxisRunner.PlaxisMapper = DummyMapper
        model = SimpleNamespace(dummy="model")
        runner.build_model(model)
        # Ensure DummyMapper was initialized with g_i and run called with model
        self.assertIs(DummyMapper.initialized_with, runner.g_i)
        self.assertIs(DummyMapper.run_called_with, model)

    def test_generate_mesh(self):
        """generate_mesh calls gotomesh() and generatemesh() on g_i, handling exceptions."""
        runner = PlaxisRunner(0, 0, "")
        # Not connected case: prints warning and returns
        runner.g_i = None
        self.assertIsNone(runner.generate_mesh())  # should simply return without error
        # Connected case
        runner.g_i = SimpleNamespace(gotomesh=lambda: None, generatemesh=lambda: None)
        # Should run without exception
        self.assertIsNone(runner.generate_mesh())
        # If generatemesh raises, it should catch and not propagate
        runner.g_i = SimpleNamespace(gotomesh=lambda: None, generatemesh=lambda: (_ for _ in ()).throw(Exception("mesh fail")))
        # No exception should escape
        try:
            runner.generate_mesh()
        except Exception as e:
            self.fail(f"generate_mesh should not propagate exceptions, but got {e}")

    def test_calculate_phase_selection(self):
        """calculate() switches to staged construction and respects phases_to_calculate list."""
        # Prepare dummy g_i with Phases list and calculate method
        phase1 = SimpleNamespace(Name="Phase1", ShouldCalculate=True)
        phase2 = SimpleNamespace(Name="Phase2", ShouldCalculate=True)
        phases_container = type("PhasesCont", (), {})()
        # Make Phases container iterable and attribute-accessible by phase name
        phases_container._phases = [phase1, phase2]
        phases_container.__iter__ = lambda self: iter(self._phases)
        phases_container.__getattr__ = lambda self, name: next((p for p in self._phases if getattr(p, "Name", "") == name), None)
        dummy_g_i = SimpleNamespace(Phases=phases_container, gotostages=lambda: None, calculate=lambda: None)
        runner = PlaxisRunner(0, 0, "")
        runner.g_i = dummy_g_i
        # No specific phases: all ShouldCalculate remain True
        runner.calculate()
        self.assertTrue(all(p.ShouldCalculate for p in phases_container._phases))
        # Specific phase selection
        phase1.ShouldCalculate = True
        phase2.ShouldCalculate = True
        runner.calculate(phases_to_calculate=["Phase1"])
        # Phase1 should be True, Phase2 False
        self.assertTrue(phase1.ShouldCalculate)
        self.assertFalse(phase2.ShouldCalculate)
        # Ensure calculate() method on g_i was called
        # (We verify indirectly by no exception; for thoroughness, check an attribute if set)
        # If any warnings are printed for missing phases, ensure no exception:
        try:
            runner.calculate(phases_to_calculate=["Nonexistent"])
        except Exception as e:
            self.fail(f"calculate should handle missing phase names without error, but got {e}")

    def test_save_functionality(self):
        """save() calls g_i.save with correct arguments and handles not connected or exceptions."""
        runner = PlaxisRunner(0, 0, "")
        # Not connected
        runner.g_i = None
        self.assertIsNone(runner.save("file.plx"))  # should simply return
        # Connected - test with file path
        saved_paths = {}
        runner.g_i = SimpleNamespace(save=lambda path=None: saved_paths.setdefault("path", path))
        runner.save("proj.plx")
        self.assertEqual(saved_paths.get("path"), "proj.plx")
        # Connected - test without file path
        saved_paths.clear()
        runner.g_i = SimpleNamespace(save=lambda path=None: saved_paths.setdefault("path", path) or saved_paths.setdefault("called", True))
        runner.save()
        # Without file_path, path arg should be None and still called
        self.assertTrue(saved_paths.get("called", False))
        # If save raises, ensure it is caught (not propagated)
        runner.g_i = SimpleNamespace(save=lambda path=None: (_ for _ in ()).throw(Exception("save error")))
        try:
            runner.save("bad.plx")
        except Exception as e:
            self.fail(f"save should catch exceptions, but propagated {e}")

    def test_get_results_and_close(self):
        """get_results returns dummy results and close() resets connections."""
        runner = PlaxisRunner(0, 0, "")
        # Not connected to output
        runner.g_o = None
        results = runner.get_results(["P1"])
        self.assertEqual(results, {})
        # Connected to output: prepare dummy Phases and global output
        last_phase = SimpleNamespace(Name="FinalPhase")
        runner.g_o = SimpleNamespace(Phases=[1, last_phase])  # so Phases[-1] is last_phase
        # Should return placeholder results for each monitor label
        res = runner.get_results(["MonA", "MonB"])
        self.assertEqual(res.get("MonA"), "Some result value")
        self.assertEqual(res.get("MonB"), "Some result value")
        # No exceptions on retrieval, missing labels handled gracefully (just result None if exception, but here none thrown)
        # Test close() sets servers and globals to None
        runner.input_server = object()
        runner.output_server = object()
        runner.g_i = object()
        runner.g_o = object()
        runner.close()
        self.assertIsNone(runner.input_server)
        self.assertIsNone(runner.output_server)
        self.assertIsNone(runner.g_i)
        self.assertIsNone(runner.g_o)
