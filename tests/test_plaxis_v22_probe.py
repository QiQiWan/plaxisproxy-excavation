"""License-free tests for the V22 API probe."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace


probe_file = Path(__file__).resolve().parents[1] / "scripts" / "probe_plaxis_v22.py"
spec = importlib.util.spec_from_file_location("probe_plaxis_v22", str(probe_file))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_read_only_phase_inspection_and_proxy_fields(self):
        phases = [SimpleNamespace(
            Identification=SimpleNamespace(value="P1"),
            CalculationResult=SimpleNamespace(value=1),
            LogInfo=0,
            ShouldCalculate=True,
        )]
        response = probe.inspect_input(SimpleNamespace(Phases=phases))
        self.assertEqual(response["inspected_phase_count"], 1)
        self.assertEqual(response["phases"][0]["calculation_result"], 1)
        self.assertEqual(response["phases"][0]["identification"], "P1")

    def test_missing_field_reports_name_not_secret(self):
        phase = SimpleNamespace(Identification="Phase_1")
        data = probe.inspect_input(SimpleNamespace(Phases=[phase]))
        self.assertEqual(data["phases"][0]["calculation_result"],
                         {"read_error": "AttributeError"})

    def test_connection_is_closed_and_password_is_not_reported(self):
        fake_server = SimpleNamespace(closed=False)
        fake_server.close = lambda: setattr(fake_server, "closed", True)
        def mock_new_server(host, port, password):
            self.assertEqual(password, "secret-value")
            return fake_server, SimpleNamespace(Phases=[])
        report = probe.run_probe("127.0.0.1", 10000,
                                 "secret-value", mock_new_server)
        self.assertTrue(fake_server.closed)
        self.assertNotIn("secret-value", str(report))

    def test_missing_password_fails_before_connection(self):
        with self.assertRaises(ValueError):
            probe.run_probe("127.0.0.1", 10000, "", lambda *a, **k: None)

    def test_no_model_exists_still_has_valid_report(self):
        data = probe.inspect_input(SimpleNamespace())
        self.assertEqual(data["phase_read_error"], "AttributeError")
        self.assertEqual(data["inspected_phase_count"], 0)


if __name__ == "__main__":
    unittest.main()
