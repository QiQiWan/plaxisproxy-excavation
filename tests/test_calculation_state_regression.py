"""Regression tests for verified PLAXIS calculation states.

These tests use fakes and do not require a PLAXIS licence.
"""
from types import SimpleNamespace
import pytest

from plaxisproxy_excavation.builder.excavation_builder import ExcavationBuilder
from plaxisproxy_excavation.plaxishelper.plaxisrunner import PlaxisRunner


def _phase(name, state):
    return SimpleNamespace(
        name=name,
        plx_id=SimpleNamespace(CalculationResult=state),
    )


@pytest.mark.parametrize("state,expected", [
    (0, False),
    (1, True),
    (2, False),
    (SimpleNamespace(value=1), True),
    (SimpleNamespace(value="1"), True),
    (None, False),
    ("unexpected", False),
])
def test_phase_calculation_state_is_strict(state, expected):
    runner = PlaxisRunner("unused", "localhost", 10000)
    assert runner.is_calculated(_phase("P1", state)) is expected


def _builder(states, answer="OK"):
    builder = ExcavationBuilder.__new__(ExcavationBuilder)
    phases = [_phase(f"P{i}", state) for i, state in enumerate(states)]
    builder.excavation_object = SimpleNamespace(phases=phases)
    def run():
        if answer != "OK":
            raise RuntimeError(answer)
        return answer
    builder.App = SimpleNamespace(
        calculate=run,
        is_calculated=lambda ph: PlaxisRunner.is_calculated(
            PlaxisRunner("unused", "localhost", 10000), ph
        ),
        mark_phase_should_calculate=lambda ph, should_cal: None,
    )
    builder._calc_done = False
    return builder


def test_builder_requires_all_phases_to_converge():
    builder = _builder([1, 1, 1])
    builder.calculate()
    assert builder._calc_done is True


@pytest.mark.parametrize("states", [[0, 1], [1, 2], []])
def test_builder_rejects_incomplete_model(states):
    builder = _builder(states)
    with pytest.raises(RuntimeError):
        builder.calculate()
    assert builder._calc_done is False


def test_builder_rejects_calculation_failure():
    builder = _builder([1], answer="failure")
    with pytest.raises(RuntimeError, match="failure"):
        builder.calculate()
    assert builder._calc_done is False

def test_builder_skips_phase_explicitly_excluded_from_calculation():
    builder = _builder([1, 2])
    builder.excavation_object.phases[1].plx_id.ShouldCalculate = False
    builder.calculate()
    assert builder._calc_done is True


def test_builder_handles_proxy_boolean_for_phase_selection():
    builder = _builder([1, 2])
    builder.excavation_object.phases[1].plx_id.ShouldCalculate = (
        SimpleNamespace(value="False")
    )
    builder.calculate()
    assert builder._calc_done is True


def test_builder_rejects_all_phases_disabled():
    builder = _builder([1])
    builder.excavation_object.phases[0].plx_id.ShouldCalculate = False
    with pytest.raises(RuntimeError, match="No phases are selected"):
        builder.calculate()
    assert builder._calc_done is False


def test_phase_lookup_does_not_trust_builder_cache():
    builder = _builder([2])
    builder._calc_done = True
    assert builder._ensure_phase_is_calculated(builder.excavation_object.phases[0]) is False


def test_unknown_phase_is_rejected():
    builder = _builder([1])
    with pytest.raises(ValueError, match="does not exist"):
        builder.mark_phase_calculate(_phase("OTHER", 1))


def test_calculate_command_rejects_non_ok_response():
    runner = PlaxisRunner("unused", "localhost", 10000)
    runner.g_i = SimpleNamespace(calculate=lambda: "Some phases failed")
    runner.goto_stages = lambda: None
    with pytest.raises(RuntimeError, match="Some phases failed"):
        runner.calculate()
