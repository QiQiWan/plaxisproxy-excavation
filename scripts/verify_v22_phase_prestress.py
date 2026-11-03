"""Windows PLAXIS V22 scratch-model probe. NEVER calculate, open, save or create models.

Uses the installed vendor plxscripting, but loads the two pure gateway files directly
so unrelated package initialization/optional dependencies cannot block this probe.
See docs/phase-prestress-v22-validation.md for setup and limits.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_gateway(name):
    path = ROOT / 'src/plaxisproxy_excavation/plaxishelper' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name + '_standalone', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROPS = load_gateway('phase_properties')
RESULTS = load_gateway('phase_results')


def snapshot(support, phase):
    return {'AdjustPrestress': PROPS._normalize(support.AdjustPrestress[phase], True),
            'PrestressForce': PROPS._normalize(support.PrestressForce[phase], 0.0)}


def error_evidence(exc):
    return {'type': type(exc).__name__, 'message': str(exc),
            'rollback_verified': getattr(exc, 'rollback_verified', None),
            'recovery_required': getattr(exc, 'recovery_required', True),
            'before': getattr(exc, 'before', {}),
            'observed': getattr(exc, 'observed', {}),
            'rollback_errors': getattr(exc, 'rollback_errors', ())}


def probe_input(g_i, support, phase, guards, force, failure, report, checkpoint):
    """Always attempt target cleanup, including KeyboardInterrupt/SystemExit."""
    old = snapshot(support, phase)
    guard_before = [snapshot(support, item) for item in guards]
    report.update(before=old, guard_before=guard_before, cleanup_verified=False,
                  input_contract_passed=False)
    checkpoint()  # Persist scalar snapshots before the first mutation.

    class InjectedSecondWriteFailure(RuntimeError):
        pass

    class InstrumentedInput:
        def __init__(self):
            self.calls = 0
            self.injected_error = None

        def set(self, prop, selected_phase, value):
            self.calls += 1
            second_forward = (self.calls == 2 and selected_phase is phase and
                              type(value) in (int, float) and value == force)
            if second_forward and failure == 'before':
                self.injected_error = InjectedSecondWriteFailure('Before second forward apply')
                raise self.injected_error
            result = g_i.set(prop, selected_phase, value)
            if second_forward and failure == 'after':
                self.injected_error = InjectedSecondWriteFailure('After second forward apply')
                raise self.injected_error
            return result

    api = InstrumentedInput()
    try:
        # Fault injection needs two actual writes (not a pre-existing partial no-op).
        if failure != 'none' and (old['AdjustPrestress'] or
                                  math.isclose(old['PrestressForce'], force,
                                               rel_tol=1e-8, abs_tol=1e-6)):
            raise RuntimeError('Fault probe requires AdjustPrestress=False and a different force')
        try:
            report['command'] = PROPS.set_node_to_node_prestress(
                api, support, phase, force, compression_sign_verified=True,
                guard_phases=guards)
        except PROPS.PhasePropertyError as exc:
            expected = (api.injected_error is not None and exc.__cause__ is api.injected_error)
            report['expected_failure' if expected else 'write_error'] = error_evidence(exc)
            if not expected or not exc.rollback_verified:
                raise
        else:
            if failure != 'none':
                raise RuntimeError('Injected failure was not exercised')
            count = api.calls
            PROPS.set_node_to_node_prestress(api, support, phase, force,
                                             compression_sign_verified=True, guard_phases=guards)
            report['repeat_was_noop'] = api.calls == count
            if not report['repeat_was_noop']:
                raise RuntimeError('Repeated update unexpectedly performed writes')
        report['after_probe'] = snapshot(support, phase)
        report['input_contract_passed'] = True
    finally:
        try:
            PROPS.set_phase_properties(g_i, support, phase, old, guard_phases=guards)
            report['after_cleanup'] = snapshot(support, phase)
            report['guard_after'] = [snapshot(support, item) for item in guards]
            # set_phase_properties verifies target snapshot restoration numerically.
            report['cleanup_verified'] = report['guard_after'] == guard_before
            if not report['cleanup_verified']:
                raise RuntimeError('Guard phase changed; do not save or calculate')
        except Exception as exc:
            report['cleanup_error'] = error_evidence(exc)
            report['input_contract_passed'] = False
            raise
        finally:
            checkpoint()


def named_handle(api, name):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        raise ValueError('Use an exact PLAXIS object name, not an expression')
    return getattr(api, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('input', 'output'), required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--support', required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--guard-phase', action='append', default=[])
    parser.add_argument('--force', type=float)
    parser.add_argument('--force-unit', choices=('kN',), required=True)
    parser.add_argument('--build', required=True, help='Manually record exact About-dialog build')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--write-test', action='store_true')
    parser.add_argument('--confirm-scratch-model', action='store_true')
    parser.add_argument('--compression-sign-verified', action='store_true')
    parser.add_argument('--confirm-converged', action='store_true')
    parser.add_argument('--failure', choices=('none', 'before', 'after'), default='none')
    parser.add_argument('--result-type', default='NodeToNodeAnchor.AnchorForce3D')
    parser.add_argument('--location', choices=('node', 'stresspoint'), default='node')
    args = parser.parse_args(argv)
    if args.mode == 'input' and args.write_test:
        if not (args.confirm_scratch_model and args.compression_sign_verified and
                args.guard_phase and args.force is not None and
                math.isfinite(args.force) and args.force != 0):
            parser.error('Writes require scratch/sign confirmation, guard phase and finite nonzero force')
        if args.phase in args.guard_phase:
            parser.error('Guard and target phase names must differ')
    if args.mode == 'output' and (args.write_test or not args.confirm_converged):
        parser.error('Output is read-only and requires manually confirmed converged phase')
    password = os.environ.get('PLAXIS_PASSWORD')
    if not password:
        parser.error('Set PLAXIS_PASSWORD in this process; do not put passwords on command lines')

    report = {'kind': 'real-session-probe', 'status': 'started',
              'build_user_reported': args.build, 'mode': args.mode,
              'support_name': args.support, 'phase_name': args.phase,
              'guard_phase_names': args.guard_phase,
              'force_unit_user_confirmed': args.force_unit,
              'compression_sign_user_confirmed': args.compression_sign_verified,
              'convergence_user_confirmed': args.confirm_converged,
              'failure_injection': args.failure,
              'python': sys.version, 'solver_started': False,
              'mechanical_validation': 'NOT_PERFORMED_BY_THIS_SCRIPT'}

    def checkpoint():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    checkpoint()  # Fail on an unwritable report before connecting/mutating.
    try:
        from plxscripting.easy import new_server
        server, api = new_server('localhost', args.port, password=password)
        support = named_handle(api, args.support)
        phase = named_handle(api, args.phase)
        if args.mode == 'input':
            if not args.write_test:
                report['snapshot'] = snapshot(support, phase)
                report['status'] = 'read_only_snapshot'
            else:
                guards = tuple(named_handle(api, name) for name in args.guard_phase)
                probe_input(api, support, phase, guards, args.force, args.failure, report, checkpoint)
                report['status'] = 'input_contract_passed_and_restored'
        else:
            result_type = api.ResultTypes
            for segment in args.result_type.split('.'):
                result_type = named_handle(result_type, segment)
            report['result_type'] = args.result_type
            report['output'] = RESULTS.read_node_to_node_axial_force(
                api, support, phase, result_type=result_type,
                force_unit=args.force_unit, location=args.location)
            report['status'] = 'output_samples_read_requires_mechanical_validation'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = error_evidence(exc)
        raise
    finally:
        checkpoint()
    print(json.dumps({'status': report['status'], 'report': str(args.report)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
