"""Phase-only writes with verified, best-effort compensation (not server transactions).

Callers must own the Input session exclusively: the lock only serializes this module
inside one Python process. Never calculate/save or continue a batch after an error
with recovery_required=True. No global-property fallback is permitted.
"""
from __future__ import annotations

import math
from threading import RLock

_LOCK = RLock()


class PhasePropertyError(RuntimeError):
    """Failure with explicit recovery certainty and scalar diagnostic evidence."""

    def __init__(self, message, *, stage='preflight', before=None, observed=None,
                 rollback_verified=None, rollback_errors=(), recovery_required=False):
        super().__init__(message)
        self.stage = stage
        self.before = before or {}
        self.observed = observed or {}
        self.rollback_verified = rollback_verified
        self.rollback_errors = tuple(rollback_errors)
        self.recovery_required = recovery_required


def _scalar(value):
    for _ in range(4):
        if type(value) in (bool, int, float, str):
            return value
        other = getattr(value, 'value', value)
        if other is value:
            break
        value = other
    raise ValueError('Expected a resolved scalar, not an unresolved proxy')


def _normalize(value, template):
    value = _scalar(value)
    if type(template) is bool:
        if type(value) is bool:
            return value
        if type(value) is int and value in (0, 1):
            return bool(value)
        if type(value) is str:
            token = value.strip().lower()
            if token in ('true', 'yes', '1'):
                return True
            if token in ('false', 'no', '0'):
                return False
        raise ValueError('Unrecognized boolean phase state')
    if type(template) in (int, float):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Expected finite numeric phase state')
        return value
    if type(template) is str and type(value) is str:
        return value
    raise ValueError('Unsupported scalar phase state')


def _matches(actual, expected, abs_tol):
    if type(expected) in (int, float):
        return math.isclose(actual, expected, rel_tol=1e-8, abs_tol=abs_tol)
    return actual == expected


def _validate_value(value):
    if type(value) not in (bool, int, float, str):
        raise ValueError('Only scalar bool/numeric/text values are supported')
    if type(value) in (int, float) and not math.isfinite(value):
        raise ValueError('Numeric phase value must be finite')


def _apply_properties(g_i, object_handle, phase_handle, updates, guard_phases, abs_tol):
    """Snapshot all targets before writes; restore the full target snapshot on failure."""
    with _LOCK:
        before, observed, records, attempted = {}, {}, [], []
        try:
            if phase_handle is None or object_handle is None:
                raise ValueError('Object and explicit phase handles are required')
            if type(abs_tol) not in (int, float) or not math.isfinite(abs_tol) or abs_tol < 0:
                raise ValueError('abs_tol must be finite and nonnegative')
            guards = tuple(guard_phases)
            if any(guard is None for guard in guards):
                raise ValueError('Guard phase cannot be None')
            props = []
            for name, value in updates:
                _validate_value(value)
                prop = getattr(object_handle, name)
                if prop is None:
                    raise ValueError('Missing phase property: ' + name)
                props.append((name, prop, value))
            for index, phase in enumerate((phase_handle,) + guards):
                for name, prop, value in props:
                    key = ('target' if index == 0 else 'guard[%d]' % (index - 1)) + '.' + name
                    old = _normalize(prop[phase], value)
                    before[key] = old
                    records.append((key, prop, phase, old, value, index == 0))
        except Exception as exc:
            raise PhasePropertyError('Phase snapshot/validation failed; no writes attempted',
                                     before=before) from exc

        recovery_interruptions = []

        def verify(restoring=False):
            errors = []
            for key, prop, phase, old, value, target in records:
                try:
                    actual = _normalize(prop[phase], value)
                    observed[key] = actual
                    expected = old if restoring or not target else value
                    if not _matches(actual, expected, abs_tol):
                        errors.append(key + ': readback mismatch')
                except BaseException as exc:
                    if not isinstance(exc, Exception):
                        if not restoring:
                            raise
                        recovery_interruptions.append(exc)
                    observed[key] = '<unreadable>'
                    errors.append(key + ': ' + type(exc).__name__)
            return errors

        try:
            for key, prop, phase, old, value, target in records:
                if not target or _matches(old, value, abs_tol):
                    continue
                # Record BEFORE the call: remote side may apply and then lose its response.
                attempted.append((key, prop, phase, old))
                g_i.set(prop, phase, value)
                actual = _normalize(prop[phase], value)
                observed[key] = actual
                if not _matches(actual, value, abs_tol):
                    raise ValueError(key + ': write readback mismatch')
            errors = verify()
            if errors:
                raise ValueError('; '.join(errors))
        except BaseException as exc:
            restore_errors = []
            # A setter can change another property before failing. Once any write
            # was attempted, compensate the entire target snapshot, never guards.
            restore_records = [record for record in records if record[-1]] if attempted else []
            for key, prop, phase, old, value, target in reversed(restore_records):
                try:
                    g_i.set(prop, phase, old)
                except BaseException as restore_exc:
                    if not isinstance(restore_exc, Exception):
                        recovery_interruptions.append(restore_exc)
                    # Continue: the remaining independent restore can still succeed.
                    restore_errors.append(key + ': ' + type(restore_exc).__name__)
            final_errors = verify(restoring=True)
            restored = not final_errors
            failure = PhasePropertyError(
                'Phase update failed; ' + ('snapshot restored and verified' if restored else
                                           'RECOVERY REQUIRED: phase state uncertain'),
                stage='write/readback', before=before, observed=observed.copy(),
                rollback_verified=restored, rollback_errors=restore_errors + final_errors,
                recovery_required=not restored)
            if not isinstance(exc, Exception):
                exc.phase_property_error = failure
                raise
            if recovery_interruptions:
                interrupt = recovery_interruptions[0]
                interrupt.phase_property_error = failure
                raise interrupt from exc
            raise failure from exc
        return {name: observed['target.' + name] for name, _ in updates}


def set_phase_properties(g_i, object_handle, phase_handle, values, *,
                         abs_tol=1e-6, guard_phases=()):
    """Compensated ordered scalar writes, useful for restoring a saved pair.

    `values` must be a nonempty mapping; insertion order is write order.
    This does not provide a native server-side transaction or inheritance rollback.
    """
    try:
        updates = tuple(values.items())
        if not updates or any(type(name) is not str or not name for name, _ in updates):
            raise ValueError('A nonempty property mapping is required')
    except Exception as exc:
        raise PhasePropertyError('Invalid phase property mapping') from exc
    return _apply_properties(g_i, object_handle, phase_handle, updates, guard_phases, abs_tol)


def set_phase_property(g_i, object_handle, phase_handle, property_name,
                       value, *, abs_tol=1e-6, guard_phases=()):
    """Set and verify one explicit phase property, compensating on failure.

    A readable scalar old value is now mandatory. Optional guard phases are checked
    before/after but never written. There is no fallback to global setattr.
    """
    result = _apply_properties(g_i, object_handle, phase_handle,
                               ((property_name, value),), guard_phases, abs_tol)
    return result[property_name]


def set_node_to_node_prestress(g_i, support, phase, signed_force_kn,
                               *, compression_sign_verified=False, guard_phases=()):
    """Set command force per anchor in a confirmed kN model; never actual force.

    The caller must verify the compression sign in the real build, model units,
    support activation and phase eligibility. Both scalar states are snapshotted
    before either write. Guard phases detect unintended changes, including possible
    inheritance effects; they are never automatically overwritten during recovery.
    """
    if compression_sign_verified is not True:
        raise PhasePropertyError('Compression sign convention has not been verified')
    if (type(signed_force_kn) not in (int, float) or
            not math.isfinite(signed_force_kn) or signed_force_kn == 0):
        raise PhasePropertyError('Prestress must be a nonzero finite signed force')
    _apply_properties(g_i, support, phase,
                      (('AdjustPrestress', True), ('PrestressForce', signed_force_kn)),
                      guard_phases, 1e-6)
    return {'command_prestress_kn': float(signed_force_kn),
            'actual_axial_force_kn': None,
            'requires_output_check': True}
