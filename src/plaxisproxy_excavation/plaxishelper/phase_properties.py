"""Fail-closed phase-aware PLAXIS property updates.

Uses native PLAXIS phase handles and the documented
NodeToNodeAnchor AdjustPrestress/PrestressForce attributes.
"""
from __future__ import annotations
import math

class PhasePropertyError(RuntimeError):
    """A phase-specific PLAXIS property could not be applied and verified."""

def _scalar(value):
    for _ in range(2):
        other = getattr(value, "value", value)
        if other is value:
            return value
        value = other
    return value

def set_phase_property(g_i, object_handle, phase_handle, property_name,
                       value, *, abs_tol=1e-6):
    """Use g_i.set(prop, phase, value), then read back THIS phase.

    Deliberately has no global setattr fallback, because such a fallback
    may modify other phases of the model without warning.
    """
    prop = getattr(object_handle, property_name, None)
    if prop is None:
        raise PhasePropertyError("Missing PLAXIS property: " + property_name)
    try:
        g_i.set(prop, phase_handle, value)
        actual = _scalar(prop[phase_handle])
    except Exception as exc:
        raise PhasePropertyError(
            "Phase-specific property write/read failed: " + property_name
        ) from exc
    if isinstance(value, bool):
        if isinstance(actual, str):
            actual = actual.strip().lower() in ("true", "yes", "1")
        valid = isinstance(actual, (bool, int)) and bool(actual) is value
    elif isinstance(value, (int, float)):
        try:
            valid = math.isclose(float(actual), float(value),
                                 rel_tol=1e-8, abs_tol=abs_tol)
        except (TypeError, ValueError):
            valid = False
    else:
        valid = actual == value
    if not valid:
        raise PhasePropertyError("Phase-specific readback mismatch: " + property_name)
    return actual

def set_node_to_node_prestress(g_i, support, phase, signed_force_kn,
                               *, compression_sign_verified=False):
    """Set command prestress; actual support axial force requires Output QA."""
    if not compression_sign_verified:
        raise PhasePropertyError("Compression sign convention has not been verified")
    if (isinstance(signed_force_kn, bool) or
            not isinstance(signed_force_kn, (int, float)) or
            not math.isfinite(signed_force_kn) or signed_force_kn == 0):
        raise PhasePropertyError("Prestress must be a nonzero finite signed force")
    set_phase_property(g_i, support, phase, "AdjustPrestress", True)
    set_phase_property(g_i, support, phase, "PrestressForce", signed_force_kn)
    return {"command_prestress_kn": float(signed_force_kn),
            "actual_axial_force_kn": None,
            "requires_output_check": True}
