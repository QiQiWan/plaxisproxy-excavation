"""Explicit-phase Output force reads; independent from Input command prestress."""
from __future__ import annotations

import math
from collections.abc import Mapping


class PhaseResultError(RuntimeError):
    """Output force data could not be retrieved and validated."""


def read_node_to_node_axial_force(g_o, support_o, phase_o, *, result_type,
                                  force_unit, location='node'):
    """Read individual signed force samples from an explicit Output support/phase.

    Caller resolves Output handles and the axial force ResultTypes handle in the
    installed build (e.g. NodeToNodeAnchor.AnchorForce3D), verifies the phase has
    converged, confirms the model force unit is kN and establishes sign/connectivity.
    Input handles, names and list indices must not be reused as Output identities.
    No fallback to another phase, location or result type. No summing endpoint
    samples: these may be repeated samples of the same physical anchor force.
    """
    if support_o is None or phase_o is None or result_type is None:
        raise PhaseResultError('Explicit Output support, phase and result type required')
    if force_unit != 'kN':
        raise PhaseResultError('Confirm Output force unit is kN before reading')
    if location not in ('node', 'stresspoint'):
        raise PhaseResultError('Explicit node or stresspoint result location required')
    try:
        raw = g_o.getresults(support_o, phase_o, result_type, location)
        if isinstance(raw, (str, bytes, Mapping)):
            raise ValueError('Output returned text, not force samples')
        values = list(raw)
        if not values or any(type(v) not in (int, float) or not math.isfinite(v)
                             for v in values):
            raise ValueError('Output force samples must be nonempty finite numbers')
    except Exception as exc:
        raise PhaseResultError('Explicit-phase Output axial force read failed') from exc
    return {'axial_force_kn': [float(v) for v in values],
            'source': 'PLAXIS Output', 'location': location,
            'force_unit': 'kN', 'requires_mechanical_validation': True}
