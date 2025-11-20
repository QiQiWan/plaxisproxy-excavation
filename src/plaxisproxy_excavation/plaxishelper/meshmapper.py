# meshmapper.py
from __future__ import annotations
from typing import Any, Optional, Sequence, Dict, List, Tuple
from ..components.mesh import Mesh, MeshCoarseness


# ########## small helpers ##########
def _as_float(v: Any) -> Optional[float]:
    """Return float(v) if v is numeric (and not bool); otherwise None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None

def _as_int(v: Any) -> Optional[int]:
    """Return int(v) if v is integral (and not bool); otherwise None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    return None

def _as_bool(v: Any) -> Optional[bool]:
    """Return v if v is bool; otherwise None."""
    if isinstance(v, bool):
        return v
    return None

def _extract_factor(mesh: Mesh) -> Optional[float]:
    """
    Decide the meshing 'factor' (relative element size) for key-value style call.
    Preferred: mesh.element_relative_size 
    Fallback : mesh.mesh_coarseness.value 
    """
    ers = getattr(mesh, "element_relative_size", None)
    if isinstance(ers, (int, float)):
        return float(ers)
    coarse = getattr(mesh, "mesh_coarseness", None)
    if isinstance(coarse, MeshCoarseness):
        return float(coarse.value)
    return None


def _build_candidate_dicts(mesh: Mesh) -> List[Dict[str, Any]]:
    """
    Build several candidate key-value dicts and try them sequentially with
    g_i.mesh("k", v, ...). The first successful call stops the process.

    IMPORTANT:
    We NEVER put ElementRelativeSize and Coarseness in the same dict to avoid
    the PLAXIS error:
        Parameters ["Coarseness" "ElementRelativeSize"] cannot be used together.
    """
    factor = _extract_factor(mesh)
    UseEnhancedRefinements = _as_bool(getattr(mesh, "enhanced_refine", None))
    EMRGlobalScale = _as_float(getattr(mesh, "emr_global_scale", None))
    EMRMinElementSize = _as_float(getattr(mesh, "emr_min_elem", None))
    UseSweptMeshing = _as_bool(getattr(mesh, "swept_mesh", None))
    # Custom parameters
    MaxCPUs = _as_int(getattr(mesh, "max_cpus", None))
    EMRProximity = _as_float(getattr(mesh, "emr_proximity", None)) 
    ElementRelativeSize = _as_float(getattr(mesh, "element_relative_size", None))
    
    # Filter None but keep False/0 values
    def pack(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if v is not None}

    # Plan A: Use Coarseness (+ optional EMR fields)
    if not UseEnhancedRefinements:
        dict_A = pack({
            "Coarseness": factor,
            "UseEnhancedRefinements": UseEnhancedRefinements,
            "UseSweptMeshing": UseSweptMeshing
        })
    else:
        dict_A = pack({
            "Coarseness": factor,
            "UseEnhancedRefinements": UseEnhancedRefinements,
            "EMRGlobalScale": EMRGlobalScale,
            "EMRMinElementSize": EMRMinElementSize,
            "UseSweptMeshing": UseSweptMeshing
        })

    # Plan B: Start from A and add optional params
    if MaxCPUs:
        dict_A["MaxCPUs"] = MaxCPUs
    if EMRProximity:
        dict_A["EMRProximity"] = EMRProximity
    if ElementRelativeSize:
        # Avoid conflict: if we set ElementRelativeSize, remove Coarseness
        dict_A["ElementRelativeSize"] = ElementRelativeSize
        if "Coarseness" in dict_A:
            del dict_A["Coarseness"]

    # Plan C: Only Coarseness
    dict_C = pack({"Coarseness": factor})

    # Plan D: No parameters (let PLAXIS defaults decide)
    dict_D: Dict[str, Any] = {}

    # Keep non-empty dicts (leave the last empty dict as fallback if needed)
    cands = [d for d in [dict_A, dict_C, dict_D] if d]
    return cands


def _flatten_kv(d: Dict[str, Any]) -> List[Any]:
    """
    {"A":1, "B":True} -> ["A", 1, "B", True]
    Order is generally not critical because PLAXIS parses by key names.
    """
    kv: List[Any] = []
    for k, v in d.items():
        kv.append(str(k))
        kv.append(v)
    return kv


def _try_mesh_kv(g_i: Any, d: Dict[str, Any]) -> str:
    """
    Try to execute one call of g_i.mesh("k1", v1, "k2", v2, ...).
    If dict is empty, call g_i.mesh() with no parameters.

    Returns a non-empty string ("OK" if PLAXIS returns nothing) on success.
    Raises the PLAXIS exception on failure.
    """
    kv = _flatten_kv(d)
    if kv:
        result = g_i.mesh(*kv)
    else:
        result = g_i.mesh()
    return "OK" if result is None else str(result)


class MeshMapper:
    """
    Minimal key-value-style meshing:
      - Always call g_i.gotomesh() first
      - Then call g_i.mesh("key", value, ...) with robust, conflict-free fallbacks

    It builds several candidate dictionaries (ElementRelativeSize-only, Coarseness-only,
    with/without EMR, etc.) and tries them in order until one succeeds.
    This avoids the conflict:
        "Parameters ['Coarseness' 'ElementRelativeSize'] cannot be used together."
    """

    @staticmethod
    def generate(g_i: Any, mesh: Mesh) -> str:
        # Enter the mesh context first
        goto = getattr(g_i, "gotomesh", None)
        if callable(goto):
            goto()

        last_err: Optional[Exception] = None
        for d in _build_candidate_dicts(mesh):
            try:
                if _try_mesh_kv(g_i, d):
                    return "OK"
            except Exception as e:
                last_err = e
                continue

        if last_err:
            raise last_err
        raise RuntimeError("g_i.mesh() failed unexpectedly.")
