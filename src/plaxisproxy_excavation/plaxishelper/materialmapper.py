from __future__ import annotations
"""
Unified PLAXIS material mappers with create/delete and back-reference handling.

This module provides static mapper classes to create and configure PLAXIS
materials from your in-memory domain objects AND to delete them later.

What you get:
- Robust creation: normalizes constructor returns (tuple/list/handle),
  sets properties with tolerant fallbacks, and writes the created handle
  back to `mat.plx_id`.
- Safe deletion: tries several common deletion entrypoints; on success,
  resets the passed material object's `plx_id` to None BUT does not
  delete the in-memory material object itself.

Mappers included:
- SoilMaterialMapper (MC / MCC / HSS)
- PlateMaterialMapper (Elastic / Elastoplastic)
- BeamMaterialMapper (Elastic / Elastoplastic)
- PileMaterialMapper (Elastic / Elastoplastic)
- AnchorMaterialMapper (Elastic / Elastoplastic / Residual)

All public entrypoints:
    create_material(g_i, mat) -> plx_handle
    delete_material(g_i, mat_or_plx) -> bool

Copy-paste and adapt the property keys to your specific PLAXIS binding if needed.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union, Optional, Iterable, Tuple
from math import tan, radians, isfinite
from enum import Enum

# =============================================================================
# Domain imports (adapt to your project structure)
# =============================================================================
# -- Soil
from ..materials.soilmaterial import (
    BaseSoilMaterial, MCMaterial, MCCMaterial, HSSMaterial,
    SoilMaterialsType, RayleighInputMethod
)
# -- Plate
from ..materials.platematerial import (
    PlateType, ElasticPlate, ElastoplasticPlate
)
# -- Beam / Pile
from ..materials.beammaterial import (
    BeamType, CrossSectionType, PreDefineSection,
    ElasticBeam, ElastoplasticBeam
)
# -- Pile
from ..materials.pilematerial import (
    LateralResistanceType,
    ElasticPile, ElastoplasticPile,
)
# -- Anchor
from ..materials.anchormaterial import (
    AnchorType,
    ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor,
)

# =============================================================================
# Small utilities shared by all mappers
# =============================================================================

def _normalize_created_handle(created: Any) -> Any:
    """Return a stable PLAXIS handle (first element if list/tuple)."""
    if isinstance(created, (list, tuple)) and created:
        return created[0]
    return created

def _write_backref(mat_obj: Any, plx_obj: Any) -> None:
    """Write PLAXIS handle back to the material object's `plx_id` attribute."""
    try:
        mat_obj.plx_id = plx_obj
    except Exception:
        pass

def _try_delete_with_gi(g_i: Any, plx_obj: Any) -> bool:
    """
    Try to delete a PLAXIS material using several common entrypoints.
    Returns True if any attempt succeeds, False otherwise.
    """
    # 0) Prefer object's own delete()
    try:
        if hasattr(plx_obj, "delete"):
            plx_obj.delete()
            return True
    except Exception:
        pass

    # 1) Try deletion methods on g_i in a robust order
    for fn_name in ("delete", "delobject", "deletematerial", "delmaterial", "remove"):
        try:
            fn = getattr(g_i, fn_name, None)
            if callable(fn):
                fn(plx_obj)
                return True
        except Exception:
            continue

    return False

# ########################### logging helpers (single-line) ###################

def _enum_to_str(v: Any) -> str:
    try:
        if hasattr(v, "name"):
            return str(v.name)
        if hasattr(v, "value"):
            return str(v.value)
        return str(v)
    except Exception:
        return str(v)

def _get_attr_value(obj: Any, key: str) -> Optional[str]:
    try:
        v = getattr(obj, key, None)
        if v is None:
            return None
        # Some PLAXIS bindings wrap values with `.value`
        if hasattr(v, "value"):
            v = v.value
        return str(v)
    except Exception:
        return None

def _format_handle(h: Any) -> str:
    if h is None:
        return "None"
    # try common identifiers first
    for k in ("Id", "ID", "id", "python_id", "Guid", "GUID", "guid",
              "MaterialName", "Identification", "Name"):
        val = _get_attr_value(h, k)
        if val:
            return f"{k}={val}"
    s = str(h)
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) > 120:
        s = s[:117] + "..."
    return s

def _one_line(s: str) -> str:
    return " ".join(str(s).split())

def _log_create(kind: str, mat_obj: Any, plx_obj: Any) -> None:
    name = getattr(mat_obj, "name", None) or _get_attr_value(plx_obj, "MaterialName") or "N/A"
    mtype = _enum_to_str(getattr(mat_obj, "type", "N/A"))
    handle = _format_handle(plx_obj)
    print(_one_line(f"[PLAXIS 3D][CREATE][{kind}] name='{name}' type={mtype} plx_id={handle}"), flush=True)

def _log_delete(kind: str, mat_obj: Any, plx_obj: Any, ok: bool) -> None:
    name = getattr(mat_obj, "name", None) if mat_obj is not None else None
    if not name:
        # try read from plx handle
        name = _get_attr_value(plx_obj, "MaterialName") or "N/A"
    mtype = _enum_to_str(getattr(mat_obj, "type", "N/A")) if mat_obj is not None else "N/A"
    handle = _format_handle(plx_obj)
    status = "OK" if ok else "FAIL"
    print(_one_line(f"[PLAXIS 3D][DELETE][{kind}] name='{name}' type={mtype} plx_id={handle} result={status}"), flush=True)

# =============================================================================
# SoilMaterialMapper
# =============================================================================

class SoilMaterialMapper:
    """
    Static helper to create and configure PLAXIS soil materials from
    BaseSoilMaterial (MC/MCC/HSS) instances.

    - Uses g_i.soilmat() to create a new material.
    - Sets common properties (name, model, unit weights, elasticity, etc.).
    - Sets model-specific properties for MC/MCC/HSS.
    - Applies optional groups: Rayleigh damping, pore stress, groundwater,
      interface, initial K0, if the corresponding flags were set on the material.

    MCC-specific interface inputs:
    - For MCC only, if the material instance provides any of
        * explicit override:  _R_inter
        * target strengths:   c_inter (cohesion), phi_in (friction angle)
        * psi_inter (accepted but not written to SoilMat)
      we set InterfaceStrength="Manual" and:
        - if _R_inter is given, we use it as Rinter;
        - else, derive Rinter conservatively:
              Rinter = min( c_inter / cref,  tan(phi_in) / tan(phi) ), clipped to [0,1].
      Derivation uses the soil's own cref/phi that have been written to the PLAXIS object.
    """

    # ########################### public entrypoints ###########################
    @staticmethod
    def create_material(g_i: Any, mat: BaseSoilMaterial) -> Any:
        """
        Create a PLAXIS soil material and push all properties from `mat`.
        Returns the created PLAXIS material handle, and writes it back to mat.plx_id.
        """
        # 1) Create raw PLAXIS material
        plx_mat = SoilMaterialMapper._create_raw_soil(g_i)
        plx_mat = _normalize_created_handle(plx_mat)

        # 2) Common properties
        common_props = SoilMaterialMapper._map_common_props(mat)
        SoilMaterialMapper._set_many_props(plx_mat, common_props)

        # 3) Model-specific properties
        if isinstance(mat, MCMaterial):
            SoilMaterialMapper._set_many_props(plx_mat, SoilMaterialMapper._map_mc_props(mat))
        elif isinstance(mat, MCCMaterial):
            SoilMaterialMapper._set_many_props(plx_mat, SoilMaterialMapper._map_mcc_props(mat))
            SoilMaterialMapper._apply_mcc_interface(plx_mat, mat)
        elif isinstance(mat, HSSMaterial):
            SoilMaterialMapper._set_many_props(plx_mat, SoilMaterialMapper._map_hss_props(mat))
        else:
            raise TypeError(f"Unsupported soil material class: {type(mat).__name__}")

        # 4) Optional groups
        SoilMaterialMapper._apply_optional_groups(plx_mat, mat)

        # 5) Backref & log
        _write_backref(mat, plx_mat)
        _log_create("Soil", mat, plx_mat)
        return plx_mat

    @staticmethod
    def delete_material(g_i: Any, mat_or_plx: Union[BaseSoilMaterial, Any]) -> bool:
        """
        Delete the soil material in PLAXIS. On success, reset `mat.plx_id` to None,
        but DO NOT delete the in-memory material object.
        """
        mat_obj: Optional[BaseSoilMaterial] = None
        plx_obj: Any = None

        if isinstance(mat_or_plx, BaseSoilMaterial):
            mat_obj = mat_or_plx
            plx_obj = getattr(mat_obj, "plx_id", None)
        else:
            plx_obj = mat_or_plx

        if plx_obj is None:
            _log_delete("Soil", mat_obj, None, ok=False)
            return False

        ok = _try_delete_with_gi(g_i, plx_obj)
        if ok and mat_obj is not None:
            _write_backref(mat_obj, None)
        _log_delete("Soil", mat_obj, plx_obj, ok=ok)
        return ok

    # ########################### creation primitive ##########################
    @staticmethod
    def _create_raw_soil(g_i: Any) -> Any:
        """
        Create a raw soil material in PLAXIS. The common pattern is:
            plx_mat = g_i.soilmat()
        Some bindings return a tuple; normalize to the first element.
        """
        created = g_i.soilmat()
        return _normalize_created_handle(created)

    # ########################### property setting ############################
    @staticmethod
    def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
        """
        Try setting properties in a robust way:
        1) If obj has 'setproperties', use a single call with key/value sequence.
        2) Else, fallback to setattr for each property name.
        Skips None values.
        """
        if not props:
            return
        filtered: Dict[str, Any] = {k: v for k, v in props.items() if v is not None}

        for k, v in filtered.items():
            if isinstance(v, Enum):
                v = v.value
            try:
                setattr(plx_obj, k, v)
                if k == "GwUseDefaults" and bool(getattr(plx_obj, k)) != v:
                    # If GwUseDefaults was not updated yet, loop the assignment operation.
                    while bool(getattr(plx_obj, k)) == v:
                        setattr(plx_obj, k, v)
                
            except Exception:
                if hasattr(plx_obj, "setproperties"):
                    try:
                        plx_obj.setproperty(k, v)
                        continue
                    except Exception:
                        pass
                if hasattr(plx_obj, "setproperty"):
                    try:
                        plx_obj.setproperty(k, v)
                        continue
                    except Exception:
                        pass

    # ########################### helpers ####################################
    @staticmethod
    def _first_not_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    @staticmethod
    def _safe_get_value(plx_obj: Any, key: str):
        """
        Safely read numeric value from PLAXIS prop: returns float or None.
        """
        try:
            attr = getattr(plx_obj, key, None)
            if attr is None:
                return None
            return float(attr.value)
        except Exception:
            return None

    # ########################### common mappings #############################
    @staticmethod
    def _map_common_props(mat: BaseSoilMaterial) -> Dict[str, Any]:
        """
        Map fields common to all soil materials.
        - For MCC only: nu prefers mat.nu; if None, falls back to mat.v_ur.
        """
        # nu fallback to v_ur only for MCC
        if isinstance(mat, MCCMaterial):
            nu_value = SoilMaterialMapper._first_not_none(
                getattr(mat, "nu", None),
                getattr(mat, "v_ur", None)
            )
        else:
            nu_value = getattr(mat, "nu", None)

        return {
            "MaterialName":   getattr(mat, "name", "Soil"),
            "Identification": getattr(mat, "name", "Soil"),
            # Do not rely on enum.value here if your binding expects a fixed string.
            "SoilModel":      getattr(mat, "type", SoilMaterialsType.MC).value,
            "gammaUnsat":     getattr(mat, "gamma", None),
            "gammaSat":       getattr(mat, "gamma_sat", None),
            "nu":             nu_value,
            "Comments":       getattr(mat, "comment", ""),
            "eInit":          getattr(mat, "e_init", None),
        }

    # ########################### model-specific mappings #####################
    @staticmethod
    def _map_mc_props(mc: MCMaterial) -> Dict[str, Any]:
        """Mohr-Coulomb model properties."""
        return {
            "Eref":            getattr(mc, "E_ref", None),
            "cref":            getattr(mc, "c_ref", None),
            "phi":             getattr(mc, "phi", None),
            "psi":             getattr(mc, "psi", None),
            "TensileStrength": getattr(mc, "tensile_strength", None),
        }

    @staticmethod
    def _map_mcc_props(mcc: MCCMaterial) -> Dict[str, Any]:
        """
        Modified Cam-clay model properties with MCC-specific interface inputs.
        """
        props = {
            "kappa":   getattr(mcc, "kar", 0.2),
            "lambda":  getattr(mcc, "lam", 0.03),
            "M":       getattr(mcc, "M_CSL", 1.2),
            # Interface inputs (kept as fields for compatible bindings)
            "CInter":   getattr(mcc, "c_inter", 30e3),
            "PhiInter": getattr(mcc, "phi_in", 30.0),
            "PsiInter": getattr(mcc, "psi_inter", 15.0),
        }

        explicitR = getattr(mcc, "_R_inter", None)
        if explicitR is not None:
            props["Rinter"] = explicitR

        return {k: v for k, v in props.items() if v is not None}

    @staticmethod
    def _map_hss_props(hss: HSSMaterial) -> Dict[str, Any]:
        """Hardening Soil small-strain properties."""
        return {
            "E50Ref":  getattr(hss, "E", None),
            "EOedRef": getattr(hss, "E_oed", None),
            "EURRef":  getattr(hss, "E_ur", None),
            "nuUR":    getattr(hss, "nu", 0.2),
            "pref":    getattr(hss, "P_ref", None),
            "G0Ref":   getattr(hss, "G0", None),
            "gamma07": getattr(hss, "gamma_07", None),
            "c":       getattr(hss, "c", None),
            "phi":     getattr(hss, "phi", None),
            "psi":     getattr(hss, "psi", None),
        }

    # ########################### MCC interface handling ######################
    @staticmethod
    def _apply_mcc_interface(plx_mat: Any, mcc: MCCMaterial) -> None:
        """
        MCC-only interface handler:
        - If the MCC instance exposes c_inter / phi_in / psi_inter or explicit _R_inter,
          enable Manual interface strength and set/derive Rinter.
        """
        c_inter   = getattr(mcc, "c_inter", None)
        phi_in    = getattr(mcc, "phi_in", None)
        explicitR = getattr(mcc, "_R_inter", None)

        # Nothing to do if MCC does not provide any interface inputs
        if all(v is None for v in (c_inter, phi_in, explicitR)):
            return

        # Switch to Manual interface strength when any interface inputs appear
        try:
            plx_mat.InterfaceStrength = "Manual"
        except Exception:
            pass

        # 1) Explicit Rinter takes precedence
        if explicitR is not None:
            try:
                plx_mat.Rinter = float(explicitR)
            except Exception:
                pass
            return

        # 2) Derive Rinter from ratios if possible
        ratios: List[float] = []

        cref_soil = SoilMaterialMapper._safe_get_value(plx_mat, "cref")
        phi_soil  = SoilMaterialMapper._safe_get_value(plx_mat, "phi")

        if (c_inter is not None) and (cref_soil is not None) and cref_soil > 1e-12:
            ratios.append(float(c_inter) / float(cref_soil))

        if (phi_in is not None) and (phi_soil is not None) and phi_soil > 1e-6:
            try:
                ratios.append(tan(radians(float(phi_in))) / tan(radians(phi_soil)))
            except Exception:
                pass

        if ratios:
            r_val = max(0.0, min(min(ratios), 1.0))
            try:
                plx_mat.Rinter = r_val
            except Exception:
                pass

    # ########################### optional groups #############################
    @staticmethod
    def _apply_optional_groups(plx_mat: Any, mat: BaseSoilMaterial) -> None:
        """
        Apply optional settings if the user enabled them on the material
        (Rayleigh, pore stress, groundwater, additional interface/initials).
        """
        # 1) Rayleigh damping (alpha/beta)  —— 如需启用，按需映射
        # if hasattr(mat, "_input_method") and mat._input_method == RayleighInputMethod.Direct:
        #     props = {}
        #     if hasattr(mat, "_alpha"): props["RayleighAlpha"] = mat._alpha
        #     if hasattr(mat, "_beta"):  props["RayleighBeta"]  = mat._beta
        #     SoilMaterialMapper._set_many_props(plx_mat, props)

        # 2) Super pore stress
        if getattr(mat, "_set_pore_stress", False):
            props = {
                "PoreStressType": getattr(mat, "_pore_stress_type", None),
                "VU":             getattr(mat, "_vu", None),
                "PoreValue":      getattr(mat, "_water_value", None),
            }
            SoilMaterialMapper._set_many_props(plx_mat, props)

        # 3) Under-ground water
        if getattr(mat, "_set_ug_water", False):
            props = {
                "GroundwaterClassificationType":    getattr(mat, "_Gw_type", None),
                "SWCCFittingMethod":                getattr(mat, "_SWCC_method", None),
                "SoilPosi":                         getattr(mat, "_soil_posi", None),
                "SoilFine":                         getattr(mat, "_soil_fine", None),
                "GwUseDefaults":                    getattr(mat, "_Gw_defaults", None),
                "Infiltration":                     getattr(mat, "_infiltration", None),
                "DefaultMethod":                    getattr(mat, "_default_method", None),
                "PermHorizontalPrimary":            getattr(mat, "_kx", None),
                "PermHorizontalSecondary":          getattr(mat, "_ky", None),
                "PermVertical":                     getattr(mat, "_kz", None),
                "GwPsiUnsat":                       getattr(mat, "_Gw_Psiunsat", None),
            }
            SoilMaterialMapper._set_many_props(plx_mat, props)

        # 4) Additional interface
        if getattr(mat, "_set_additional_interface", False):
            props = {
                "InterfaceStiffnessDetermination":  getattr(mat, "_stiffness_define", None),
                "InterfaceStrengthDef":             getattr(mat, "_strengthen_define", None),
                "knInter":                          getattr(mat, "_k_n", None),
                "ksInter":                          getattr(mat, "_k_s", None),
                "GapClosure":                       getattr(mat, "_gap_closure", None),
                "CrossPermeability":                getattr(mat, "_cross_permeability", None),
                "DrainageConductivity1":            getattr(mat, "_drainage_conduct1", None),
                "DrainageConductivity2":            getattr(mat, "_drainage_conduct2", None),
            }
            SoilMaterialMapper._set_many_props(plx_mat, props)

        # 5) Additional initial K0
        if getattr(mat, "_set_additional_initial", False):
            props = {
                "K0Determination": getattr(mat, "_K_0_define", None),
                "K0Primary":           getattr(mat, "_K_0_x", None),
                "K0Secondary":           getattr(mat, "_K_0_y", None),
            }
            SoilMaterialMapper._set_many_props(plx_mat, props)

# =============================================================================
# PlateMaterialMapper
# =============================================================================

class PlateMaterialMapper:
    """
    Static helper to create and configure PLAXIS plate materials from
    ElasticPlate / ElastoplasticPlate instances.

    - Uses g_i.platemat() to create a new plate material.
    - Sets common properties (name, identification, E, nu, thickness, etc.).
    - Handles isotropy: if isotropic=True and any G* is missing/zero,
      fills G = E / [2(1+nu)].
    - For ElastoplasticPlate, also sets yield strengths and section moduli.
    """

    # ########################### public entrypoints ###########################
    @staticmethod
    def create_material(g_i: Any, mat: ElasticPlate) -> Any:
        """
        Create a PLAXIS plate material and push all properties from `mat`.
        Returns the created PLAXIS material handle, and writes it back to mat.plx_id.
        """
        if not isinstance(mat, (ElasticPlate, ElastoplasticPlate)):
            raise TypeError(f"Unsupported plate material class: {type(mat).__name__}")

        PlateMaterialMapper._normalize_plate_object(mat)
        PlateMaterialMapper._ensure_isotropic_G(mat)

        plx_mat = PlateMaterialMapper._create_raw_plate(g_i)
        plx_mat = _normalize_created_handle(plx_mat)

        # Write properties
        PlateMaterialMapper._set_many_props(plx_mat, PlateMaterialMapper._map_common_props(mat))
        PlateMaterialMapper._set_many_props(plx_mat, PlateMaterialMapper._map_elastic_props(mat))
        if isinstance(mat, ElastoplasticPlate):
            PlateMaterialMapper._set_many_props(plx_mat, PlateMaterialMapper._map_elastoplastic_props(mat))
        PlateMaterialMapper._set_many_props(plx_mat, PlateMaterialMapper._map_derived_annotations(mat))

        # Backref & log
        _write_backref(mat, plx_mat)
        _log_create("Plate", mat, plx_mat)
        return plx_mat

    @staticmethod
    def delete_material(g_i: Any, mat_or_plx: Union[ElasticPlate, ElastoplasticPlate, Any]) -> bool:
        """
        Delete the plate material in PLAXIS. On success reset `mat.plx_id = None`.
        """
        mat_obj: Optional[Union[ElasticPlate, ElastoplasticPlate]] = None
        plx_obj: Any = None

        if isinstance(mat_or_plx, (ElasticPlate, ElastoplasticPlate)):
            mat_obj = mat_or_plx
            plx_obj = getattr(mat_obj, "plx_id", None)
        else:
            plx_obj = mat_or_plx

        if plx_obj is None:
            _log_delete("Plate", mat_obj, None, ok=False)
            return False

        ok = _try_delete_with_gi(g_i, plx_obj)
        if ok and mat_obj is not None:
            _write_backref(mat_obj, None)
        _log_delete("Plate", mat_obj, plx_obj, ok=ok)
        return ok

    # ########################### creation primitive ##########################
    @staticmethod
    def _create_raw_plate(g_i: Any) -> Any:
        """
        Create a raw plate material in PLAXIS. Common pattern:
            plx_mat = g_i.platemat()
        """
        created = getattr(g_i, "platemat")()
        return _normalize_created_handle(created)

    # ########################### property setting ############################
    @staticmethod
    def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
        """
        Prefer per-key plx_obj.setproperties(k, v); fallback to setattr/setproperty; skip None.
        """
        if not props:
            return
        filtered = {k: v for k, v in props.items() if v is not None}

        for k, v in filtered.items():
            try:
                if hasattr(plx_obj, "setproperties"):
                    plx_obj.setproperties(k, v)
                    continue
            except Exception:
                pass
            try:
                setattr(plx_obj, k, v)
                continue
            except Exception:
                pass
            try:
                if hasattr(plx_obj, "setproperty"):
                    plx_obj.setproperty(k, v)
            except Exception:
                pass

    # ########################### helpers ####################################
    @staticmethod
    def _normalize_plate_object(mat: ElasticPlate) -> None:
        """Normalize/repair fields on the in-memory plate object."""
        if hasattr(mat, "_preventpuch") and not hasattr(mat, "_preventpunch"):
            try:
                mat._preventpunch = bool(getattr(mat, "_preventpuch"))
            except Exception:
                pass

    @staticmethod
    def _ensure_isotropic_G(mat: ElasticPlate) -> None:
        """If isotropic and any G* is None or <=0, back-fill from E, nu."""
        try:
            if not mat.isotropic:
                return
        except Exception:
            return

        def _need_fill(val) -> bool:
            try:
                return (val is None) or (float(val) <= 0.0) or (not isfinite(float(val)))
            except Exception:
                return True

        try:
            G_iso = float(mat.E) / (2.0 * (1.0 + float(mat.nu)))
        except Exception:
            return

        if _need_fill(getattr(mat, "_G12", None)):
            mat._G12 = G_iso
        if _need_fill(getattr(mat, "_G13", None)):
            mat._G13 = G_iso
        if _need_fill(getattr(mat, "_G23", None)):
            mat._G23 = G_iso

    # ########################### mappings ###################################
    @staticmethod
    def _map_common_props(mat: ElasticPlate) -> Dict[str, Any]:
        """Map fields common to all plate materials (adapt keys as needed)."""
        return {
            "MaterialName":   getattr(mat, "name", "Plate"),
            "Identification": getattr(mat, "name", "Plate"),
            "Comments":       getattr(mat, "comment", ""),
            "MaterialType":   getattr(mat, "type", PlateType.Elastic).value,
            "Gamma":          getattr(mat, "gamma", None),
            "PreventPunching": getattr(mat, "preventpunch", True),
            "Isotropic":      getattr(mat, "isotropic", True),
        }

    @staticmethod
    def _map_elastic_props(mat: ElasticPlate) -> Dict[str, Any]:
        """Elastic (orthotropic-capable) properties."""
        return {
            "E1":         getattr(mat, "E", None),
            "StructNu12": getattr(mat, "nu", None),
            "D3d":        getattr(mat, "d", None),
            "E2":         getattr(mat, "E2", None),
            "G12":        getattr(mat, "G12", None),
            "G13":        getattr(mat, "G13", None),
            "G23":        getattr(mat, "G23", None),
        }

    @staticmethod
    def _map_elastoplastic_props(mat: ElastoplasticPlate) -> Dict[str, Any]:
        """Elastoplastic additions (yield strengths and section moduli)."""
        return {
            "YieldStress11": getattr(mat, "sigma_y_11", None),
            "W11":           getattr(mat, "W_11", None),
            "YieldStress22": getattr(mat, "sigma_y_22", None),
            "W22":           getattr(mat, "W_22", None),
        }

    @staticmethod
    def _map_derived_annotations(mat: ElasticPlate) -> Dict[str, Any]:
        """
        Optional derived quantities as comments. If your PLAXIS API allows
        setting 'Comments', you can append a compact note here.
        """
        try:
            D = mat.bending_rigidity()  # kPa·m^3
        except Exception:
            D = None
        try:
            q = mat.self_weight_load()  # kN/m^2
        except Exception:
            q = None

        if D is None and q is None:
            return {}

        note_parts = []
        if D is not None:
            note_parts.append(f"D={D:.6g} kPa·m^3")
        if q is not None:
            note_parts.append(f"q_self={q:.6g} kN/m^2")
        note = " | ".join(note_parts)

        base_cmt = getattr(mat, "comment", "") or ""
        if base_cmt:
            cmt = f"{base_cmt} | {note}"
        else:
            cmt = note
        return {"Comments": cmt}

# =============================================================================
# BeamMaterialMapper
# =============================================================================

# Centralized key map — edit here to match your PLAXIS binding (names on the left are mapper keys)
BEAM_KEYS = {
    "MaterialName": "MaterialName",
    "Identification": "Identification",
    "Comments": "Comments",
    "MaterialType": "MaterialType",        # "Elastic" / "Elasto-plastic"
    "Gamma": "gamma",                      # optional unit weight
    # Elastic base
    "E": "E",
    "nu": "nu",
    # Cross-section typing
    "CrossSectionType": "CrossSectionType",     # "Predefined" / "userdefined"
    "PredefinedSection": "PredefinedCrossSectionType",   # "Cylinder"/"Rectangle"/"CircularArcBeam"
    # Geometry for predefined sections (traceable fields)
    "Diameter": "Diameter",                     # Cylinder
    "Thickness": "Thickness",                   # CircularArcBeam
    "Width": "Width",                           # Rectangle
    "Height": "Height",                         # Rectangle
    # Section properties (when Custom or after compute)
    "A": "A",
    "Iy": "I2",
    "Iz": "I3",
    "W": "W2",
    # Elastoplastic
    "YieldStress": "YieldStress",
    "YieldDirection": "YieldDirection",
    # Optional Rayleigh
    "RayleighAlpha": "RayleighAlpha",
    "RayleighBeta": "RayleighBeta",
}

class BeamMaterialMapper:
    """
    Create and configure PLAXIS beam materials from ElasticBeam / ElastoplasticBeam.

    - Uses g_i.beammat() to create the beam material.
    - Writes common props (name/type/gamma/E/nu).
    - Cross-section:
        * Predefined: writes PredefinedSection + geometry
                      and computed section props (A, Iy, Iz, W).
        * Custom: passes user-provided A/Iy/Iz/W.
    - Elastoplastic adds YieldStress / YieldDirection.
    """

    # ########################### public entrypoints ###########################
    @staticmethod
    def create_material(
        g_i: Any,
        mat: Union[ElasticBeam, ElastoplasticBeam],
    ) -> Any:
        if not isinstance(mat, (ElasticBeam, ElastoplasticBeam)):
            raise TypeError(f"Unsupported beam material class: {type(mat).__name__}")

        plx_mat = BeamMaterialMapper._create_raw_beam(g_i)
        plx_mat = _normalize_created_handle(plx_mat)

        BeamMaterialMapper._set_many_props(plx_mat, BeamMaterialMapper._map_common(mat))
        BeamMaterialMapper._set_many_props(plx_mat, BeamMaterialMapper._map_elastic_and_section(mat))

        if isinstance(mat, ElastoplasticBeam):
            BeamMaterialMapper._set_many_props(plx_mat, BeamMaterialMapper._map_elastoplastic(mat))

        BeamMaterialMapper._maybe_apply_rayleigh(plx_mat, mat)

        _write_backref(mat, plx_mat)
        _log_create("Beam", mat, plx_mat)
        return plx_mat

    @staticmethod
    def delete_material(g_i: Any, mat_or_plx: Union[ElasticBeam, ElastoplasticBeam, Any]) -> bool:
        mat_obj: Optional[Union[ElasticBeam, ElastoplasticBeam]] = None
        plx_obj: Any = None
        if isinstance(mat_or_plx, (ElasticBeam, ElastoplasticBeam)):
            mat_obj = mat_or_plx
            plx_obj = getattr(mat_obj, "plx_id", None)
        else:
            plx_obj = mat_or_plx

        if plx_obj is None:
            _log_delete("Beam", mat_obj, None, ok=False)
            return False

        ok = _try_delete_with_gi(g_i, plx_obj)
        if ok and mat_obj is not None:
            _write_backref(mat_obj, None)
        _log_delete("Beam", mat_obj, plx_obj, ok=ok)
        return ok

    # ########################### creation primitive ##########################
    @staticmethod
    def _create_raw_beam(g_i: Any) -> Any:
        created = getattr(g_i, "beammat")()
        return _normalize_created_handle(created)

    # ########################### property setting ############################
    @staticmethod
    def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
        """Prefer per-key setproperties(k, v); fallback to setattr/setproperty; skip None."""
        if not props:
            return
        for k, v in props.items():
            if v is None:
                continue
            try:
                if hasattr(plx_obj, "setproperties"):
                    plx_obj.setproperties(k, v)
                    continue
            except Exception:
                pass
            try:
                setattr(plx_obj, k, v)
                continue
            except Exception:
                pass
            try:
                if hasattr(plx_obj, "setproperty"):
                    plx_obj.setproperty(k, v)
            except Exception:
                pass

    # ########################### mappings ###################################
    @staticmethod
    def _map_common(mat: ElasticBeam) -> Dict[str, Any]:
        K = BEAM_KEYS
        type_map = {
            BeamType.Elastic: "Elastic",
            BeamType.Elastoplastic: "Elasto-plastic",
        }
        return {
            K["MaterialName"]:   getattr(mat, "name", "Beam"),
            K["Identification"]: getattr(mat, "name", "Beam"),
            K["Comments"]:       getattr(mat, "comment", ""),
            K["MaterialType"]:   type_map.get(getattr(mat, "type", BeamType.Elastic), "Elastic"),
            K["Gamma"]:          getattr(mat, "gamma", None),
            K["E"]:              getattr(mat, "E", None),
            K["nu"]:             getattr(mat, "nu", None),
        }

    @staticmethod
    def _map_elastic_and_section(mat: ElasticBeam) -> Dict[str, Any]:
        """Build dict with cross-section typing + geometry + section properties."""
        K = BEAM_KEYS
        props: Dict[str, Any] = {}

        if mat.cross_section == CrossSectionType.Custom:
            props.update({
                K["CrossSectionType"]: "userdefined",
                K["A"]:  getattr(mat, "A", None),
                K["Iy"]: getattr(mat, "Iy", None),
                K["Iz"]: getattr(mat, "Iz", None),
                K["W"]:  getattr(mat, "W", None),
            })
            return props

        # Predefined
        props[K["CrossSectionType"]] = "Predefined"

        predef = getattr(mat, "predefined_section", None)
        predef = predef or PreDefineSection.Cylinder
        props[K["PredefinedSection"]] = predef.value

        # Geometry
        if predef == PreDefineSection.Cylinder:
            props[K["Diameter"]] = getattr(mat, "len1", None) or getattr(mat, "diameter", None)
        elif predef == PreDefineSection.Rectangle:
            # width=len1, height=len2 by the domain object's convention
            props[K["Width"]]  = getattr(mat, "len1", None) or getattr(mat, "width", None)
            props[K["Height"]] = getattr(mat, "len2", None) or getattr(mat, "height", None)
        elif predef == PreDefineSection.CircularArcBeam:
            # treated as circular tube: outer from len1/diameter, thickness from len2/thickness
            props[K["Diameter"]]  = getattr(mat, "len1", None) or getattr(mat, "diameter", None)
            props[K["Thickness"]] = getattr(mat, "len2", None) or getattr(mat, "thickness", None)

        # Section properties (compute from helper)
        A, Iy, Iz, W = mat.section_properties(predefined=predef)
        props.update({
            K["A"]:  A,
            K["Iy"]: Iy,
            K["Iz"]: Iz,
            K["W"]:  W,
        })
        return props

    @staticmethod
    def _map_elastoplastic(mat: ElastoplasticBeam) -> Dict[str, Any]:
        K = BEAM_KEYS
        return {
            K["YieldStress"]:    getattr(mat, "sigma_y", None),
            K["YieldDirection"]: getattr(mat, "yield_dir", None),
        }

    # ########################### optional groups #############################
    @staticmethod
    def _maybe_apply_rayleigh(plx_mat: Any, mat: Union[ElasticBeam, ElastoplasticBeam]) -> None:
        K = BEAM_KEYS
        props: Dict[str, Any] = {}
        a = getattr(mat, "RayleighAlpha", None)
        b = getattr(mat, "RayleighBeta", None)
        if a is not None:
            props[K["RayleighAlpha"]] = a
        if b is not None:
            props[K["RayleighBeta"]]  = b
        BeamMaterialMapper._set_many_props(plx_mat, props)

# =============================================================================
# PileMaterialMapper
# =============================================================================

# Centralized keys – align with your binding
PILE_KEYS = {
    # Common
    "MaterialName": "MaterialName",
    "Identification": "Identification",
    "Comments": "Comments",
    "MaterialType": "MaterialType",
    "Gamma": "gamma",
    "E": "E",
    "nu": "nu",
    # Cross-section typing
    "CrossSectionType": "CrossSectionType",
    "PredefinedSection": "PredefinedCrossSectionType",
    # Geometry
    "Diameter": "Diameter",
    "Thickness": "Thickness",
    "Width": "Width",
    "Height": "Height",
    # Section properties
    "A": "A",
    "Iy": "I2",
    "Iz": "I3",
    "W": "W2",
    # Axial skin resistance (per UI)
    "AxialSkinType": "AxialSkinResistance",   # "Linear" / "Multi-linear" / "Layer dependent"
    "TSkinStartMax": "TSkinStartMax",         # kN/m
    "TSkinEndMax": "TSkinEndMax",             # kN/m
    # Base resistance
    "Fmax": "Fmax",                           # kN
    # Elastoplastic strength (optional)
    "YieldStress": "YieldStress",             # kN/m^2
    "YieldDirection": "YieldDirection",       # 1 / 2 / "local-2"
    # Optional Rayleigh
    "RayleighAlpha": "RayleighAlpha",
    "RayleighBeta": "RayleighBeta",
}

class PileMaterialMapper:
    """Create PLAXIS embedded pile/beam and push pile properties."""

    # ########################### public entrypoints ###########################
    @staticmethod
    def create_material(g_i: Any, mat: Union[ElasticPile, ElastoplasticPile]) -> Any:
        if not isinstance(mat, (ElasticPile, ElastoplasticPile)):
            raise TypeError(f"Unsupported pile material class: {type(mat).__name__}")

        plx_mat = PileMaterialMapper._create_raw_pile(g_i)
        plx_mat = _normalize_created_handle(plx_mat)

        PileMaterialMapper._set_many_props(plx_mat, PileMaterialMapper._map_common(mat))
        PileMaterialMapper._set_many_props(plx_mat, PileMaterialMapper._map_elastic_and_section(mat))
        PileMaterialMapper._set_many_props(plx_mat, PileMaterialMapper._map_axial_skin_and_base(mat))

        if isinstance(mat, ElastoplasticPile):
            PileMaterialMapper._set_many_props(plx_mat, PileMaterialMapper._map_elastoplastic(mat))

        PileMaterialMapper._maybe_apply_rayleigh(plx_mat, mat)

        _write_backref(mat, plx_mat)
        _log_create("Pile", mat, plx_mat)
        return plx_mat

    @staticmethod
    def delete_material(g_i: Any, mat_or_plx: Union[ElasticPile, ElastoplasticPile, Any]) -> bool:
        mat_obj: Optional[Union[ElasticPile, ElastoplasticPile]] = None
        plx_obj: Any = None
        if isinstance(mat_or_plx, (ElasticPile, ElastoplasticPile)):
            mat_obj = mat_or_plx
            plx_obj = getattr(mat_obj, "plx_id", None)
        else:
            plx_obj = mat_or_plx

        if plx_obj is None:
            _log_delete("Pile", mat_obj, None, ok=False)
            return False

        ok = _try_delete_with_gi(g_i, plx_obj)
        if ok and mat_obj is not None:
            _write_backref(mat_obj, None)
        _log_delete("Pile", mat_obj, plx_obj, ok=ok)
        return ok

    # ########################### creation primitive ##########################
    @staticmethod
    def _create_raw_pile(g_i: Any) -> Any:
        """
        Your binding uses embedded pile/beam constructor.
        Try `embeddedbeammat()`; normalize tuple/list to a single handle.
        """
        fn = getattr(g_i, "embeddedbeammat", None)
        if not callable(fn):
            raise RuntimeError("No embedded pile/beam constructor found in g_i (expected 'embeddedbeammat').")
        obj = fn()
        return _normalize_created_handle(obj)

    # ########################### property setter #############################
    @staticmethod
    def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
        if not props:
            return
        for k, v in props.items():
            if v is None:
                continue
            try:
                if hasattr(plx_obj, "setproperties"):
                    plx_obj.setproperties(k, v)
                    continue
            except Exception:
                pass
            try:
                setattr(plx_obj, k, v)
                continue
            except Exception:
                pass
            try:
                if hasattr(plx_obj, "setproperty"):
                    plx_obj.setproperty(k, v)
            except Exception:
                pass

    # ########################### mappings ###################################
    @staticmethod
    def _map_common(mat: ElasticPile) -> Dict[str, Any]:
        K = PILE_KEYS
        type_map = {BeamType.Elastic: "Elastic", BeamType.Elastoplastic: "Elasto-plastic"}
        return {
            K["MaterialName"]:   getattr(mat, "name", "Pile"),
            K["Identification"]: getattr(mat, "name", "Pile"),
            K["Comments"]:       getattr(mat, "comment", ""),
            K["MaterialType"]:   type_map.get(getattr(mat, "type", BeamType.Elastic), "Elastic"),
            K["Gamma"]:          getattr(mat, "gamma", None),
            K["E"]:              getattr(mat, "E", None),
            K["nu"]:             getattr(mat, "nu", None),
        }

    @staticmethod
    def _map_elastic_and_section(mat: ElasticPile) -> Dict[str, Any]:
        K = PILE_KEYS
        props: Dict[str, Any] = {}

        if mat.cross_section == CrossSectionType.Custom:
            props.update({
                K["CrossSectionType"]: "userdefined",
                K["A"]:  getattr(mat, "A", None),
                K["Iy"]: getattr(mat, "Iy", None),
                K["Iz"]: getattr(mat, "Iz", None),
                K["W"]:  getattr(mat, "W", None),
            })
            return props

        props[K["CrossSectionType"]] = "Predefined"
        predef = getattr(mat, "predefined_section", None) or PreDefineSection.Cylinder
        props[K["PredefinedSection"]] = predef.value

        if predef == PreDefineSection.Cylinder:
            props[K["Diameter"]] = getattr(mat, "diameter", None)
        elif predef == PreDefineSection.CircularArcBeam:
            props[K["Diameter"]]  = getattr(mat, "diameter", None)
            props[K["Thickness"]] = getattr(mat, "thickness", None)
        elif predef == PreDefineSection.Square:
            # rectangle pile (if used in your domain)
            props[K["Width"]]  = getattr(mat, "width", None)

        # Compute section properties
        A, Iy, Iz, W = mat.section_properties(predefined=predef)
        props.update({K["A"]: A, K["Iy"]: Iy, K["Iz"]: Iz, K["W"]: W})
        return props

    @staticmethod
    def _map_axial_skin_and_base(mat: ElasticPile) -> Dict[str, Any]:
        """Map Axial skin resistance + Base resistance per UI."""
        K = PILE_KEYS
        props: Dict[str, Any] = {}
        lt = getattr(mat, "lateral_type", None)
        lt_val = getattr(lt, "value", None) if lt is not None else None

        # Axial skin resistance mode
        props[K["AxialSkinType"]] = lt_val  # "Linear" / "Multi-linear" / "Layer dependent"

        # Linear → write T_skin max values
        if lt == LateralResistanceType.Linear:
            props[K["TSkinStartMax"]] = getattr(mat, "T_skin_start_max", None)
            props[K["TSkinEndMax"]]   = getattr(mat, "T_skin_end_max", None)

        # Base resistance
        props[K["Fmax"]] = getattr(mat, "F_max", None)
        return props

    @staticmethod
    def _map_elastoplastic(mat: ElastoplasticPile) -> Dict[str, Any]:
        """Strength block for elastoplastic pile: σ_y & critical direction (+ W2 safeguard)."""
        K = PILE_KEYS
        props: Dict[str, Any] = {
            K["YieldStress"]:    getattr(mat, "sigma_y", None),   # kN/m^2
            K["YieldDirection"]: getattr(mat, "yield_dir", None), # 1 / 2 / "local-2"
        }

        # If some bindings expect W2 within strength block, backfill if not set
        try:
            _, _, _, W = mat.section_properties(predefined=getattr(mat, "predefined_section", None))
            if W is not None and props.get(K["W"]) is None:
                props[K["W"]] = W
        except Exception:
            pass

        return props

    # ########################### optional groups #############################
    @staticmethod
    def _maybe_apply_rayleigh(plx_mat: Any, mat: Union[ElasticPile, ElastoplasticPile]) -> None:
        K = PILE_KEYS
        props: Dict[str, Any] = {}
        a = getattr(mat, "RayleighAlpha", None)
        b = getattr(mat, "RayleighBeta", None)
        if a is not None:
            props[K["RayleighAlpha"]] = a
        if b is not None:
            props[K["RayleighBeta"]]  = b
        PileMaterialMapper._set_many_props(plx_mat, props)

# =============================================================================
# AnchorMaterialMapper
# =============================================================================

# Centralized key map — adjust right-hand side to your PLAXIS binding keys
ANCHOR_KEYS = {
    "MaterialName": "MaterialName",
    "Identification": "Identification",
    "Comments": "Comments",
    "MaterialType": "MaterialType",     # "Elastic" / "Elasto-plastic" / "Elasto-plastic (Residual)"
    "EA": "EA",                         # kN
    # Elastoplastic capacities
    "FMaxTension": "FmaxTension",       # kN
    "FMaxCompression": "FmaxCompression",
    # Residual capacities (if your binding exposes them)
    "FResTension": "FresTension",
    "FResCompression": "FresCompression",
}

class AnchorMaterialMapper:
    """
    Static helper to create and configure PLAXIS anchor materials
    (node-to-node / cable anchors) from Elastic/Elastoplastic/Residual objects.

    - Tries g_i.n2nanchormat() first (Node-to-Node Anchor material).
      Falls back to common alternates like anchormat().
    - Writes common props (name/type/EA).
    - For elastoplastic variants, writes capacity limits; residual adds residual capacities.
    """

    # ########################### public entrypoints ###########################
    @staticmethod
    def create_material(
        g_i: Any,
        mat: Union[ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor],
    ) -> Any:
        if not isinstance(mat, (ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor)):
            raise TypeError(f"Unsupported anchor material class: {type(mat).__name__}")

        plx_mat = AnchorMaterialMapper._create_raw_anchor(g_i)
        plx_mat = _normalize_created_handle(plx_mat)

        # Common
        AnchorMaterialMapper._set_many_props(plx_mat, AnchorMaterialMapper._map_common(mat))

        # Elastoplastic additions
        if isinstance(mat, ElastoplasticAnchor):
            AnchorMaterialMapper._set_many_props(plx_mat, AnchorMaterialMapper._map_elastoplastic(mat))

        # Residual additions
        if isinstance(mat, ElastoPlasticResidualAnchor):
            AnchorMaterialMapper._set_many_props(plx_mat, AnchorMaterialMapper._map_residual(mat))

        # Backref & log
        _write_backref(mat, plx_mat)
        _log_create("Anchor", mat, plx_mat)
        return plx_mat

    @staticmethod
    def delete_material(
        g_i: Any,
        mat_or_plx: Union[ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor, Any]
    ) -> bool:
        """
        Delete the anchor material in PLAXIS. On success reset `mat.plx_id = None`.
        """
        mat_obj: Optional[Union[ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor]] = None
        plx_obj: Any = None

        if isinstance(mat_or_plx, (ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor)):
            mat_obj = mat_or_plx
            plx_obj = getattr(mat_obj, "plx_id", None)
        else:
            plx_obj = mat_or_plx

        if plx_obj is None:
            _log_delete("Anchor", mat_obj, None, ok=False)
            return False

        ok = _try_delete_with_gi(g_i, plx_obj)
        if ok and mat_obj is not None:
            _write_backref(mat_obj, None)
        _log_delete("Anchor", mat_obj, plx_obj, ok=ok)
        return ok

    # ########################### creation primitive ##########################
    @staticmethod
    def _create_raw_anchor(g_i: Any) -> Any:
        """
        Prefer Node-to-Node anchor material. Try a set of known constructor names.
        """
        candidates = ["n2nanchormat", "anchormat", "cablemat", "anchor_mat"]
        for fn_name in candidates:
            fn = getattr(g_i, fn_name, None)
            if callable(fn):
                obj = fn()
                return _normalize_created_handle(obj)
        raise RuntimeError("No anchor material constructor found on g_i (tried: n2nanchormat, anchormat, cablemat, anchor_mat).")

    # ########################### property setting ############################
    @staticmethod
    def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
        """Prefer per-key setproperties(k, v); fallback to setattr/setproperty; skip None."""
        if not props:
            return
        for k, v in props.items():
            if v is None:
                continue
            try:
                if hasattr(plx_obj, "setproperties"):
                    plx_obj.setproperties(k, v)
                    continue
            except Exception:
                pass
            try:
                setattr(plx_obj, k, v)
                continue
            except Exception:
                pass
            try:
                if hasattr(plx_obj, "setproperty"):
                    plx_obj.setproperty(k, v)
            except Exception:
                pass

    # ########################### mappings ###################################
    @staticmethod
    def _map_common(
            mat: Union[ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor]
        ) -> Dict[str, Any]:
        K = ANCHOR_KEYS
        type_label = mat.type.value if isinstance(mat.type, AnchorType) else str(mat.type)
        return {
            K["MaterialName"]:   getattr(mat, "name", "Anchor"),
            K["Identification"]: getattr(mat, "name", "Anchor"),
            K["Comments"]:       getattr(mat, "comment", ""),
            K["MaterialType"]:   type_label,
            K["EA"]:             getattr(mat, "EA", None),  # kN
        }

    @staticmethod
    def _map_elastoplastic(mat: ElastoplasticAnchor) -> Dict[str, Any]:
        K = ANCHOR_KEYS
        return {
            K["FMaxTension"]:     getattr(mat, "F_max_tens", None),
            K["FMaxCompression"]: getattr(mat, "F_max_comp", None),
        }

    @staticmethod
    def _map_residual(mat: ElastoPlasticResidualAnchor) -> Dict[str, Any]:
        K = ANCHOR_KEYS
        return {
            K["FResTension"]:     getattr(mat, "F_res_tens", None),
            K["FResCompression"]: getattr(mat, "F_res_comp", None),
        }
