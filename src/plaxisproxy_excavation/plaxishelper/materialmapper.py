# soil_material_plaxis_mapper.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from ..materials.soilmaterial import (
    BaseSoilMaterial, MCMaterial, MCCMaterial, HSSMaterial,
    SoilMaterialsType, RayleighInputMethod
)

class SoilMaterialMapper:
    """
    Static helper to create and configure PLAXIS soil materials from
    BaseSoilMaterial (MC/MCC/HSS) instances.

    - Uses g_i.soilmat() to create a new material.
    - Sets common properties (name, model, unit weights, elasticity, etc.).
    - Sets model-specific properties for MC/MCC/HSS.
    - Applies optional groups: Rayleigh damping, pore stress, groundwater,
      interface, initial K0, if the corresponding flags were set on the material.
    """

    # --------------------------- public entrypoint ---------------------------
    @staticmethod
    def create_material(g_i: Any, mat: BaseSoilMaterial) -> Any:
        """
        Create a PLAXIS soil material and push all properties from `mat`.
        Returns the created PLAXIS material handle, and writes it back to mat.plx_id.
        """
        # 1) Create raw PLAXIS material
        plx_mat = SoilMaterialMapper._create_raw_soil(g_i)

        # 2) Common properties (name/model/weights/elasticity)
        common_props = SoilMaterialMapper._map_common_props(mat)
        SoilMaterialMapper._set_many_props(plx_mat, common_props)

        # 3) Model-specific properties
        if isinstance(mat, MCMaterial):
            mc_props = SoilMaterialMapper._map_mc_props(mat)
            SoilMaterialMapper._set_many_props(plx_mat, mc_props)
        elif isinstance(mat, MCCMaterial):
            mcc_props = SoilMaterialMapper._map_mcc_props(mat)
            SoilMaterialMapper._set_many_props(plx_mat, mcc_props)
        elif isinstance(mat, HSSMaterial):
            hss_props = SoilMaterialMapper._map_hss_props(mat)
            SoilMaterialMapper._set_many_props(plx_mat, hss_props)
        else:
            raise TypeError(f"Unsupported soil material class: {type(mat).__name__}")

        # 4) Optional groups
        SoilMaterialMapper._apply_optional_groups(plx_mat, mat)

        # 5) Write plx_id back and return
        try:
            mat.plx_id = plx_mat  # 与你的 BaseMaterial 保持一贯写法
        except Exception:
            pass
        return plx_mat

    # --------------------------- creation primitive --------------------------
    @staticmethod
    def _create_raw_soil(g_i: Any) -> Any:
        """
        Create a raw soil material in PLAXIS. The common pattern is:
            plx_mat = g_i.soilmat()
        Some bindings return a tuple; normalize is the first.
        """
        created = g_i.soilmat()
        if isinstance(created, (list, tuple)) and created:
            return created[0]
        return created

    # --------------------------- property setting ----------------------------
    @staticmethod
    def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
        """
        Try setting properties in a robust way:
        1) If obj has 'setproperties', use a single call with key/value sequence.
        2) Else, fallback to setattr for each property name.
        """
        if not props:
            return
        if hasattr(plx_obj, "setproperties"):
            kv: List[Any] = []
            for k, v in props.items():
                kv.extend([k, v])
            try:
                plx_obj.setproperties(*kv)
                return
            except Exception:
                # fall back to setattr loop
                pass
        # fallback: setattr one by one
        for k, v in props.items():
            try:
                setattr(plx_obj, k, v)
            except Exception:
                # 最后再尝试 setproperty(key, val)（如有）
                if hasattr(plx_obj, "setproperty"):
                    try:
                        plx_obj.setproperty(k, v)
                        continue
                    except Exception:
                        pass
                # 这里不raise，尽量容错；由上层在需要时校验
                # print(f"[Warn] Failed to set property {k} on {plx_obj}")

    # --------------------------- common mappings -----------------------------
    @staticmethod
    def _map_common_props(mat: BaseSoilMaterial) -> Dict[str, Any]:
        """
        Map fields common to all soil materials.
        注意：以下键名是典型 PLAXIS 属性名；如与你版本不一致，请在此处统一调整。
        """
        model_name_map = {
            SoilMaterialsType.MC:  "Mohr-Coulomb",
            SoilMaterialsType.MCC: "Modified Cam Clay",
            SoilMaterialsType.HSS: "HSsmall",  # 或 "Hardening Soil small strain"
        }
        return {
            "MaterialName": getattr(mat, "name", "Soil"),     # 名称
            "SoilModel":    model_name_map.get(mat.type, ""), # 本构模型
            "gammaUnsat":   getattr(mat, "gamma", None),      # 非饱和重度
            "gammaSat":     getattr(mat, "gamma_sat", None),  # 饱和重度
            "nu":           getattr(mat, "nu", None),         # 泊松比
            # 备注与初始孔隙比（若你的 PLAXIS 版本支持相应键名）
            "Comments":     getattr(mat, "comment", ""),
            "eInit":        getattr(mat, "e_init", None),
        }

    # --------------------------- model-specific mappings ---------------------
    @staticmethod
    def _map_mc_props(mc: MCMaterial) -> Dict[str, Any]:
        """
        Mohr-Coulomb model properties.
        常见键名：Eref, cref, phi, psi, TensileStrength
        """
        return {
            "Eref":            mc.E_ref,
            "cref":            mc.c_ref,
            "phi":             mc.phi,
            "psi":             mc.psi,
            "TensileStrength": mc.tensile_strength,  # 若你的版本键名不同，可在此改
        }

    @staticmethod
    def _map_mcc_props(mcc: MCCMaterial) -> Dict[str, Any]:
        """
        Modified Cam Clay model properties.
        典型键名：kappa(κ), lambda(λ), M
        """
        return {
            "kappa":  mcc.kar,      # 你的类里 'kar' 表示 κ
            "lambda": mcc.lam,      # 你的类里 'lam' 表示 λ
            "M":      mcc.M_CSL,    # 临界状态线斜率
            # 一些版本还需要 eInit 等在 common 中已给
        }

    @staticmethod
    def _map_hss_props(hss: HSSMaterial) -> Dict[str, Any]:
        """
        Hardening Soil small-strain properties.
        常见键名：E50ref, Eoedref, Eurref, m, pref, G0ref, gamma07, c, phi, psi
        """
        return {
            "E50ref":  getattr(hss, "E", None),   # 你 BaseSoilMaterial.E 注释写的是 E50_ref
            "Eoedref": hss.E_oed,
            "Eurref":  hss.E_ur,
            "m":       hss.m,
            "pref":    hss.P_ref,
            "G0ref":   hss.G0,
            "gamma07": hss.gamma_07,
            "c":       hss.c,
            "phi":     hss.phi,
            "psi":     hss.psi,
        }

    # --------------------------- optional groups -----------------------------
    @staticmethod
    def _apply_optional_groups(plx_mat: Any, mat: BaseSoilMaterial) -> None:
        """
        Apply optional sets if the user enabled them on the material:
        - Rayleigh damping
        - Super pore stress
        - Under-ground water
        - Additional interface parameters
        - Additional initial parameters (K0)
        """
        # 1) Rayleigh damping (alpha/beta)
        if hasattr(mat, "_input_method") and mat._input_method == RayleighInputMethod.Direct:
            props = {}
            if hasattr(mat, "_alpha"): props["RayleighAlpha"] = mat._alpha
            if hasattr(mat, "_beta"):  props["RayleighBeta"]  = mat._beta
            SoilMaterialMapper._set_many_props(plx_mat, props)

        # 2) Super pore stress (示例键名，按你的版本调整)
        if getattr(mat, "_set_pore_stress", False):
            props = {
                "PoreStressType": getattr(mat, "_pore_stress_type", None),
                "VU":             getattr(mat, "_vu", None),            # 或 'SkemptonB'（若选择 B 法）
                "PoreValue":      getattr(mat, "_water_value", None),
            }
            SoilMaterialMapper._set_many_props(plx_mat, props)

        # 3) Under-ground water (示例键名，按你的版本调整)
        if getattr(mat, "_set_ug_water", False):
            props = {
                "GW_Type":          getattr(mat, "_Gw_type", None),
                "SWCC_Method":      getattr(mat, "_SWCC_method", None),
                "SoilPosi":         getattr(mat, "_soil_posi", None),
                "SoilFine":         getattr(mat, "_soil_fine", None),
                "GW_Defaults":      getattr(mat, "_Gw_defaults", None),
                "Infiltration":     getattr(mat, "_infiltration", None),
                "DefaultMethod":    getattr(mat, "_default_method", None),
                "kx":               getattr(mat, "_kx", None),
                "ky":               getattr(mat, "_ky", None),
                "kz":               getattr(mat, "_kz", None),
                "GW_PsiUnsat":      getattr(mat, "_Gw_Psiunsat", None),
            }
            SoilMaterialMapper._set_many_props(plx_mat, props)

        # 4) Additional interface (示例键名，按你的版本调整)
        if getattr(mat, "_set_additional_interface", False):
            props = {
                "InterfaceStiffnessDef": getattr(mat, "_stiffness_define", None),
                "InterfaceStrengthDef":  getattr(mat, "_strengthen_define", None),
                "Kref_n":                getattr(mat, "_k_n", None),
                "Kref_s":                getattr(mat, "_k_s", None),
                "Rinter":                getattr(mat, "_R_inter", None),
                "GapClosure":            getattr(mat, "_gap_closure", None),
                "CrossPermeability":     getattr(mat, "_cross_permeability", None),
                "DrainageCond1":         getattr(mat, "_drainage_conduct1", None),
                "DrainageCond2":         getattr(mat, "_drainage_conduct2", None),
            }
            SoilMaterialMapper._set_many_props(plx_mat, props)

        # 5) Additional initial K0 (示例键名，按你的版本调整)
        if getattr(mat, "_set_additional_initial", False):
            props = {
                "K0_Definition": getattr(mat, "_K_0_define", None),
                "K0x":           getattr(mat, "_K_0_x", None),
                "K0y":           getattr(mat, "_K_0_y", None),
            }
            SoilMaterialMapper._set_many_props(plx_mat, props)
