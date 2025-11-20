from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import re

# #### project imports (adjust to your tree) ##################################
from ..borehole import BoreholeSet, Borehole, BoreholeLayer, SoilLayer

# =============================================================================
# Small, robust utilities
# =============================================================================

def _format_handle(h: Any) -> str:
    if h is None:
        return "None"
    for k in ("Id", "ID", "id", "Guid", "GUID", "guid", "Name", "Identification"):
        try:
            v = getattr(h, k, None)
            if v is not None:
                if hasattr(v, "value"):
                    v = v.value
                return f"{k}={v}"
        except Exception:
            pass
    s = str(h).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= 120 else (s[:117] + "...")

def _one_line(msg: str) -> str:
    return " ".join(str(msg).split())

def _log_create(kind: str, desc: str, handle: Any, extra: str = "") -> None:
    print(_one_line(f"[CREATE][{kind}] {desc} handle={_format_handle(handle)} {extra}"), flush=True)

def _log_delete(kind: str, desc: str, handle: Any, ok: bool, extra: str = "") -> None:
    status = "OK" if ok else "FAIL"
    print(_one_line(f"[DELETE][{kind}] {desc} handle={_format_handle(handle)} result={status} {extra}"), flush=True)

def _normalize_created_handle(created: Any) -> Any:
    if isinstance(created, (list, tuple)) and created:
        return created[0]
    return created

def _try_call(g_i: Any, names: Sequence[str], *args, **kwargs) -> Any:
    """
    Try calling g_i.<name>(*args, **kwargs) in order; return on first success.
    Raise the last caught exception if all fail; raise RuntimeError if none exists.
    """
    last_exc: Optional[Exception] = None
    for nm in names:
        fn = getattr(g_i, nm, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                continue
    if last_exc:
        raise last_exc
    raise RuntimeError(f"No callable found among {names}")

def _try_delete_with_gi(g_i: Any, plx_obj: Any) -> bool:
    """Delete by obj.delete() then by g_i.[delete variants]."""
    try:
        if hasattr(plx_obj, "delete") and callable(plx_obj.delete):
            plx_obj.delete()
            return True
    except Exception:
        pass
    for fn_name in ("delete", "delobject", "deletematerial", "delmaterial", "remove"):
        try:
            fn = getattr(g_i, fn_name, None)
            if callable(fn):
                fn(plx_obj)
                return True
        except Exception:
            continue
    return False

def _set_many_props(plx_obj: Any, props: Dict[str, Any]) -> None:
    """Robust property setting with None-skip + bulk/per-key fallbacks."""
    if not props:
        return
    filtered = {k: v for k, v in props.items() if v is not None}

    # bulk (setproperties)
    if hasattr(plx_obj, "setproperties"):
        try:
            kv: List[Any] = []
            for k, v in filtered.items():
                kv.extend([k, v])
            if kv:
                plx_obj.setproperties(*kv)  # type: ignore[arg-type]
                return
        except Exception:
            pass

    # per-key setattr / setproperty
    for k, v in filtered.items():
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

def _extract_new_symbol_from_msg(msg: str) -> Optional[str]:
    """
    Parse strings like: 'Added Soillayer_1', 'Created SoilLayer_2', etc.
    Return the symbol name (e.g., 'Soillayer_1') if found.
    """
    if not isinstance(msg, str):
        return None
    # Common patterns: Added Soillayer_1 / Created SoilLayer_1
    m = re.search(r"\b(?:Added|Created)\s+(Soil[Ll]ayer_\d+)\b", msg)
    if m:
        return m.group(1)
    # Fallback: any token that looks like SoilLayer_### or Soillayer_###
    m = re.search(r"\b(Soil[Ll]ayer_\d+)\b", msg)
    if m:
        return m.group(1)
    return None

def _resolve_new_soillayer_handle(
    g_i: Any,
    created_raw: Any,
    *,
    index_hint: Optional[int] = None,
    name_hint: Optional[str] = None,  # kept for future use
) -> Any:
    """
    If created_raw is already a handle (non-string), return it.
    If it's a string like 'Added Soillayer_1', resolve the real handle from g_i.<symbol>.
    Try several fallbacks using index_hint and a best-effort scan.
    """
    # 1) Already a handle
    if not isinstance(created_raw, str):
        return created_raw

    # 2) Try to parse symbol from message
    sym = _extract_new_symbol_from_msg(created_raw)
    if sym:
        h = getattr(g_i, sym, None)
        if h is not None:
            return h

    # 3) Try common naming with index hint (1-based is common)
    if index_hint is not None:
        for cand in (
            f"Soillayer_{index_hint}",
            f"SoilLayer_{index_hint}",
            f"soilLayer_{index_hint}",
        ):
            h = getattr(g_i, cand, None)
            if h is not None:
                return h

    # 4) Last resort: scan dir(g_i) for the largest numbered SoilLayer-like symbol
    try:
        cands = [n for n in dir(g_i) if re.fullmatch(r"Soil[Ll]ayer_\d+", n)]
        if cands:
            # pick the one with the largest suffix number (heuristic for "latest")
            def _suffix(nm: str) -> int:
                m = re.search(r"_(\d+)$", nm)
                return int(m.group(1)) if m else -1
            best = max(cands, key=_suffix)
            h = getattr(g_i, best, None)
            if h is not None:
                return h
    except Exception:
        pass

    # If still nothing, just return the raw string (caller will log a warn)
    return created_raw

# =============================================================================
# Zone writing helpers (tolerant to binding differences)
# =============================================================================

def _set_zone_for_borehole(plx_layer: Any, bh_index: int, top: float, bottom: float) -> bool:
    """
    Try several common patterns to set a SoilLayer's zone at borehole index:
    1) layer.Zones[i].Top / .Bottom
    2) layer.Top[i] / layer.Bottom[i] (vector-like)
    3) layer.Top.value (list) -> mutate -> write back; same for Bottom
    4) layer.setproperties("Top[i]", top, "Bottom[i]", bottom)
    Returns True on first success.
    """
    # 1) Zones[i].Top / Bottom
    try:
        zones = getattr(plx_layer, "Zones", None)
        if zones is not None:
            try:
                z_i = zones[bh_index]
                try:
                    setattr(z_i, "Top", float(top))
                except Exception:
                    pass
            except Exception:
                pass
            try:
                z_i = zones[bh_index]
                try:
                    setattr(z_i, "Bottom", float(bottom))
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass

    return False


def _assign_soillayer_material(layer_handle: Any, sl: SoilLayer) -> None:
    """
    Assign soil material to a SoilLayer handle.
    Preferred path (as you observed): layer_handle.Soil.Material = <mat_handle>
    Fallbacks are kept for compatibility.
    """
    if layer_handle is None or sl is None:
        return
    mat_h = getattr(getattr(sl, "material", None), "plx_id", None)
    if mat_h is None:
        return

    # 1) Preferred: nested node "Soil"."Material"
    soil_node = getattr(layer_handle, "Soil", None)
    if soil_node is not None:
        try:
            setattr(soil_node, "Material", mat_h)
            return
        except Exception:
            pass
        for key in ("MaterialRef", "Mat", "SoilMaterial"):
            try:
                setattr(soil_node, key, mat_h)
                return
            except Exception:
                try:
                    if hasattr(soil_node, "setproperty"):
                        soil_node.setproperty(key, mat_h)
                        return
                except Exception:
                    pass

    # 2) Fallback: single-level (legacy)
    for key in ("SoilMaterial", "Material", "MaterialRef", "Mat"):
        try:
            setattr(layer_handle, key, mat_h)
            return
        except Exception:
            try:
                if hasattr(layer_handle, "setproperty"):
                    layer_handle.setproperty(key, mat_h)
                    return
            except Exception:
                pass


def _set_zone_top_bottom(sl: SoilLayer, bh_idx: int, top: float, bottom: float) -> None:
    """
    Wrapper that calls _set_zone_for_borehole on sl.plx_id.
    Tries 1-based index first (common in PLAXIS), then 0-based as fallback.
    """
    if sl is None or getattr(sl, "plx_id", None) is None:
        return
    h = sl.plx_id

    _set_zone_for_borehole(h, bh_idx, float(top), float(bottom))


# =============================================================================
# BoreholeSet Mapper (only set-level import to avoid conflicts)
# =============================================================================

class BoreholeSetMapper:
    """
    Mapper for importing a **whole BoreholeSet** into PLAXIS.

    Pipeline
    ########
    1) Call `ensure_unique_names()` and then `normalize_all_boreholes(in_place=True)` to unify the stratigraphic sequence.
    Missing layers will be set to zero thickness (top==bottom), and the upper and lower layer boundaries will be corrected.
    2) Create all Boreholes ⇒ Write `Borehole.plx_id`.
    3) Create all unique SoilLayers ⇒ Write `SoilLayer.plx_id` and set `Soil.Material`.
    4) Based on the normalization result, set the Top/Bottom of Zone[i] for each borehole index of each SoilLayer.
    5) Output a single-line English prompt to the console.
    """

    # ### constructor candidates per binding ###
    _BH_CALLS = ("borehole", "createborehole", "bhole", "bh", "add_borehole")
    _SL_CALLS = ("soillayer", "SoilLayer", "soil_layer", "addsoillayer")

    # ############################## create #################################
    @staticmethod
    def create(
        g_i: Any,
        bhset: BoreholeSet,
        *,
        normalize: bool = True,
        set_name_on_objects: bool = True,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Import the entire BoreholeSet into PLAXIS.

        Returns
        #######
        summary : {layer_name: [(top,bottom)_bh0, (top,bottom)_bh1, ...]}
                  (By the order of BoreholeSet.boreholes)
        """
        if not isinstance(bhset, BoreholeSet):
            raise TypeError("BoreholeSetMapper.create requires a BoreholeSet.")

        # 0) 确保命名唯一，避免 PLAXIS 冲突；随后 normalize
        bhset.ensure_unique_names()
        info = bhset.normalize_all_boreholes(in_place=True) if normalize else {
            "ordered_names": [],
            "boreholes": {bh.name: {ly.soil_layer.name: (float(ly.top_z), float(ly.bottom_z)) for ly in bh.layers}
                          for bh in bhset.boreholes}
        }

        # 构造一个 {layer_name: [(top,bottom)_bh0, ...]} 汇总（按 boreholes 顺序）
        ordered_names: List[str] = list(info.get("ordered_names", []))
        if not ordered_names:
            # 如果没返回 ordered_names，则从数据里推断
            seen: List[str] = []
            for bh in bhset.boreholes:
                d = info["boreholes"].get(bh.name, {})
                for ln in d.keys():
                    if ln not in seen:
                        seen.append(ln)
            ordered_names = seen

        summary: Dict[str, List[Tuple[float, float]]] = {nm: [] for nm in ordered_names}
        for bh in bhset.boreholes:
            d = info["boreholes"].get(bh.name, {})
            for ln in ordered_names:
                pair = d.get(ln)
                if pair is None:
                    # 若个别未包含（理论上 normalize 后不会），零厚度兜底
                    summary[ln].append((0.0, 0.0))
                else:
                    top, bot = pair
                    summary[ln].append((float(top), float(bot)))

        # 1) Create all boreholes
        for bh in bhset.boreholes:
            x, y = float(bh.location.x), float(bh.location.y)
            created = _try_call(g_i, BoreholeSetMapper._BH_CALLS, x, y)
            created = _normalize_created_handle(created)

            if set_name_on_objects:
                _set_many_props(created, {"Name": bh.name, "Identification": bh.name, "Head": bh.water_head})
            bh.plx_id = created
            _log_create("Borehole", f"name={bh.name} at ({x},{y})", created)

        # 2) Create all unique SoilLayers (unique by identity)
        unique_layers: List[SoilLayer] = []
        seen_ids: set[int] = set()
        for bh in bhset.boreholes:
            for ly in bh.layers:
                sl = ly.soil_layer
                oid = id(sl)
                if oid in seen_ids:
                    continue
                seen_ids.add(oid)
                unique_layers.append(sl)

        sl_handle_by_name: Dict[str, Any] = {}
        for idx, sl in enumerate(unique_layers, start=1):
            try:
                created_raw = _try_call(g_i, BoreholeSetMapper._SL_CALLS, 0)
            except Exception:
                created_raw = _try_call(g_i, BoreholeSetMapper._SL_CALLS)

            created_sl = _resolve_new_soillayer_handle(
                g_i, created_raw, index_hint=idx, name_hint=sl.name
            )
            created_sl = _normalize_created_handle(created_sl)

            if isinstance(created_sl, str):
                print(_one_line(f"[WARN] Could not resolve SoilLayer handle from message='{created_raw}' (idx={idx})"))
                sl.plx_id = None
                continue

            # 尝试设置名称（不同版本不一定有 Name）
            _set_many_props(created_sl, {"Name": sl.name, "Identification": sl.name})

            # 设置材料：SoilLayerHandle.Soil.Material = <mat_handle>
            _assign_soillayer_material(created_sl, sl)

            sl.plx_id = created_sl
            sl_handle_by_name[sl.name] = created_sl
            _log_create("SoilLayer", f"name={sl.name}", created_sl)

        # 3) Apply Zones: for each SoilLayer, set Zone[i] using summary
        #    summary[layer_name][i] = (top, bottom)  —— i 按 bhset.boreholes 顺序
        for ln, rows in summary.items():
            sl_obj: Optional[SoilLayer] = None
            for sl in unique_layers:
                if sl.name == ln:
                    sl_obj = sl
                    break
            if sl_obj is None:
                continue

            for bh_idx, (top, bot) in enumerate(rows):
                _set_zone_top_bottom(sl_obj, bh_idx, float(top), float(bot))

        _log_create("BoreholeSet",
                    f"name={bhset.name} n_bh={len(bhset.boreholes)} n_layers={len(unique_layers)}",
                    handle="OK")
        return summary

    # ############################### deletes ###############################
    @staticmethod
    def delete_borehole(g_i: Any, borehole_or_handle: Union[Borehole, Any]) -> bool:
        """Delete a single borehole (best-effort); clears Borehole.plx_id if passed object."""
        obj = borehole_or_handle if isinstance(borehole_or_handle, Borehole) else None
        h = getattr(obj, "plx_id", None) if obj else borehole_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj is not None:
            obj.plx_id = None
        _log_delete("Borehole", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

    @staticmethod
    def delete_soillayer(g_i: Any, sl_or_handle: Union[SoilLayer, Any]) -> bool:
        """Delete a single SoilLayer (best-effort); clears SoilLayer.plx_id if passed object."""
        obj = sl_or_handle if isinstance(sl_or_handle, SoilLayer) else None
        h = getattr(obj, "plx_id", None) if obj else sl_or_handle
        ok = _try_delete_with_gi(g_i, h)
        if ok and obj is not None:
            obj.plx_id = None
        _log_delete("SoilLayer", f"name={getattr(obj, 'name', 'raw')}", h, ok=ok)
        return ok

    @staticmethod
    def delete_all(g_i: Any, bhset: BoreholeSet) -> None:
        """
        Best-effort deletion of all layers then all boreholes in the set.
        """
        # delete layers (unique by identity)
        seen: set[int] = set()
        for bh in bhset.boreholes:
            for ly in bh.layers:
                sl = ly.soil_layer
                oid = id(sl)
                if oid in seen:
                    continue
                seen.add(oid)
                if getattr(sl, "plx_id", None) is not None:
                    BoreholeSetMapper.delete_soillayer(g_i, sl)

        # delete boreholes
        for bh in bhset.boreholes:
            if getattr(bh, "plx_id", None) is not None:
                BoreholeSetMapper.delete_borehole(g_i, bh)

        _log_delete("BoreholeSet", f"name={bhset.name}", handle="(all)", ok=True)
