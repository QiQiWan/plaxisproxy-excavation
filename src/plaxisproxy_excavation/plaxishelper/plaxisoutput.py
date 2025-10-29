from __future__ import annotations

from contextlib import contextmanager
from typing import Any, List, Optional, Tuple, Union
from enum import Enum
try:
    from plxscripting.easy import new_server
except:
    from plxscripting.server import new_server

# ---------------------------------------------------------------------
# PLAXIS scripting API (fallback stub so static analysis won't break)
# ---------------------------------------------------------------------
try:
    from plxscripting.easy import new_server  # type: ignore
except Exception:  # pragma: no cover
    def new_server(host: str, port: int, password: str) -> Tuple[Any, Any]:
        class _DummyServer:
            def close(self) -> None: ...
        class _DummyG:
            ResultTypes = type("RT", (), {})()
        return _DummyServer(), _DummyG()  # type: ignore[misc]

# ---------------------------------------------------------------------
# Use your uploaded ResultTypes enums resolver
# DO NOT re-define any result leaves here.
# ---------------------------------------------------------------------
# Adjust the import to your project structure if needed.
from .resulttypes import resolve_resulttype  # type: ignore

__all__ = ["PlaxisOutput"]

class PlaxisOutput:
    """
    Stateless-ish facade for PLAXIS Output:
    - Bind to a phase via Input.g_i.view(phase_id) to get a port
    - Create s_o/g_o via new_server(host, port, password)
    - Fetch results for the *currently bound* phase using g_o.getresults(...)
    """
    def __init__(self, host: str, password: str) -> None:
        self.host = host
        self.password = password
        self.port: Optional[int] = None
        self.s_o = None
        self.g_o = None
        self._current_phase_id: Optional[int] = None

    @property
    def is_connected(self) -> bool:
        return bool(self.g_o)

    def close(self) -> None:
        try:
            if self.s_o and hasattr(self.s_o, "close"):
                self.s_o.close()
        except Exception:
            pass
        finally:
            self.s_o = None
            self.g_o = None
            self.port = None
            self._current_phase_id = None

    # ---------- resolvers ----------

    def _resolve_phase_id(self, phase: Any) -> int:
        """Accept Phase-like (with .plx_id) or numeric id."""
        if hasattr(phase, "plx_id"):
            return getattr(phase, "plx_id")
        raise TypeError("phase must be a Phase-like object with plx_id or an integer id")

    def _resolve_structure_id(self, structure: Any) -> int:
        """Accept structure object (with .plx_id) or numeric id."""
        if hasattr(structure, "plx_id"):
            return getattr(structure, "plx_id")
        raise TypeError("structure must be an object with plx_id or an integer id.")

    # ---------- connections ----------

    def connect_via_input(self, g_i: Any, phase: Any) -> "PlaxisOutput":
        """
        Use Input.g_i.view(phase_id) -> port -> new_server(host, port, password).
        Stores s_o/g_o and the bound phase id.
        """
        ph_id = self._resolve_phase_id(phase)
        port = g_i.view(ph_id)
        s_o, g_o = new_server(self.host, port, password=self.password)
        self.s_o, self.g_o, self.port = s_o, g_o, port
        self._current_phase_id = ph_id
        return self

    def set_default_phase(self, g_i: Any, phase: Any) -> None:
        """Rebind Output to a different phase (recreate s_o/g_o)."""
        self.close()
        self.connect_via_input(g_i, phase)

    # ---------- resulttypes resolution ----------

    def _resolve_leaf(self, leaf: Enum) -> object:
        """
        Resolve a result type into g_o.ResultTypes.* member.

        Accepts:
        - Enum member from our resulttypes (e.g., Plate.Ux whose value is 'Ux')
        - 'Domain.Leaf' string (e.g., 'Plate.Ux')
        - ('Domain','Leaf') / ['Domain','Leaf']
        - Already a g_o.ResultTypes member (proxy) -> returned as-is

        Case handling:
        - Tries exact match first, then TitleCase / UPPER / lower fallbacks per segment.
        """
        if not self.g_o:
            raise RuntimeError("Output not connected (g_o is None).")

        rt_root = getattr(self.g_o, "ResultTypes", None)
        if rt_root is None:
            raise RuntimeError("g_o.ResultTypes is not available on the current Output session.")

        # 0) If it's already a proxy/member under ResultTypes, just return it.
        #    We do a best-effort heuristic: try using it directly later; here just short-circuit.
        #    (If callers pass a proxy, they know what they are doing.)
        try:
            # Simple heuristic: proxies from plxscripting typically have a repr that contains 'ProxyObject'
            if "ProxyObject" in repr(type(leaf)):
                return leaf
        except Exception:
            pass

        # 1) Build a path list: ['Domain','Leaf']
        parts: list[str] = []

        if isinstance(leaf, Enum):
            # Enum from our resulttypes: class name is the domain, value is the leaf or path.
            domain = leaf.__class__.__name__
            val = leaf.value
            if isinstance(val, (tuple, list)):
                parts = [str(p) for p in val]  # e.g. ('Plate','Ux')
            else:
                s = str(val)
                if "." in s:
                    parts = s.split(".")
                else:
                    parts = [domain, s]        # e.g. Plate + 'Ux' -> Plate.Ux
        elif isinstance(leaf, (tuple, list)) and len(leaf) >= 1:
            parts = [str(p) for p in leaf]     # e.g. ('Plate','Ux')
        elif isinstance(leaf, str):
            s = leaf.strip()
            if "." in s:
                parts = s.split(".")
            else:
                # 单段字符串不够确定（比如 'Ux'），直接提示更规范用法
                raise ValueError(
                    f"Ambiguous leaf '{leaf}'. Use 'Plate.Ux' or the enum member resulttypes.Plate.Ux."
                )
        else:
            # Unknown type: assume it's already a valid member and return
            return leaf

        # 2) Walk the tree: g_o.ResultTypes.<Domain>.<Leaf>
        node = rt_root

        def _get_with_fallback(obj: object, name: str) -> object:
            # Try exact first
            if hasattr(obj, name):
                return getattr(obj, name)
            # Try TitleCase (Ux), UPPER (UX), lower (ux)
            title = name[:1].upper() + name[1:]
            if hasattr(obj, title):
                return getattr(obj, title)
            upper = name.upper()
            if hasattr(obj, upper):
                return getattr(obj, upper)
            lower = name.lower()
            if hasattr(obj, lower):
                return getattr(obj, lower)
            # Give a helpful error
            raise AttributeError(f"ResultTypes path segment '{name}' not found under {obj!r}")

        try:
            for seg in parts:
                node = _get_with_fallback(node, str(seg))
        except AttributeError as e:
            # 给出更清晰的信息：告诉用户当前支持的用法
            raise AttributeError(
                f"Invalid ResultTypes path {'/'.join(parts)}. "
                f"Make sure you are using resulttypes enums (e.g., Plate.Ux) or 'Plate.Ux' string."
            ) from e

        return node


    # ---------- results ----------

    def get_results(self, structure: Any, leaf: Any, *, smoothing: bool = False
                    ) -> Union[list, float, str]:
        """
        Fetch results for the CURRENT connected phase.
        Tries 'nodes' first, then 'stresspoint'.
        Returns:
          - list[float] if iterable
          - float if scalar-number
          - str for text statuses
        """
        if not self.g_o:
            raise RuntimeError("Output not connected. Call connect_via_input/set_default_phase first.")

        struct_id = self._resolve_structure_id(structure)
        leaf_member = self._resolve_leaf(leaf)

        def _try(loc: str):
            return self.g_o.getresults(struct_id, leaf_member, loc, bool(smoothing))

        # prefer nodes, fallback to stresspoint
        try:
            data = _try("node")
        except Exception:
            data = _try("stresspoint")

        # normalize return
        if isinstance(data, str):
            return data
        try:
            iter(data)  # iterable?
            return [float(x) for x in data]
        except TypeError:
            try:
                return float(data)  # scalar numeric
            except Exception:
                return data