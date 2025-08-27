# load.py  —— 完整可粘贴版本
from __future__ import annotations
from enum import Enum, auto
from typing import Optional, Dict, Tuple, List, Any, Type
from ..core.plaxisobject import PlaxisObject
from ..geometry import Point, Line3D, Polygon3D

__all__ = [
    "LoadStage",
    "DistributionType",
    "SignalType",
    "LoadMultiplier",
    "PointLoad",
    "LineLoad",
    "SurfaceLoad",
    "DynPointLoad",
    "DynLineLoad",
    "DynSurfaceLoad",
    "UniformSurfaceLoad",
    "LinearSurfaceLoad",
    "XAlignedIncrementSurfaceLoad",
    "YAlignedIncrementSurfaceLoad",
    "ZAlignedIncrementSurfaceLoad",
    "VectorAlignedIncrementSurfaceLoad",
    "FreeIncrementSurfaceLoad",
    "PerpendicularSurfaceLoad",
    "DynUniformSurfaceLoad",
    "DynLinearSurfaceLoad",
    "DynXAlignedIncrementSurfaceLoad",
    "DynYAlignedIncrementSurfaceLoad",
    "DynZAlignedIncrementSurfaceLoad",
    "DynVectorAlignedIncrementSurfaceLoad",
    "DynFreeIncrementSurfaceLoad",
    "DynPerpendicularSurfaceLoad",
]

# =============================================================================
#  Enumerations
# =============================================================================
class LoadStage(Enum):
    """Static (default) or Dynamic (time-dependent) load phase."""
    STATIC = "Static"
    DYNAMIC = "Dynamic"


class DistributionType(Enum):
    """Supported distribution options (mirrors Plaxis drop-down lists)."""
    # —— generic (Point / Line / Surface) ——————————————
    UNIFORM = auto()          # constant over entity
    LINEAR = auto()           # start → end linear variation (Line / Surface)

    # —— Surface-only extended options ————————————————
    X_ALIGNED_INC = auto()
    Y_ALIGNED_INC = auto()
    Z_ALIGNED_INC = auto()
    VECTOR_ALIGNED_INC = auto()
    FREE_INCREMENT = auto()
    PERPENDICULAR = auto()
    PERPENDICULAR_VERT_INC = auto()


# =============================================================================
#  Multiplier Types
# =============================================================================
class SignalType(Enum):
    """Supported signal types for multipliers."""
    HARMONIC = "Harmonic"
    TABLE = "Table"


class LoadMultiplier(PlaxisObject):
    """Represents a load multiplier to be applied to dynamic loads."""
    def __init__(
        self,
        name: str,
        comment: str,
        signal_type: SignalType,
        amplitude: float = 1.0,
        phase: float = 0.0,
        frequency: float = 0.0,
        table_data: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(name, comment)
        self._signal_type = signal_type
        self._amplitude: Optional[float] = None
        self._phase: Optional[float] = None
        self._frequency: Optional[float] = None
        self._table_data: Optional[List[Tuple[float, float]]] = None

        if signal_type == SignalType.HARMONIC:
            if not isinstance(amplitude, (int, float)) or amplitude < 0:
                raise ValueError("Harmonic signal requires a non-negative 'amplitude'.")
            if not isinstance(phase, (int, float)):
                raise ValueError("Harmonic signal requires a numeric 'phase'.")
            if not isinstance(frequency, (int, float)) or frequency < 0:
                raise ValueError("Harmonic signal requires a non-negative 'frequency'.")
            self._amplitude = float(amplitude)
            self._phase = float(phase)
            self._frequency = float(frequency)

        elif signal_type == SignalType.TABLE:
            if table_data is None or not isinstance(table_data, list):
                raise ValueError("Table signal requires 'table_data' list.")
            last_t: Optional[float] = None
            clean: List[Tuple[float, float]] = []
            for idx, point in enumerate(table_data):
                if not (isinstance(point, (tuple, list)) and len(point) == 2):
                    raise ValueError(f"Table data at index {idx} must be a (time, value) pair.")
                t, v = point
                if not isinstance(t, (int, float)) or not isinstance(v, (int, float)):
                    raise ValueError(f"Table data values at index {idx} must be numeric.")
                t_f = float(t)
                v_f = float(v)
                if last_t is not None and t_f < last_t:
                    raise ValueError(f"Table data must be ordered by time. Error at index {idx}.")
                last_t = t_f
                clean.append((t_f, v_f))
            self._table_data = clean
        else:
            raise NotImplementedError(f"Signal type '{signal_type.value}' is not implemented.")

    @property
    def signal_type(self) -> SignalType:
        return self._signal_type

    @property
    def amplitude(self) -> Optional[float]:
        return self._amplitude

    @property
    def phase(self) -> Optional[float]:
        return self._phase

    @property
    def frequency(self) -> Optional[float]:
        return self._frequency

    @property
    def table_data(self) -> Optional[List[Tuple[float, float]]]:
        return self._table_data

    def __repr__(self) -> str:
        base_repr = f"<plx.structures.LoadMultiplier(name='{self.name}', signal='{self._signal_type.value}')"
        if self._signal_type == SignalType.HARMONIC:
            return f"{base_repr}, amplitude={self._amplitude:.3f}, phase={self._phase:.1f}°, frequency={self._frequency:.3f} Hz>"
        elif self._signal_type == SignalType.TABLE:
            if self._table_data and len(self._table_data) > 4:
                first, second = self._table_data[0], self._table_data[1]
                penultimate, last = self._table_data[-2], self._table_data[-1]
                data_str = f"[{first}, {second}, ..., {penultimate}, {last}]"
            else:
                data_str = str(self._table_data)
            return f"{base_repr}, table_data={data_str}>"
        return f"{base_repr}>"


# =============================================================================
#  Base class (forces + optional moments)
# =============================================================================
class _BaseLoad(PlaxisObject):
    """Shared attributes for load entities (no direct instantiation)."""
    def __init__(
        self,
        name: str,
        comment: str,
        stage: LoadStage,
        distribution: DistributionType,
        Fx: float = 0.0,
        Fy: float = 0.0,
        Fz: float = 0.0,
        Mx: float = 0.0,
        My: float = 0.0,
        Mz: float = 0.0,
        Fx_end: float = 0.0,
        Fy_end: float = 0.0,
        Fz_end: float = 0.0,
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None
    ) -> None:
        super().__init__(name, comment)
        if not isinstance(stage, LoadStage):
            raise TypeError("stage must be a LoadStage value.")
        if not isinstance(distribution, DistributionType):
            raise TypeError("distribution must be a DistributionType value.")

        for val in (Fx, Fy, Fz, Mx, My, Mz, Fx_end, Fy_end, Fz_end):
            if not isinstance(val, (int, float)):
                raise TypeError("Force and moment components must be numeric.")

        if gradients is not None:
            if not isinstance(gradients, dict):
                raise TypeError("'gradients' must be a dict of str->float.")
            for key, value in gradients.items():
                if not isinstance(key, str) or not isinstance(value, (int, float)):
                    raise ValueError("Gradient entries must have string keys and numeric values.")

        if ref_point is not None:
            if not (isinstance(ref_point, (tuple, list)) and len(ref_point) == 3
                    and all(isinstance(c, (int, float)) for c in ref_point)):
                raise ValueError("ref_point must be a tuple of three numbers.")

        self._stage: LoadStage = stage
        self._distribution: DistributionType = distribution
        self._Fx, self._Fy, self._Fz = float(Fx), float(Fy), float(Fz)
        self._Mx, self._My, self._Mz = float(Mx), float(My), float(Mz)
        self._Fx_end, self._Fy_end, self._Fz_end = float(Fx_end), float(Fy_end), float(Fz_end)
        self._grad: Dict[str, float] = gradients or {}
        self._ref_point: Tuple[float, float, float] = ref_point if ref_point else (0.0, 0.0, 0.0)

    # Pretty helpers
    def _force_str(self) -> str:
        return (
            f"F=({self._Fx:+.2f}, {self._Fy:+.2f}, {self._Fz:+.2f}); "
            f"M=({self._Mx:+.2f}, {self._My:+.2f}, {self._Mz:+.2f})"
        )

    def _moment_str(self) -> str:
        return f"M=({self._Mx:+.2f},{self._My:+.2f},{self._Mz:+.2f})"

    # Accessors
    @property
    def stage(self):
        return self._stage

    @property
    def distribution(self):
        return self._distribution

    @property
    def Fx(self) -> float:
        return self._Fx

    @property
    def Fy(self) -> float:
        return self._Fy

    @property
    def Fz(self) -> float:
        return self._Fz

    @property
    def Mx(self) -> float:
        return self._Mx

    @property
    def My(self) -> float:
        return self._My

    @property
    def Mz(self) -> float:
        return self._Mz

    @property
    def Fx_end(self) -> float:
        return self._Fx_end

    @property
    def Fy_end(self) -> float:
        return self._Fy_end

    @property
    def Fz_end(self) -> float:
        return self._Fz_end

    # Surface Gradients
    @property
    def grad(self) -> Dict[str, float]:
        return self._grad

    @property
    def ref_point(self) -> Tuple[float, float, float]:
        return self._ref_point


# =============================================================================
#  Dynamic base (normalizes multipliers and provides mult property)
# =============================================================================
class _DynBaseLoad(_BaseLoad):
    """Shared logic for dynamic loads: multiplier normalization & access."""
    _MUL_KEYS: Tuple[str, ...] = tuple()

    def __init__(
        self,
        *args,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
        **kwargs
    ) -> None:
        # 强制动态阶段
        kwargs["stage"] = LoadStage.DYNAMIC
        super().__init__(*args, **kwargs)
        # 归一化 multiplier
        self._mult: Dict[str, LoadMultiplier] = {}
        self._init_multiplier(multiplier)

    @property
    def mult(self) -> Dict[str, LoadMultiplier]:
        return self._mult

    @mult.setter
    def mult(self, value: Dict[str, LoadMultiplier]) -> None:
        self._init_multiplier(value)

    def _allowed_mul_keys(self) -> Tuple[str, ...]:
        return getattr(self, "_MUL_KEYS", tuple())

    def _default_multiplier_key(self) -> str:
        keys = self._allowed_mul_keys()
        if keys:
            return keys[-1]
        raise RuntimeError("No multiplier keys defined on this dynamic load.")

    def _init_multiplier(self, multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier]) -> None:
        self._mult = {}
        if multiplier is None:
            return
        allowed = set(self._allowed_mul_keys())
        if isinstance(multiplier, LoadMultiplier):
            self._mult[self._default_multiplier_key()] = multiplier
            return
        if not isinstance(multiplier, dict):
            raise TypeError("multiplier must be a LoadMultiplier or a dict[str, LoadMultiplier].")
        for k, v in multiplier.items():
            if k not in allowed:
                raise ValueError(f"Invalid multiplier key '{k}'. Allowed: {sorted(allowed)}")
            if not isinstance(v, LoadMultiplier):
                raise TypeError(f"Multiplier for '{k}' must be a LoadMultiplier.")
            self._mult[k] = v

    def _mult_str(self) -> str:
        if not self._mult:
            return ""
        items = ", ".join(f"{k}: {v.name}" for k, v in self._mult.items())
        return f" mult={{ {items} }}"

# =============================================================================
#  Point Load – supports forces *and* moments
# =============================================================================
class PointLoad(_BaseLoad):
    """Concentrated node load (Fx/Fy/Fz + optional Mx/My/Mz)."""
    def __init__(
        self,
        name: str,
        comment: str,
        point: Point,
        stage: LoadStage = LoadStage.STATIC,
        distribution: DistributionType = DistributionType.UNIFORM,
        Fx: float = 0.0,
        Fy: float = 0.0,
        Fz: float = 0.0,
        Mx: float = 0.0,
        My: float = 0.0,
        Mz: float = 0.0,
    ) -> None:
        super().__init__(name, comment, stage, distribution, Fx=Fx, Fy=Fy, Fz=Fz, Mx=Mx, My=My, Mz=Mz)
        if not isinstance(point, Point):
            raise TypeError("PointLoad requires a Point object for 'point'.")
        self._point = point

    @property
    def point(self) -> Point:
        return self._point

    def __repr__(self) -> str:
        return (
            f"<PointLoad name='{self.name}' stage='{self._stage.value}' "
            f"dist='{self._distribution.name}' {self._force_str()} "
            f"at=({self._point.x:.1f}, {self._point.y:.1f}, {self._point.z:.1f})>"
        )

class DynPointLoad(_DynBaseLoad, PointLoad):
    """Dynamic point load with per-component multipliers."""
    _MUL_KEYS: Tuple[str, ...] = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

    def __init__(
        self,
        name: str,
        comment: str,
        point: Point,
        Fx: float = 0.0,
        Fy: float = 0.0,
        Fz: float = 0.0,
        Mx: float = 0.0,
        My: float = 0.0,
        Mz: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        # 只调用一次 super().__init__，把所有参数都传上去
        super().__init__(
            name=name, comment=comment,
            distribution=DistributionType.UNIFORM,  # 点荷载通常只有 UNIFORM
            point=point,
            Fx=Fx, Fy=Fy, Fz=Fz, Mx=Mx, My=My, Mz=Mz,
            multiplier=multiplier,
        )

    def _default_multiplier_key(self) -> str:
        return "Fz"

    def __repr__(self) -> str:
        base = (f"<DynPointLoad name='{self.name}' stage='{self._stage.value}' "
                f"dist='{self._distribution.name}' {self._force_str()} "
                f"at=({self.point.x:.1f}, {self.point.y:.1f}, {self.point.z:.1f})>")
        return base + (self._mult_str() if self._mult else "")

# =============================================================================
#  Line Load – distributed forces (q)
# =============================================================================
class LineLoad(_BaseLoad):
    """Distributed load along a line (uniform or linear)."""
    def __init__(
        self,
        name: str,
        comment: str,
        line: Line3D,
        distribution: DistributionType = DistributionType.UNIFORM,
        stage: LoadStage = LoadStage.STATIC,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qx_end: float = 0.0,
        qy_end: float = 0.0,
        qz_end: float = 0.0,
    ) -> None:
        if not isinstance(line, Line3D):
            raise TypeError("LineLoad requires a Line3D object for 'line'.")
        if len(line) < 2:
            raise ValueError("LineLoad requires at least two points.")
        if distribution not in (DistributionType.UNIFORM, DistributionType.LINEAR):
            raise ValueError("LineLoad supports only UNIFORM or LINEAR distribution.")
        # map q* to base Fx/Fy/Fz
        super().__init__(
            name, comment, stage, distribution,
            Fx=qx, Fy=qy, Fz=qz, Fx_end=qx_end, Fy_end=qy_end, Fz_end=qz_end
        )
        self._line: Line3D = line

    @property
    def line(self) -> Line3D:
        return self._line

    # q* aliases
    @property
    def qx(self) -> float:
        return self._Fx

    @property
    def qy(self) -> float:
        return self._Fy

    @property
    def qz(self) -> float:
        return self._Fz

    @property
    def qx_end(self) -> float:
        return self._Fx_end

    @property
    def qy_end(self) -> float:
        return self._Fy_end

    @property
    def qz_end(self) -> float:
        return self._Fz_end

    def __repr__(self) -> str:
        base_info = (
            f"<LineLoad name='{self.name}' stage='{self._stage.value}' "
            f"dist='{self._distribution.name}' q=({self._Fx:.2f}, {self._Fy:.2f}, {self._Fz:.2f})"
        )
        if self._distribution == DistributionType.LINEAR:
            return f"{base_info} end=({self._Fx_end:.2f}, {self._Fy_end:.2f}, {self._Fz_end:.2f})>"
        return f"{base_info}>"


class DynLineLoad(_DynBaseLoad, LineLoad):
    """Distributed dynamic line load along a line (uniform or linear)."""
    _MUL_KEYS: Tuple[str, ...] = ("qx", "qy", "qz")

    def __init__(
        self,
        name: str,
        comment: str,
        line: Line3D,
        distribution: DistributionType = DistributionType.UNIFORM,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qx_end: float = 0.0,
        qy_end: float = 0.0,
        qz_end: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None
    ):
        super().__init__(
            name=name, comment=comment,
            line=line, distribution=distribution,
            qx=qx, qy=qy, qz=qz, qx_end=qx_end, qy_end=qy_end, qz_end=qz_end,
            multiplier=multiplier,
        )

    def _default_multiplier_key(self) -> str:
        return "qz"

    def __repr__(self) -> str:
        extra = ""
        if self._distribution == DistributionType.LINEAR:
            extra = f" end=({self._Fx_end:.2f}, {self._Fy_end:.2f}, {self._Fz_end:.2f})"
        return (
            f"<DynLineLoad name='{self.name}' stage='{self._stage.value}' "
            f"dist='{self._distribution.name}' q=({self._Fx:.2f}, {self._Fy:.2f}, {self._Fz:.2f})"
            f"{extra}>"
            f"{self._mult_str()}"
        )

# =============================================================================
#  Surface Load – distributed stresses (σ)
# =============================================================================
class SurfaceLoad(_BaseLoad):
    """
    Surface load with σx/σy/σz only; supports full Plaxis distribution list.
    Acts as the base for STATIC surface loads.
    """
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        distribution: DistributionType = DistributionType.UNIFORM,
        stage: LoadStage = LoadStage.STATIC,
        # constant stresses (kN/m²)
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        # end stresses for linear distribution
        sigmax_end: float = 0.0,
        sigmay_end: float = 0.0,
        sigmaz_end: float = 0.0,
        # gradients dict
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if not isinstance(surface, Polygon3D):
            raise TypeError("surface must be a Polygon3D instance.")
        # Map σ* → Fx/Fy/Fz to reuse base machinery
        super().__init__(
            name, comment, stage, distribution,
            Fx=sigmax, Fy=sigmay, Fz=sigmaz,
            Fx_end=sigmax_end, Fy_end=sigmay_end, Fz_end=sigmaz_end,
            gradients=gradients, ref_point=ref_point
        )
        self._surface = surface

    @property
    def surface(self) -> Polygon3D:
        return self._surface

    @property
    def sigmax(self) -> float:
        return self._Fx

    @property
    def sigmay(self) -> float:
        return self._Fy

    @property
    def sigmaz(self) -> float:
        return self._Fz

    @property
    def sigmax_end(self) -> float:
        return self._Fx_end

    @property
    def sigmay_end(self) -> float:
        return self._Fy_end

    @property
    def sigmaz_end(self) -> float:
        return self._Fz_end

    def __repr__(self) -> str:
        return (
            f"<SurfaceLoad name='{self.name}' stage='{self._stage.value}' "
            f"dist='{self._distribution.name}' s=({self._Fx:.2f}, {self._Fy:.2f}, {self._Fz:.2f})>"
        )


# =============================================================================
#  Dynamic Surface base
# =============================================================================
class _DynSurfaceLoadBase(_DynBaseLoad, SurfaceLoad):
    """Base class for dynamic surface loads (σx/σy/σz + multipliers)."""
    _MUL_KEYS: Tuple[str, ...] = ("sigmax", "sigmay", "sigmaz")

    def __init__(
        self,
        surface: Polygon3D,
        name: str,
        distribution: DistributionType,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        sigmax_end: float = 0.0,
        sigmay_end: float = 0.0,
        sigmaz_end: float = 0.0,
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
        **kwargs
    ) -> None:
        super().__init__(
            name=name, comment=kwargs.get("comment", ""),
            surface=surface, distribution=distribution,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            sigmax_end=sigmax_end, sigmay_end=sigmay_end, sigmaz_end=sigmaz_end,
            gradients=gradients, ref_point=ref_point,
            multiplier=multiplier,
        )

    def _default_multiplier_key(self) -> str:
        return "sigmaz"

    def __repr__(self) -> str:
        mult_info = self._mult_str()
        surface_repr = str(self.surface)
        base_load_info = (
            f"{self.name}: {self._force_str()}, {self._distribution.name}, "
            f"{self._stage.value} @ {surface_repr}"
        )
        return f"<plx.structures.DynSurfaceLoad {base_load_info}{',' + mult_info if mult_info else ''}>"

class DynSurfaceLoad(_DynSurfaceLoadBase):
    """Helper alias: dynamic surface load (stage preset to DYNAMIC)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# =============================================================================
#  Distribution-specific helper classes – STATIC
# =============================================================================
class UniformSurfaceLoad(SurfaceLoad):
    """σx / σy / σz constant over the surface."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        super().__init__(
            name, comment, surface,
            distribution=DistributionType.UNIFORM,
            stage=stage,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
        )


class LinearSurfaceLoad(SurfaceLoad):
    """Linear variation between start and end stresses (σ_start → σ_end)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax_start: float = 0.0,
        sigmay_start: float = 0.0,
        sigmaz_start: float = 0.0,
        sigmax_end: float = 0.0,
        sigmay_end: float = 0.0,
        sigmaz_end: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        super().__init__(
            name, comment, surface,
            distribution=DistributionType.LINEAR,
            stage=stage,
            sigmax=sigmax_start, sigmay=sigmay_start, sigmaz=sigmaz_start,
            sigmax_end=sigmax_end, sigmay_end=sigmay_end, sigmaz_end=sigmaz_end,
        )


class XAlignedIncrementSurfaceLoad(SurfaceLoad):
    """Stress with gradient only in +X direction (dσ/dx)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        sigmax_inc_x: float = 0.0,
        sigmay_inc_x: float = 0.0,
        sigmaz_inc_x: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {"gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x}
        ref_point_tuple = (x_ref, y_ref, z_ref)
        super().__init__(
            name, comment, surface,
            distribution=DistributionType.X_ALIGNED_INC,
            stage=stage,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=ref_point_tuple,
        )


class YAlignedIncrementSurfaceLoad(SurfaceLoad):
    """Stress gradient only in +Y direction (dσ/dy)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        sigmax_inc_y: float = 0.0,
        sigmay_inc_y: float = 0.0,
        sigmaz_inc_y: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {"gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y}
        ref_point_tuple = (x_ref, y_ref, z_ref)
        super().__init__(
            name, comment, surface,
            distribution=DistributionType.Y_ALIGNED_INC,
            stage=stage,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=ref_point_tuple,
        )


class ZAlignedIncrementSurfaceLoad(SurfaceLoad):
    """Stress gradient only in +Z direction (dσ/dz)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        sigmax_inc_z: float = 0.0,
        sigmay_inc_z: float = 0.0,
        sigmaz_inc_z: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {"gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z}
        ref_point_tuple = (x_ref, y_ref, z_ref)
        super().__init__(
            name, comment, surface,
            distribution=DistributionType.Z_ALIGNED_INC,
            stage=stage,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=ref_point_tuple,
        )


class VectorAlignedIncrementSurfaceLoad(SurfaceLoad):
    """Stress gradient aligned with a specified vector (dσ/dv)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        vector_x: float = 0.0,
        vector_y: float = 0.0,
        vector_z: float = 0.0,
        sigmax_inc_v: float = 0.0,
        sigmay_inc_v: float = 0.0,
        sigmaz_inc_v: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {
            "gx_v": sigmax_inc_v, "gy_v": sigmay_inc_v, "gz_v": sigmaz_inc_v,
            "vector_x": vector_x, "vector_y": vector_y, "vector_z": vector_z,
        }
        ref_point_tuple = (x_ref, y_ref, z_ref)
        super().__init__(
            name, comment, surface,
            distribution=DistributionType.VECTOR_ALIGNED_INC,
            stage=stage,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=ref_point_tuple,
        )


class FreeIncrementSurfaceLoad(SurfaceLoad):
    """Independent gradients in all three directions (free increment)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        sigmax_inc_x: float = 0.0,
        sigmay_inc_x: float = 0.0,
        sigmaz_inc_x: float = 0.0,
        sigmax_inc_y: float = 0.0,
        sigmay_inc_y: float = 0.0,
        sigmaz_inc_y: float = 0.0,
        sigmax_inc_z: float = 0.0,
        sigmay_inc_z: float = 0.0,
        sigmaz_inc_z: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        ref_point_tuple = (x_ref, y_ref, z_ref)
        gradients = {
            "gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x,
            "gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y,
            "gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z,
        }
        super().__init__(
            name, comment, surface,
            distribution=DistributionType.FREE_INCREMENT,
            stage=stage,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=gradients, ref_point=ref_point_tuple,
        )


class PerpendicularSurfaceLoad(SurfaceLoad):
    """Pressure applied normal to the surface (Perpendicular distribution)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        pressure: float,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        super().__init__(
            name, comment, surface,
            distribution=DistributionType.PERPENDICULAR,
            stage=stage,
            sigmaz=pressure,
        )


# =============================================================================
#  Distribution-specific helper classes – DYNAMIC
# =============================================================================
class DynUniformSurfaceLoad(_DynSurfaceLoadBase):
    """Dynamic uniform surface load."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        super().__init__(
            surface=surface, name=name, distribution=DistributionType.UNIFORM,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            multiplier=multiplier, comment=comment
        )


class DynLinearSurfaceLoad(_DynSurfaceLoadBase):
    """Dynamic linear surface load (σ_start → σ_end)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax_start: float = 0.0,
        sigmay_start: float = 0.0,
        sigmaz_start: float = 0.0,
        sigmax_end: float = 0.0,
        sigmay_end: float = 0.0,
        sigmaz_end: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        super().__init__(
            surface=surface, name=name, distribution=DistributionType.LINEAR,
            sigmax=sigmax_start, sigmay=sigmay_start, sigmaz=sigmaz_start,
            sigmax_end=sigmax_end, sigmay_end=sigmay_end, sigmaz_end=sigmaz_end,
            multiplier=multiplier, comment=comment
        )


class DynXAlignedIncrementSurfaceLoad(_DynSurfaceLoadBase):
    """Dynamic X-aligned increment surface load (dσ/dx)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        sigmax_inc_x: float = 0.0,
        sigmay_inc_x: float = 0.0,
        sigmaz_inc_x: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        grads = {"gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x}
        super().__init__(
            surface=surface, name=name, distribution=DistributionType.X_ALIGNED_INC,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier, comment=comment
        )


class DynYAlignedIncrementSurfaceLoad(_DynSurfaceLoadBase):
    """Dynamic Y-aligned increment surface load (dσ/dy)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        sigmax_inc_y: float = 0.0,
        sigmay_inc_y: float = 0.0,
        sigmaz_inc_y: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        grads = {"gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y}
        super().__init__(
            surface=surface, name=name, distribution=DistributionType.Y_ALIGNED_INC,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier, comment=comment
        )


class DynZAlignedIncrementSurfaceLoad(_DynSurfaceLoadBase):
    """Dynamic Z-aligned increment surface load (dσ/dz)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        sigmax_inc_z: float = 0.0,
        sigmay_inc_z: float = 0.0,
        sigmaz_inc_z: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        grads = {"gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z}
        super().__init__(
            surface=surface, name=name, distribution=DistributionType.Z_ALIGNED_INC,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier, comment=comment
        )


class DynVectorAlignedIncrementSurfaceLoad(_DynSurfaceLoadBase):
    """Dynamic vector-aligned increment surface load (dσ/dv)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        vector_x: float = 0.0,
        vector_y: float = 0.0,
        vector_z: float = 0.0,
        sigmax_inc_v: float = 0.0,
        sigmay_inc_v: float = 0.0,
        sigmaz_inc_v: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        grads = {
            "gx_v": sigmax_inc_v, "gy_v": sigmay_inc_v, "gz_v": sigmaz_inc_v,
            "vector_x": vector_x, "vector_y": vector_y, "vector_z": vector_z,
        }
        super().__init__(
            surface=surface, name=name, distribution=DistributionType.VECTOR_ALIGNED_INC,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier, comment=comment
        )


class DynFreeIncrementSurfaceLoad(_DynSurfaceLoadBase):
    """Dynamic free increment surface load (independent gradients)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        sigmax_inc_x: float = 0.0,
        sigmay_inc_x: float = 0.0,
        sigmaz_inc_x: float = 0.0,
        sigmax_inc_y: float = 0.0,
        sigmay_inc_y: float = 0.0,
        sigmaz_inc_y: float = 0.0,
        sigmax_inc_z: float = 0.0,
        sigmay_inc_z: float = 0.0,
        sigmaz_inc_z: float = 0.0,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        grads = {
            "gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x,
            "gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y,
            "gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z,
        }
        super().__init__(
            surface=surface, name=name, distribution=DistributionType.FREE_INCREMENT,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier, comment=comment
        )


class DynPerpendicularSurfaceLoad(_DynSurfaceLoadBase):
    """Dynamic perpendicular surface load (pressure normal to surface)."""
    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        pressure: float,
        multiplier: Optional[Dict[str, LoadMultiplier] | LoadMultiplier] = None,
    ):
        super().__init__(
            surface=surface, name=name, distribution=DistributionType.PERPENDICULAR,
            sigmaz=pressure,
            multiplier=multiplier, comment=comment
        )
