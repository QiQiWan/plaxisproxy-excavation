# =========================================
# Dynamic loads keep a reference to a base static load and
# carry the same mechanical parameters + per-component multipliers.
# Static Point/Line/Surface loads provide `create_dyn()` helpers
# that clone current magnitudes into a dynamic counterpart.
# (All comments in English as requested.)
# =========================================
from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, Tuple, List, Any, Union

from ..core.plaxisobject import PlaxisObject
from ..geometry import Point, Line3D, Polygon3D


__all__ = [
    # enums
    "LoadStage", "DistributionType", "SignalType",
    # multiplier
    "LoadMultiplier",
    # static loads
    "PointLoad", "LineLoad",
    "SurfaceLoad", "UniformSurfaceLoad",
    "XAlignedIncrementSurfaceLoad", "YAlignedIncrementSurfaceLoad", "ZAlignedIncrementSurfaceLoad",
    "VectorAlignedIncrementSurfaceLoad", "FreeIncrementSurfaceLoad",
    "PerpendicularSurfaceLoad",
    # dynamic (same params + multipliers, and keep `base`)
    "DynPointLoad", "DynLineLoad", "DynSurfaceLoad",
    "DynUniformSurfaceLoad",
    "DynXAlignedIncrementSurfaceLoad", "DynYAlignedIncrementSurfaceLoad", "DynZAlignedIncrementSurfaceLoad",
    "DynVectorAlignedIncrementSurfaceLoad", "DynFreeIncrementSurfaceLoad",
    "DynPerpendicularSurfaceLoad",
]


# ==============================
#  Enumerations
# ==============================
class LoadStage(Enum):
    STATIC = "Static"
    DYNAMIC = "Dynamic"


class DistributionType(Enum):
    # Point/Line/Surface
    UNIFORM = "Uniform"   # Constant
    LINEAR = "Linear"     # Start/End
    # Surface-only variants
    X_ALIGNED_INC = "x-aligned increment"
    Y_ALIGNED_INC = "y-aligned increment"
    Z_ALIGNED_INC = "z-aligned increment"
    VECTOR_ALIGNED_INC = "Vector-aligned increment"
    FREE_INCREMENT = "Free increment"
    PERPENDICULAR = "Perpendicular"
    PERPENDICULAR_VERT_INC = "Perpendicular, vertical increment"


class SignalType(Enum):
    HARMONIC = "Harmonic"
    TABLE = "Table"


class LoadMultiplierKey(Enum):
    X = "Multiplierx"
    Y = "Multipliery"
    Z = "Multiplierz"
    # For DynPointLoad
    Fx = "MultiplierFx"
    Fy = "MultiplierFy"
    Fz = "MultiplierFz"
    Mx = "MultiplierMx"
    My = "MultiplierMy"
    Mz = "MultiplierMz"

# ==============================
#  Load Multiplier
# ==============================
class LoadMultiplier(PlaxisObject):
    """Dynamic load multiplier (Harmonic or Table)."""
    def __init__(
        self,
        name: str,
        comment: str,
        signal_type: SignalType,
        *,
        amplitude: float | None = None,
        phase: float | None = None,
        frequency: float | None = None,
        table_data: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        super().__init__(name, comment)
        self._signal_type = signal_type
        self._amplitude: Optional[float] = None
        self._phase: Optional[float] = None
        self._frequency: Optional[float] = None
        self._table_data: Optional[List[Tuple[float, float]]] = None

        if signal_type == SignalType.HARMONIC:
            if amplitude is None or frequency is None or phase is None:
                raise ValueError("Harmonic requires amplitude, phase, and frequency.")
            self._amplitude = float(amplitude)
            self._phase = float(phase)
            self._frequency = float(frequency)
        elif signal_type == SignalType.TABLE:
            if not table_data or not isinstance(table_data, list):
                raise ValueError("Table signal requires 'table_data' list.")
            last_t: Optional[float] = None
            cleaned: List[Tuple[float, float]] = []
            for idx, (t, v) in enumerate(table_data):
                if not isinstance(t, (int, float)) or not isinstance(v, (int, float)):
                    raise ValueError(f"Table data at index {idx} must be numeric.")
                t, v = float(t), float(v)
                if last_t is not None and t < last_t:
                    raise ValueError("Table times must be non-decreasing.")
                last_t = t
                cleaned.append((t, v))
            self._table_data = cleaned
        else:
            raise NotImplementedError(f"Unknown signal type: {signal_type}")

    @property
    def signal_type(self) -> SignalType: return self._signal_type
    @property
    def amplitude(self) -> Optional[float]: return self._amplitude
    @property
    def phase(self) -> Optional[float]: return self._phase
    @property
    def frequency(self) -> Optional[float]: return self._frequency
    @property
    def table_data(self) -> Optional[List[Tuple[float, float]]]: return self._table_data


# ==============================
#  Base static loads
# ==============================
class _BaseLoad(PlaxisObject):
    """Store base magnitudes; mapper will translate to PLAXIS properties."""
    def __init__(
        self,
        name: str = "",
        comment: str = "",
        stage: LoadStage = LoadStage.STATIC,
        distribution: DistributionType = DistributionType.UNIFORM,
        *,
        Fx: float = 0.0, Fy: float = 0.0, Fz: float = 0.0,
        Fx_end: float = 0.0, Fy_end: float = 0.0, Fz_end: float = 0.0,
        Mx: float = 0.0, My: float = 0.0, Mz: float = 0.0,
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None,
        vector: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__(name, comment)
        if not isinstance(stage, LoadStage):
            raise TypeError("stage must be LoadStage")
        if not isinstance(distribution, DistributionType):
            raise TypeError("distribution must be DistributionType")
        self._stage, self._distribution = stage, distribution
        self._Fx, self._Fy, self._Fz = float(Fx), float(Fy), float(Fz)
        self._Mx, self._My, self._Mz = float(Mx), float(My), float(Mz)
        self._Fx_end, self._Fy_end, self._Fz_end = float(Fx_end), float(Fy_end), float(Fz_end)
        self._grad = dict(gradients or {})
        self._ref_point = ref_point
        self._vector = vector

    # readable formatting (optional)
    def _force_str(self) -> str:
        return f"Fx={self._Fx}, Fy={self._Fy}, Fz={self._Fz}, " \
               f"Fx_end={self._Fx_end}, Fy_end={self._Fy_end}, Fz_end={self._Fz_end}, " \
               f"Mx={self._Mx}, My={self._My}, Mz={self._Mz}"

    # properties
    @property
    def stage(self) -> LoadStage: return self._stage
    @property
    def distribution(self) -> DistributionType: return self._distribution

    # force/moment bases
    @property
    def Fx(self) -> float: return self._Fx
    @property
    def Fy(self) -> float: return self._Fy
    @property
    def Fz(self) -> float: return self._Fz
    @property
    def Fx_end(self) -> float: return self._Fx_end
    @property
    def Fy_end(self) -> float: return self._Fy_end
    @property
    def Fz_end(self) -> float: return self._Fz_end

    @property
    def Mx(self) -> float: return self._Mx
    @property
    def My(self) -> float: return self._My
    @property
    def Mz(self) -> float: return self._Mz

    # gradient / ref / vector (surface-specific)
    @property
    def grad(self) -> Dict[str, float]: return self._grad
    @property
    def ref_point(self) -> Optional[Tuple[float, float, float]]: return self._ref_point
    @property
    def vector(self) -> Optional[Tuple[float, float, float]]: return self._vector


# ==============================
#  Dynamic mixin (shared by all dynamic loads)
# ==============================
class _DynamicMixin:
    """
    Shared logic for dynamic loads:
      - keep `base` pointer to the static load
      - normalize and store per-component multipliers
    Subclasses must define `_MUL_KEYS` and `_canonical_key`.
    """
    _MUL_KEYS: Tuple[str, ...] = tuple()

    def _init_dynamic(
        self,
        *,
        base: Union["PointLoad", "LineLoad", "SurfaceLoad"],
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier],
        default_key: LoadMultiplierKey,
        key_aliases: Optional[Dict[str, LoadMultiplierKey]] = None,
    ) -> None:
        if base is None:
            raise ValueError("Dynamic load requires a valid `base` static load.")
        self._base = base
        self._mult: Dict[LoadMultiplierKey, LoadMultiplier] = {}
        self._aliases: dict[str, LoadMultiplierKey] = {
            "X": LoadMultiplierKey.X,
            "Y": LoadMultiplierKey.Y,
            "Z": LoadMultiplierKey.Z,
        }

        if multiplier is None:
            return

        if isinstance(multiplier, LoadMultiplier):
            # attach to default component (e.g., Fz / qz / sigz)
            self._mult[default_key] = multiplier
        elif isinstance(multiplier, dict):
            for k, v in multiplier.items():
                if not isinstance(v, LoadMultiplier):
                    raise TypeError(f"Multiplier for '{k}' must be LoadMultiplier.")
                self._mult[k] = v
        else:
            raise TypeError("multiplier must be LoadMultiplier or Dict[LoadMultiplierKey, LoadMultiplier].")

    # API sugar
    @property
    def base(self) -> Union["PointLoad", "LineLoad", "SurfaceLoad"]:
        return self._base

    @property
    def mult(self) -> Dict[LoadMultiplierKey, LoadMultiplier]:
        return self._mult

    def add_multiplier(self, k: LoadMultiplierKey, m: LoadMultiplier) -> None:
        if not isinstance(m, LoadMultiplier):
            raise TypeError("m must be LoadMultiplier")
        self._mult[k] = m


# ==============================
#  Point loads
# ==============================
class PointLoad(_BaseLoad):
    def __init__(
        self,
        name: str, comment: str, point: Point,
        *, Fx: float = 0.0, Fy: float = 0.0, Fz: float = 0.0,
        Mx: float = 0.0, My: float = 0.0, Mz: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        if not isinstance(point, Point):
            raise TypeError("point must be Point")
        super().__init__(name, comment, stage, DistributionType.UNIFORM,
                         Fx=Fx, Fy=Fy, Fz=Fz, Mx=Mx, My=My, Mz=Mz)
        self._point = point

    @property
    def point(self) -> Point: return self._point

    # ---- factory to create a dynamic counterpart with identical magnitudes ----
    def create_dyn(
        self,
        name: str,
        comment: str = "",
        *,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> "DynPointLoad":
        """Create a dynamic point load cloned from this static one."""
        return DynPointLoad(
            name=name,
            comment=comment,
            point=self.point,
            base=self,
            Fx=self.Fx, Fy=self.Fy, Fz=self.Fz,
            Mx=self.Mx, My=self.My, Mz=self.Mz,
            multiplier=multiplier,
        )


class DynPointLoad(_DynamicMixin, PointLoad):
    """Dynamic point load = PointLoad + multipliers + reference to `base`."""
    _MUL_KEYS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

    def __init__(
        self,
        name: str, comment: str, point: Point,
        *, base: PointLoad,
        Fx: float = 0.0, Fy: float = 0.0, Fz: float = 0.0,
        Mx: float = 0.0, My: float = 0.0, Mz: float = 0.0,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        super().__init__(name=name, comment=comment, point=point,
                         Fx=Fx, Fy=Fy, Fz=Fz, Mx=Mx, My=My, Mz=Mz,
                         stage=LoadStage.DYNAMIC)
        self._init_dynamic(base=base, multiplier=multiplier, default_key=LoadMultiplierKey.Fz)


# ==============================
#  Line loads
# ==============================
class LineLoad(_BaseLoad):
    def __init__(
        self,
        name: str, comment: str, line: Line3D,
        *,
        distribution: DistributionType = DistributionType.UNIFORM,
        stage: LoadStage = LoadStage.STATIC,
        qx: float = 0.0, qy: float = 0.0, qz: float = 0.0,
        qx_end: float = 0.0, qy_end: float = 0.0, qz_end: float = 0.0,
    ) -> None:
        if not isinstance(line, Line3D):
            raise TypeError("line must be Line3D")
        # Note: if Line3D doesn't implement __len__, remove this check in your project.
        try:
            if len(line) < 2:
                raise ValueError("LineLoad needs ≥ 2 points.")
        except Exception:
            pass
        if distribution not in (DistributionType.UNIFORM, DistributionType.LINEAR):
            raise ValueError("LineLoad supports UNIFORM / LINEAR only.")
        super().__init__(name, comment, stage, distribution,
                         Fx=qx, Fy=qy, Fz=qz, Fx_end=qx_end, Fy_end=qy_end, Fz_end=qz_end)
        self._line = line

    @property
    def line(self) -> Line3D: return self._line

    # q* aliases (mapper relies on these)
    @property
    def qx(self) -> float: return self._Fx
    @property
    def qy(self) -> float: return self._Fy
    @property
    def qz(self) -> float: return self._Fz
    @property
    def qx_end(self) -> float: return self._Fx_end
    @property
    def qy_end(self) -> float: return self._Fy_end
    @property
    def qz_end(self) -> float: return self._Fz_end

    # ---- factory to create a dynamic counterpart with identical magnitudes ----
    def create_dyn(
        self,
        name: str,
        comment: str = "",
        *,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> "DynLineLoad":
        """Create a dynamic line load cloned from this static one."""
        return DynLineLoad(
            name=name,
            comment=comment,
            line=self.line,
            base=self,
            distribution=self.distribution,
            qx=self.qx, qy=self.qy, qz=self.qz,
            qx_end=self.qx_end, qy_end=self.qy_end, qz_end=self.qz_end,
            multiplier=multiplier,
        )


class DynLineLoad(_DynamicMixin, LineLoad):
    """Dynamic line load = LineLoad + multipliers + reference to `base`."""
    _MUL_KEYS = ("qx", "qy", "qz")

    def __init__(
        self,
        name: str, comment: str, line: Line3D,
        *, base: LineLoad,
        distribution: DistributionType = DistributionType.UNIFORM,
        qx: float = 0.0, qy: float = 0.0, qz: float = 0.0,
        qx_end: float = 0.0, qy_end: float = 0.0, qz_end: float = 0.0,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        super().__init__(name=name, comment=comment, line=line, distribution=distribution,
                         qx=qx, qy=qy, qz=qz, qx_end=qx_end, qy_end=qy_end, qz_end=qz_end,
                         stage=LoadStage.DYNAMIC)
        self._init_dynamic(base=base, multiplier=multiplier, default_key=LoadMultiplierKey.Z)


# ==============================
#  Surface loads
# ==============================
class SurfaceLoad(_BaseLoad):
    def __init__(
        self,
        name: str = "", comment: str = "", surface: Polygon3D = Polygon3D(),
        *,
        distribution: DistributionType = DistributionType.UNIFORM,
        stage: LoadStage = LoadStage.STATIC,
        sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_end: float = 0.0, sigmay_end: float = 0.0, sigmaz_end: float = 0.0,
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None,
        vector: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        if not isinstance(surface, Polygon3D):
            raise TypeError("surface must be Polygon3D")
        super().__init__(name, comment, stage, distribution,
                         Fx=sigmax, Fy=sigmay, Fz=sigmaz,
                         Fx_end=sigmax_end, Fy_end=sigmay_end, Fz_end=sigmaz_end,
                         gradients=gradients, ref_point=ref_point, vector=vector)
        self._surface = surface

    @property
    def surface(self) -> Polygon3D: return self._surface

    # aliases used by mapper
    @property
    def sigmax(self) -> float: return self._Fx
    @property
    def sigmay(self) -> float: return self._Fy
    @property
    def sigmaz(self) -> float: return self._Fz
    @property
    def sigmax_end(self) -> float: return self._Fx_end
    @property
    def sigmay_end(self) -> float: return self._Fy_end
    @property
    def sigmaz_end(self) -> float: return self._Fz_end

    # ---- factory to create a dynamic counterpart with identical magnitudes ----
    def create_dyn(
        self,
        name: str,
        comment: str = "",
        *,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> "DynSurfaceLoad":
        """Create a dynamic surface load cloned from this static one."""
        return DynSurfaceLoad(
            name=name,
            comment=comment,
            surface=self.surface,
            base=self,
            distribution=self.distribution,
            sigmax=self.sigmax, sigmay=self.sigmay, sigmaz=self.sigmaz,
            sigmax_end=self.sigmax_end, sigmay_end=self.sigmay_end, sigmaz_end=self.sigmaz_end,
            gradients=dict(self.grad or {}),
            ref_point=self.ref_point,
            vector=self.vector,
            multiplier=multiplier,
        )


class DynSurfaceLoad(_DynamicMixin, SurfaceLoad):
    """Dynamic surface load = SurfaceLoad + multipliers + reference to `base`."""
    # We accept both 'sigx/sigy/sigz' and 'sigmax/sigmay/sigmaz' keys and normalize to sigx/sigy/sigz.
    _MUL_KEYS = ("sigx", "sigy", "sigz")

    def __init__(
        self,
        name: str = "", comment: str = "", surface: Polygon3D = Polygon3D(),
        *, base: SurfaceLoad,
        distribution: DistributionType = DistributionType.UNIFORM,
        sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_end: float = 0.0, sigmay_end: float = 0.0, sigmaz_end: float = 0.0,
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None,
        vector: Optional[Tuple[float, float, float]] = None,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        super().__init__(name=name, comment=comment, surface=surface, distribution=distribution,
                         stage=LoadStage.DYNAMIC,
                         sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
                         sigmax_end=sigmax_end, sigmay_end=sigmay_end, sigmaz_end=sigmaz_end,
                         gradients=gradients, ref_point=ref_point, vector=vector)
        self._init_dynamic(base=base, multiplier=multiplier, default_key=LoadMultiplierKey.Z)


# ==============================
#  Static convenience surface classes
# ==============================
class UniformSurfaceLoad(SurfaceLoad):
    def __init__(
        self, name: str, comment: str, surface: Polygon3D,
        *, sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        super().__init__(name, comment, surface, distribution=DistributionType.UNIFORM, stage=stage,
                         sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz)


class XAlignedIncrementSurfaceLoad(SurfaceLoad):
    """Gradient only along +X; mapper expects gx_x/gy_x/gz_x + ref point."""
    def __init__(
        self, name: str, comment: str, surface: Polygon3D,
        *, sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_inc_x: float = 0.0, sigmay_inc_x: float = 0.0, sigmaz_inc_x: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {"gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x}
        super().__init__(name, comment, surface, distribution=DistributionType.X_ALIGNED_INC, stage=stage,
                         sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz, gradients=grads, ref_point=(x_ref, y_ref, z_ref))


class YAlignedIncrementSurfaceLoad(SurfaceLoad):
    def __init__(
        self, name: str, comment: str, surface: Polygon3D,
        *, sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_inc_y: float = 0.0, sigmay_inc_y: float = 0.0, sigmaz_inc_y: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {"gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y}
        super().__init__(name, comment, surface, distribution=DistributionType.Y_ALIGNED_INC, stage=stage,
                         sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz, gradients=grads, ref_point=(x_ref, y_ref, z_ref))


class ZAlignedIncrementSurfaceLoad(SurfaceLoad):
    def __init__(
        self, name: str, comment: str, surface: Polygon3D,
        *, sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_inc_z: float = 0.0, sigmay_inc_z: float = 0.0, sigmaz_inc_z: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {"gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z}
        super().__init__(name, comment, surface, distribution=DistributionType.Z_ALIGNED_INC, stage=stage,
                         sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz, gradients=grads, ref_point=(x_ref, y_ref, z_ref))


class VectorAlignedIncrementSurfaceLoad(SurfaceLoad):
    def __init__(
        self, name: str, comment: str, surface: Polygon3D,
        *, sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        vector_x: float = 1.0, vector_y: float = 0.0, vector_z: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {"gx_v": 0.0, "gy_v": 0.0, "gz_v": 0.0}
        super().__init__(name, comment, surface, distribution=DistributionType.VECTOR_ALIGNED_INC, stage=stage,
                         sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
                         gradients=grads, ref_point=(x_ref, y_ref, z_ref), vector=(vector_x, vector_y, vector_z))


class FreeIncrementSurfaceLoad(SurfaceLoad):
    def __init__(
        self, name: str, comment: str, surface: Polygon3D,
        *, sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_inc_x: float = 0.0, sigmay_inc_x: float = 0.0, sigmaz_inc_x: float = 0.0,
        sigmax_inc_y: float = 0.0, sigmay_inc_y: float = 0.0, sigmaz_inc_y: float = 0.0,
        sigmax_inc_z: float = 0.0, sigmay_inc_z: float = 0.0, sigmaz_inc_z: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC,
    ) -> None:
        grads = {
            "gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x,
            "gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y,
            "gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z,
        }
        super().__init__(name, comment, surface, distribution=DistributionType.FREE_INCREMENT, stage=stage,
                         sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz, gradients=grads, ref_point=(x_ref, y_ref, z_ref))


class PerpendicularSurfaceLoad(SurfaceLoad):
    def __init__(self, name: str, comment: str, surface: Polygon3D, *, pressure: float,
                 stage: LoadStage = LoadStage.STATIC) -> None:
        super().__init__(name, comment, surface, distribution=DistributionType.PERPENDICULAR, stage=stage,
                         sigmax=0.0, sigmay=0.0, sigmaz=pressure)


# ==============================
#  Dynamic convenience surface classes (aliases)
# ==============================

class DynUniformSurfaceLoad(DynSurfaceLoad):
    """Dynamic uniform surface load: same params as UniformSurfaceLoad + multipliers."""
    def __init__(
        self,
        name: str, comment: str, surface: Polygon3D,
        *, base: SurfaceLoad,
        sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        super().__init__(
            name=name, comment=comment, surface=surface, base=base,
            distribution=DistributionType.UNIFORM,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            multiplier=multiplier,
        )

class DynXAlignedIncrementSurfaceLoad(DynSurfaceLoad):
    """Dynamic x-aligned increment surface load: gradient along +X + multipliers."""
    def __init__(
        self,
        name: str, comment: str, surface: Polygon3D,
        *, base: SurfaceLoad,
        sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_inc_x: float = 0.0, sigmay_inc_x: float = 0.0, sigmaz_inc_x: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        grads = {"gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x}
        super().__init__(
            name=name, comment=comment, surface=surface, base=base,
            distribution=DistributionType.X_ALIGNED_INC,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier,
        )


class DynYAlignedIncrementSurfaceLoad(DynSurfaceLoad):
    """Dynamic y-aligned increment surface load: gradient along +Y + multipliers."""
    def __init__(
        self,
        name: str, comment: str, surface: Polygon3D,
        *, base: SurfaceLoad,
        sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_inc_y: float = 0.0, sigmay_inc_y: float = 0.0, sigmaz_inc_y: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        grads = {"gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y}
        super().__init__(
            name=name, comment=comment, surface=surface, base=base,
            distribution=DistributionType.Y_ALIGNED_INC,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier,
        )


class DynZAlignedIncrementSurfaceLoad(DynSurfaceLoad):
    """Dynamic z-aligned increment surface load: gradient along +Z + multipliers."""
    def __init__(
        self,
        name: str, comment: str, surface: Polygon3D,
        *, base: SurfaceLoad,
        sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_inc_z: float = 0.0, sigmay_inc_z: float = 0.0, sigmaz_inc_z: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        grads = {"gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z}
        super().__init__(
            name=name, comment=comment, surface=surface, base=base,
            distribution=DistributionType.Z_ALIGNED_INC,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier,
        )


class DynVectorAlignedIncrementSurfaceLoad(DynSurfaceLoad):
    """Dynamic vector-aligned increment surface load: gradient along a given vector + multipliers."""
    def __init__(
        self,
        name: str, comment: str, surface: Polygon3D,
        *, base: SurfaceLoad,
        sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        vector_x: float = 1.0, vector_y: float = 0.0, vector_z: float = 0.0,
        # per-component increments along vector are optional; keep zeros to signal 'present but 0'
        sigmax_inc_v: float = 0.0, sigmay_inc_v: float = 0.0, sigmaz_inc_v: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        grads = {"gx_v": sigmax_inc_v, "gy_v": sigmay_inc_v, "gz_v": sigmaz_inc_v}
        super().__init__(
            name=name, comment=comment, surface=surface, base=base,
            distribution=DistributionType.VECTOR_ALIGNED_INC,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            vector=(vector_x, vector_y, vector_z),
            multiplier=multiplier,
        )


class DynFreeIncrementSurfaceLoad(DynSurfaceLoad):
    """Dynamic free-increment surface load: independent gradients in x/y/z for each component + multipliers."""
    def __init__(
        self,
        name: str, comment: str, surface: Polygon3D,
        *, base: SurfaceLoad,
        sigmax: float = 0.0, sigmay: float = 0.0, sigmaz: float = 0.0,
        sigmax_inc_x: float = 0.0, sigmay_inc_x: float = 0.0, sigmaz_inc_x: float = 0.0,
        sigmax_inc_y: float = 0.0, sigmay_inc_y: float = 0.0, sigmaz_inc_y: float = 0.0,
        sigmax_inc_z: float = 0.0, sigmay_inc_z: float = 0.0, sigmaz_inc_z: float = 0.0,
        x_ref: float = 0.0, y_ref: float = 0.0, z_ref: float = 0.0,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        grads = {
            "gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x,
            "gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y,
            "gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z,
        }
        super().__init__(
            name=name, comment=comment, surface=surface, base=base,
            distribution=DistributionType.FREE_INCREMENT,
            sigmax=sigmax, sigmay=sigmay, sigmaz=sigmaz,
            gradients=grads, ref_point=(x_ref, y_ref, z_ref),
            multiplier=multiplier,
        )


class DynPerpendicularSurfaceLoad(DynSurfaceLoad):
    """Dynamic perpendicular pressure: uses sigz as 'pressure' + multipliers."""
    def __init__(
        self,
        name: str, comment: str, surface: Polygon3D,
        *, base: SurfaceLoad,
        pressure: float,
        multiplier: Optional[Dict[LoadMultiplierKey, LoadMultiplier] | LoadMultiplier] = None,
    ) -> None:
        super().__init__(
            name=name, comment=comment, surface=surface, base=base,
            distribution=DistributionType.PERPENDICULAR,
            sigmax=0.0, sigmay=0.0, sigmaz=pressure,
            multiplier=multiplier,
        )

