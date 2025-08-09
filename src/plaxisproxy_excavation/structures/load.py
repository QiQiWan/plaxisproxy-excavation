from enum import Enum, auto
from typing import Optional, Dict, Tuple, List, Any
from core.plaxisobject import PlaxisObject
from ..geometry import Point, Line3D, Polygon3D

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
#  Multiplier Types (Copied from 'load_multiplier_update' immersive)
# =============================================================================
class SignalType(Enum):
    """
    Supported signal types for multipliers (mirrors Plaxis/similar software options).
    """
    HARMONIC = "Harmonic"         # Harmonic signal (Amplitude, Phase, Frequency)
    TABLE = "Table"               # Table/Time History curve (List of Time-Multiplier pairs)

class LoadMultiplier(PlaxisObject):
    """
    Represents a load multiplier, defining a time or frequency function
    to be applied to dynamic loads.
    """
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
            
            self._amplitude = amplitude
            self._phase = phase
            self._frequency = frequency

        elif signal_type == SignalType.TABLE:
            if table_data is None:
                raise ValueError("Table signal requires 'table_data'.")
            if not isinstance(table_data, list):
                raise ValueError("'table_data' must be a list.")
            for i, point in enumerate(table_data):
                if not (isinstance(point, (tuple, list)) and len(point) == 2 and
                        isinstance(point[0], (int, float)) and isinstance(point[1], (int, float))):
                    raise ValueError(f"Table data point at index {i} must be a (time, value) tuple of numbers.")
                if i > 0 and point[0] < table_data[i-1][0]:
                    raise ValueError(f"Table data must be ordered by time (ascending). Error at index {i}.")
            self._table_data = table_data
        else:
            raise NotImplementedError(f"Signal type '{signal_type.value}' is not yet fully implemented.")

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
        base_repr = (
            f"<plx.structures.LoadMultiplier(name='{self._name}', "
            f"signal='{self._signal_type.value}')"
        )
        if self._signal_type == SignalType.HARMONIC:
            return (
                f"{base_repr}, "
                f"amplitude={self._amplitude:.3f}, "
                f"phase={self._phase:.1f}°, "
                f"frequency={self._frequency:.3f} Hz>"
            )
        elif self._signal_type == SignalType.TABLE:
            if self._table_data and len(self._table_data) > 4:
                table_str = (
                    f"[{self._table_data[0]}, {self._table_data[1]}, ..., "
                    f"{self._table_data[-2]}, {self._table_data[-1]}]"
                )
            else:
                table_str = str(self._table_data)
            return (
                f"{base_repr}, "
                f"table_data={table_str}>"
            )
        else:
            return f"{base_repr}>"


# =============================================================================
#  Base class (forces + optional moments) – moments used by PointLoad only
# =============================================================================
class _BaseLoad(PlaxisObject):
    """Shared attributes for load entities (no direct instantiation)."""

    # ------------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        name: str,
        comment: str,
        stage: LoadStage,
        distribution: DistributionType,
        # —— primary force components ——
        Fx: float = 0.0,
        Fy: float = 0.0,
        Fz: float = 0.0,
        # —— optional moment components (PointLoad only) ——
        Mx: float = 0.0,
        My: float = 0.0,
        Mz: float = 0.0,
        # —— end values (Line / Surface, linear only) ——
        Fx_end: float = 0.0,
        Fy_end: float = 0.0,
        Fz_end: float = 0.0,
        # —— gradients dictionary (Surface) ——
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None
    ) -> None:
        
        super().__init__(name, comment)
        if not isinstance(stage, LoadStage):
            raise TypeError("stage must be a LoadStage value.")
        if not isinstance(distribution, DistributionType):
            raise TypeError("distribution must be a DistributionType value.")
        if any(not isinstance(v, (int, float)) for v in [Fx, Fy, Fz, Mx, My, Mz, Fx_end, Fy_end, Fz_end]):
            raise TypeError("Force and moment components must be numeric.")
        if gradients is not None:
            if not isinstance(gradients, dict):
                raise TypeError("'gradients' must be a dict of str to float values.")
            for k, val in gradients.items():
                if not isinstance(k, str) or not isinstance(val, (int, float)):
                    raise ValueError("Gradient entries must have string keys and numeric values.")
        if ref_point is not None:
            if not (isinstance(ref_point, (tuple, list)) and len(ref_point) == 3 and 
                    all(isinstance(c, (int, float)) for c in ref_point)):
                raise ValueError("ref_point must be a tuple of three numbers.")
        self._stage = stage
        self._distribution = distribution
        # force & moment assignments remain unchanged...
        self._Fx, self._Fy, self._Fz = Fx, Fy, Fz
        self._Mx, self._My, self._Mz = Mx, My, Mz
        self._Fx_end, self._Fy_end, self._Fz_end = Fx_end, Fy_end, Fz_end
        # ... (Mx/My/Mz and end values assignments unchanged)
        self._grad = gradients or {}
        self._ref_point = ref_point or (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    #  Pretty helpers
    # ------------------------------------------------------------------
    def _force_str(self) -> str:
        s = f"F=({self._Fx:+.2f},{self._Fy:+.2f},{self._Fz:+.2f})"
        if self._distribution is DistributionType.LINEAR:
            s += f"→({self._Fx_end:+.2f},{self._Fy_end:+.2f},{self._Fz_end:+.2f})"
        if self._grad:
            s += f", grad={self._grad}"
        return s

    def _moment_str(self) -> str:
        return f"M=({self._Mx:+.2f},{self._My:+.2f},{self._Mz:+.2f})"

    # ------------------------------------------------------------------
    #  Public accessors (id, name, …)
    # ------------------------------------------------------------------
    @property
    def id(self):
        return self._id

    @property
    def plx_id(self):
        return self._plx_id
    
    @plx_id.setter
    def plx_id(self, value):
        self._plx_id = value

    @property
    def stage(self):
        return self._stage

    @property
    def distribution(self):
        return self._distribution
    
    @property
    def Fx(self) -> float:
        """Gets the force component in the X-direction (start/uniform value)."""
        return self._Fx

    @property
    def Fy(self) -> float:
        """Gets the force component in the Y-direction (start/uniform value)."""
        return self._Fy

    @property
    def Fz(self) -> float:
        """Gets the force component in the Z-direction (start/uniform value)."""
        return self._Fz

    @property
    def Mx(self) -> float:
        """Gets the moment component around the X-axis (start/uniform value)."""
        return self._Mx

    @property
    def My(self) -> float:
        """Gets the moment component around the Y-axis (start/uniform value)."""
        return self._My

    @property
    def Mz(self) -> float:
        """Gets the moment component around the Z-axis (start/uniform value)."""
        return self._Mz

    @property
    def Fx_end(self) -> float:
        """Gets the force component in the X-direction at the end of the line."""
        return self._Fx_end

    @property
    def Fy_end(self) -> float:
        """Gets the force component in the Y-direction at the end of the line."""
        return self._Fy_end

    @property
    def Fz_end(self) -> float:
        """Gets the force component in the Z-direction at the end of the line."""
        return self._Fz_end

    # --- Surface Gradients --------------------------------------------------------

    @property
    def grad(self) -> dict:
        """Gets the dictionary defining force/moment gradients over a surface."""
        return self._grad

    @property
    def ref_point(self) -> tuple:
        """Gets the reference point (x, y, z) for gradient calculations."""
        return self._ref_point

class _DynBaseLoad(_BaseLoad):
    """Shared attributes for dynamic load entities (no direct instantiation)."""

    def __init__(
        self, 
        bind_obj, 
        multiplier: Optional[Dict[str, LoadMultiplier]] = None, # Updated type hint
        *args, 
        **kwargs
    ) -> None:
        if multiplier is not None and not isinstance(multiplier, dict):
            raise TypeError("multiplier must be a dictionary of LoadMultiplier objects.")
        kwargs["stage"] = LoadStage.DYNAMIC
        super().__init__(*args, **kwargs)
        self._bind_obj = bind_obj
        self._mult: Dict[str, LoadMultiplier] = multiplier.copy() if multiplier else {}

    # ------------------------------------------------------------------
    #  Multiplier helpers (validated per-class)
    # ------------------------------------------------------------------
    def _allowed_mul_keys(self) -> Tuple[str, ...]:
        """Override in subclasses to constrain accepted multiplier keys."""
        return ()

    def multiplier(self, comp: str) -> Optional[LoadMultiplier]: # Updated return type
        """Return multiplier LoadMultiplier object for component."""
        return self._mult.get(comp)

    def set_multiplier(self, comp: str, multiplier_obj: LoadMultiplier): # Updated parameter type
        if comp not in self._allowed_mul_keys():
            raise ValueError(
                f"Component '{comp}' not allowed for {self.__class__.__name__}. "
                f"Allowed: {self._allowed_mul_keys()}"
            )
        if not isinstance(multiplier_obj, LoadMultiplier): # Added type check
            raise TypeError(f"Multiplier for component '{comp}' must be a LoadMultiplier instance.")
        self._mult[comp] = multiplier_obj

    def _mult_str(self) -> str: # Updated logic to show multiplier names
        if not self._mult:
            return ""
        # Example: "mult={Fx: LoadMultiplier_1, Fy: LoadMultiplier_2}"
        mult_parts = [f"{comp}: {m.name}" for comp, m in self._mult.items()]
        return f"mult={{{', '.join(mult_parts)}}}"

    @property
    def bind_obj(self):
        return self._bind_obj
    
    @bind_obj.setter
    def bind_obj(self, value):
        self._bind_obj = value

    @property
    def mult(self):
        return self._mult
    
    @mult.setter
    def mult(self, value):
        self._mult = value

    def __repr__(self) -> str:
        return f"<{self.__class__.__module__}.{self.__class__.__name__}>"
        

# =============================================================================
#  Point Load – supports forces *and* moments + 6 multipliers
# =============================================================================
class PointLoad(_BaseLoad):
    """Concentrated node load (Fx/Fy/Fz + optional Mx/My/Mz)."""

    # ----- ctor ---------------------------------------------------------
    def __init__(
        self,
        name: str,
        comment: str, 
        point: Point,
        stage: LoadStage = LoadStage.STATIC,
        Fx: float = 0.0,
        Fy: float = 0.0,
        Fz: float = 0.0,
        Mx: float = 0.0,
        My: float = 0.0,
        Mz: float = 0.0,
    ) -> None:
        """
        Args:
            ...
            Fx (float): Force component in the x-direction [kN].
            Fy (float): Force component in the y-direction [kN].
            Fz (float): Force component in the z-direction [kN].
            Mx (float): Bending moment in the x-direction [kN m].
            My (float): Bending moment in the y-direction [kN m].
            Mz (float): Bending moment in the z-direction [kN m].
        """
        super().__init__(
            name,
            comment,
            stage,
            DistributionType.UNIFORM,
            Fx,
            Fy,
            Fz,
            Mx,
            My,
            Mz,
        )
        super().__init__(name, comment, stage, DistributionType.UNIFORM, Fx, Fy, Fz, Mx, My, Mz)
        if not isinstance(point, Point):
            raise TypeError("PointLoad requires a Point object for 'point'.")
        self._point = point



    # ----- properties ---------------------------------------------------
    @property
    def point(self):
        return self._point

    # ----- representation ----------------------------------------------
    def __repr__(self):
        # Assuming _point has a get_point() method or similar for representation
        point_repr = self._point.get_point() if hasattr(self._point, 'get_point') else str(self._point)
        return (
            f"<plx.structures.PointLoad {self._name}: {self._force_str()}, {self._moment_str()}, "
            f"{self._stage.value} @ {point_repr}>"
        )

class DynPointLoad(PointLoad, _DynBaseLoad):
    """DynPointLoad is a extra object to bind a general point load"""

    _MUL_KEYS: Tuple[str, ...] = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC
        super().__init__(*args, **kwargs)
        # validate multiplier keys - this loop now expects LoadMultiplier objects in _mult
        # The _mult is initialized in _DynBaseLoad's __init__
        for k, mul_obj in self._mult.items(): # Iterate over items to get both key and object
            self.set_multiplier(k, mul_obj)  # will raise if key invalid or type invalid

    # ----- multiplier key list -----------------------------------------
    def _allowed_mul_keys(self) -> Tuple[str, ...]:
        return self._MUL_KEYS

    def __repr__(self) -> str:
        mult_info = self._mult_str()
        # Assuming _point has a get_point() method or similar for representation
        point_repr = self._point.get_point() if hasattr(self._point, 'get_point') else str(self._point)
        return (
            f"<plx.structures.DynPointLoad {self._name}: "
            f"{self._force_str()}, {self._moment_str()}, "
            f"{self._stage.value} @ {point_repr}"
            f"{', ' + mult_info if mult_info else ''}>"
        )

# =============================================================================
#  Line Load – distributed forces (q) with 3 multipliers (qx/qy/qz)
# =============================================================================
class LineLoad(_BaseLoad):
    """Distributed load along a line (uniform or linear)."""

    # ----- ctor ---------------------------------------------------------
    def __init__(
        self,
        name: str,
        comment: str,
        line: Any, # Assuming Line3D is from ..geometry, using Any for standalone code
        distribution: DistributionType = DistributionType.UNIFORM,
        stage: LoadStage = LoadStage.STATIC,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qx_end: float = 0.0,
        qy_end: float = 0.0,
        qz_end: float = 0.0,
        
    ) -> None:
        """
        Args:
            ...
            qx (float): Load component in the x-direction [kN/m].
            qy (float): Load component in the y-direction [kN/m].
            qz (float): Load component in the z-direction [kN/m].
        """
        # Assuming line object has a __len__ method
        if not isinstance(line, Line3D):
            raise TypeError("LineLoad requires a Line3D object for 'line'.")
        if hasattr(line, '__len__') and len(line) < 2:
            raise ValueError("LineLoad requires at least two points.")
        if distribution not in (DistributionType.UNIFORM, DistributionType.LINEAR):
            raise ValueError("LineLoad supports only UNIFORM or LINEAR distribution.")
        super().__init__(name, comment, stage, distribution)
        self._line = line
        self._qx = qx
        self._qy = qy
        self._qz = qz
        self._qx_end = qx_end
        self._qy_end = qy_end
        self._qz_end = qz_end

    # ----- properties ---------------------------------------------------
    @property
    def line(self):
        return self._line

    @property
    def qx(self) -> float:
        """Gets the distributed load component in the X-direction (start/uniform value)."""
        return self._qx

    @property
    def qy(self) -> float:
        """Gets the distributed load component in the Y-direction (start/uniform value)."""
        return self._qy

    @property
    def qz(self) -> float:
        """Gets the distributed load component in the Z-direction (start/uniform value)."""
        return self._qz

    @property
    def qx_end(self) -> float:
        """Gets the distributed load component in the X-direction at the end of the line."""
        return self._qx_end

    @property
    def qy_end(self) -> float:
        """Gets the distributed load component in the Y-direction at the end of the line."""
        return self._qy_end

    @property
    def qz_end(self) -> float:
        """Gets the distributed load component in the Z-direction at the end of the line."""
        return self._qz_end

    # ----- representation ----------------------------------------------
    def __repr__(self):
        # Assuming _line has a 'name' attribute or a good __repr__
        line_repr = self._line.name if hasattr(self._line, 'name') else str(self._line)
        return (
            f"<plx.structures.LineLoad {self._name}: {self._force_str()}, {self._distribution.name}, "
            f"{self._stage.value} @ {line_repr}>"
        )

class DynLineLoad(LineLoad, _DynBaseLoad):
    """Distrubuted dynamic line load along a line (uniform or linear)."""

    _MUL_KEYS: Tuple[str, ...] = ("qx", "qy", "qz")

    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC
        super().__init__(*args, **kwargs)
        # validate multiplier keys - this loop now expects LoadMultiplier objects in _mult
        for k, mul_obj in self._mult.items():
            self.set_multiplier(k, mul_obj)  # will raise if key invalid or type invalid

    # ----- multiplier key list -----------------------------------------
    def _allowed_mul_keys(self) -> Tuple[str, ...]:
        return self._MUL_KEYS
    
    # ----- representation ----------------------------------------------
    def __repr__(self) -> str: 
        mult_info = self._mult_str() 
        # Assuming _line has a 'name' attribute or a good __repr__
        line_repr = self._line.name if hasattr(self._line, 'name') else str(self._line)
        base_load_info = (
            f"{self._name}: {self._force_str()}, "
            f"{self._distribution.name}, {self._stage.value} @ {line_repr}"
        )
        if mult_info:
            final_repr = (
                f"<plx.structures.DynLineLoad {base_load_info}, {mult_info}>"
            )
        else:
            final_repr = (
                f"<plx.structures.DynLineLoad {base_load_info}>"
            )
        return final_repr
    
# =============================================================================
#  Surface Load – distributed stresses (σ) with 3 multipliers (sigmax/sigmay/sigmaz)
# =============================================================================
class SurfaceLoad(_BaseLoad):
    """
    Surface load with σx/σy/σz only; supports full Plaxis distribution list.
    This class now acts as the base for STATIC surface loads.
    """

    # _MUL_KEYS is now defined in _DynSurfaceLoadBase
    # multiplier parameter and related logic removed from here

    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D, # Assuming Polygon3D is from ..geometry, using Any for standalone code
        distribution: DistributionType = DistributionType.UNIFORM,
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
        # constant stresses (kN/m²)
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        # end stresses for linear distribution
        sigmax_end: float = 0.0,
        sigmay_end: float = 0.0,
        sigmaz_end: float = 0.0,
        # gradients dict, e.g. {'gx_x': -0.2, 'gz_z': -9.81}
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Args:
            ...
            sigmax (float): Load component in the x-direction [kN/m2].
            sigmay (float): Load component in the y-direction [kN/m2].
            sigmaz (float): Load component in the z-direction [kN/m2].
            gradients (Optional[Dict[str, float]]):
                A dictionary defining stress gradients.
                Keys are strings like 'gz_z' (gradient of sigma_z along z-axis).
                Values are the rate of change. [kN/m²/m]
            ref_point (Optional[Tuple[float, float, float]]):
                The 3D reference point (x, y, z) from which gradients or linear
                distributions are calculated. [m]
        """
        # Assuming surface object has as_tuple_list method
        if not isinstance(surface, Polygon3D):
            raise TypeError("surface must be a Polygon3D instance.")
        if not hasattr(surface, "as_tuple_list") or not callable(surface.as_tuple_list):
            raise ValueError("Surface polygon object must implement as_tuple_list().")
        super().__init__(name, comment, stage, distribution, gradients=gradients, ref_point=ref_point)
        self._surface = surface
        self._sigmax = sigmax
        self._sigmay = sigmay
        self._sigmaz = sigmaz
        self._sigmax_end = sigmax_end
        self._sigmay_end = sigmay_end
        self._sigmaz_end = sigmaz_end

        
        # Multiplier logic removed from SurfaceLoad

    # Multiplier helpers (_allowed_mul_keys, multiplier, set_multiplier, _mult_str)
    # are now moved to _DynSurfaceLoadBase

    # ------------------------------------------------------------------
    @property
    def surface(self):
        return self._surface

    @property
    def sigmax(self) -> float:
        """Gets the normal stress component in the X-direction (start/uniform value)."""
        return self._sigmax

    @property
    def sigmay(self) -> float:
        """Gets the normal stress component in the Y-direction (start/uniform value)."""
        return self._sigmay

    @property
    def sigmaz(self) -> float:
        """Gets the normal stress component in the Z-direction (start/uniform value)."""
        return self._sigmaz

    @property
    def sigmax_end(self) -> float:
        """Gets the normal stress component in the X-direction at the end of the line."""
        return self._sigmax_end

    @property
    def sigmay_end(self) -> float:
        """Gets the normal stress component in the Y-direction at the end of the line."""
        return self._sigmay_end

    @property
    def sigmaz_end(self) -> float:
        """Gets the normal stress component in the Z-direction at the end of the line."""
        return self._sigmaz_end

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        # Assuming _surface has a 'name' attribute or a good __repr__
        surface_repr = str(self._surface)
        return (
            f"<plx.structures.SurfaceLoad {self._name}: {self._force_str()}, {self._distribution.name}, "
            f"{self._stage.value} @ {surface_repr}>"
        )

# =============================================================================
#  NEW: Base class for Dynamic Surface Loads
# =============================================================================
class _DynSurfaceLoadBase(SurfaceLoad, _DynBaseLoad):
    """
    Base class for dynamic surface loads, combining SurfaceLoad properties
    with dynamic multiplier handling from _DynBaseLoad.
    """
    _MUL_KEYS: Tuple[str, ...] = ("sigmax", "sigmay", "sigmaz") # Default for surface loads

    def __init__(
        self,
        surface: Any,
        name: str,
        distribution: DistributionType,
        # Forces/Stresses
        sigmax: float = 0.0,
        sigmay: float = 0.0,
        sigmaz: float = 0.0,
        sigmax_end: float = 0.0,
        sigmay_end: float = 0.0,
        sigmaz_end: float = 0.0,
        gradients: Optional[Dict[str, float]] = None,
        ref_point: Optional[Tuple[float, float, float]] = None,
        multiplier: Optional[Dict[str, LoadMultiplier]] = None, # Multiplier handled by _DynBaseLoad
        **kwargs # Catch any extra args for super() calls
    ) -> None:
        # Call SurfaceLoad's __init__ (which calls _BaseLoad's __init__)
        # Note: stage is set to DYNAMIC by _DynBaseLoad's __init__
        super().__init__(
            surface=surface,
            name=name,
            distribution=distribution,
            stage=LoadStage.DYNAMIC, # Explicitly set for dynamic base
            sigmax=sigmax,
            sigmay=sigmay,
            sigmaz=sigmaz,
            sigmax_end=sigmax_end,
            sigmay_end=sigmay_end,
            sigmaz_end=sigmaz_end,
            gradients=gradients,
            ref_point=ref_point,
            **kwargs # Pass any remaining kwargs
        )
        # Call _DynBaseLoad's __init__
        # bind_obj can be self, or a more specific binding object if needed
        _DynBaseLoad.__init__(self, bind_obj=self, multiplier=multiplier)

    # Override _allowed_mul_keys from _DynBaseLoad
    def _allowed_mul_keys(self) -> Tuple[str, ...]:
        return self._MUL_KEYS

    def __repr__(self) -> str:
        mult_info = self._mult_str()
        surface_repr = str(self._surface)
        base_load_info = (
            f"{self._name}: {self._force_str()}, {self._distribution.name}, "
            f"{self._stage.value} @ {surface_repr}"
        )
        if mult_info:
            return (
                f"<plx.structures.DynSurfaceLoad {base_load_info}, {mult_info}>"
            )
        else:
            return (
                f"<plx.structures.DynSurfaceLoad {base_load_info}>"
            )

class DynSurfaceLoad(_DynSurfaceLoadBase):
    """Helper alias: dynamic surface load (stage preset to DYNAMIC)."""

    def __init__(self, *args, **kwargs):
        # _DynSurfaceLoadBase already sets stage to DYNAMIC
        super().__init__(*args, **kwargs)


# =============================================================================
#  Distribution‑specific helper classes (Surface)
# =============================================================================
# These subclasses expose only the parameters relevant for a given distribution
# type, making the API self‑documenting and harder to misuse.
# -----------------------------------------------------------------------------

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
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
    ) -> None:
        super().__init__(
            name,
            comment,
            surface,
            distribution=DistributionType.UNIFORM,
            stage=stage,
            sigmax=sigmax,
            sigmay=sigmay,
            sigmaz=sigmaz,
        )

# Dynamic version of UniformSurfaceLoad
class DynUniformSurfaceLoad(UniformSurfaceLoad, _DynSurfaceLoadBase):
    """Dynamic uniform surface load."""
    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC # Ensure dynamic stage
        super().__init__(*args, **kwargs)

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
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
    ) -> None:
        super().__init__(
            name,
            comment,
            surface,
            distribution=DistributionType.LINEAR,
            stage=stage,
            sigmax=sigmax_start,
            sigmay=sigmay_start,
            sigmaz=sigmaz_start,
            sigmax_end=sigmax_end,
            sigmay_end=sigmay_end,
            sigmaz_end=sigmaz_end,
        )

# Dynamic version of LinearSurfaceLoad
class DynLinearSurfaceLoad(LinearSurfaceLoad, _DynSurfaceLoadBase):
    """Dynamic linear surface load."""
    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC # Ensure dynamic stage
        super().__init__(*args, **kwargs)

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
        # Renamed parameters to match image terminology
        sigmax_inc_x: float = 0.0,
        sigmay_inc_x: float = 0.0,
        sigmaz_inc_x: float = 0.0,
        # Added individual reference point parameters
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
    ) -> None:
        """
        Args:
            surface (Any): The 3D geometry surface on which the load is applied.
            name (str): The name of the load object.
            distribution (DistributionType): The distribution type of the stress.
            stage (LoadStage): The calculation stage in which the load is active.
            sigmax (float): Base normal stress in the x-direction at the reference point. [kN/m²]
            sigmay (float): Base normal stress in the y-direction at the reference point. [kN/m²]
            sigmaz (float): Base normal stress in the z-direction at the reference point. [kN/m²]
            sigmax_inc_x (float): The rate of change (gradient) of sigmax along the x-axis. [kN/m²/m]
            sigmay_inc_x (float): The rate of change (gradient) of sigmay along the x-axis. [kN/m²/m]
            sigmaz_inc_x (float): The rate of change (gradient) of sigmaz along the x-axis. [kN/m²/m]
            x_ref (float): The x-coordinate of the reference point where base stresses are defined. [m]
            y_ref (float): The y-coordinate of the reference point where base stresses are defined. [m]
            z_ref (float): The z-coordinate of the reference point where base stresses are defined. [m]
        """
        # Map new parameter names to internal gradient keys expected by _BaseLoad
        grads = {"gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x}
        # Combine individual ref point parameters into a tuple for _BaseLoad
        ref_point_tuple = (x_ref, y_ref, z_ref)
        super().__init__(
            name,
            comment,
            surface,
            distribution=DistributionType.X_ALIGNED_INC,
            stage=stage,
            sigmax=sigmax,
            sigmay=sigmay,
            sigmaz=sigmaz,
            gradients=grads,
            ref_point=ref_point_tuple, # Pass the constructed tuple
        )

# Dynamic version of XAlignedIncrementSurfaceLoad
class DynXAlignedIncrementSurfaceLoad(XAlignedIncrementSurfaceLoad, _DynSurfaceLoadBase):
    """Dynamic X-aligned increment surface load."""
    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC # Ensure dynamic stage
        super().__init__(*args, **kwargs)


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
        # Renamed parameters to match image terminology
        sigmax_inc_y: float = 0.0,
        sigmay_inc_y: float = 0.0,
        sigmaz_inc_y: float = 0.0,
        # Added individual reference point parameters
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
    ) -> None:
        """
        Args:
            surface (Any): The 3D geometry surface on which the load is applied.
            name (str): The name of the load object.
            distribution (DistributionType): The distribution type of the stress.
            stage (LoadStage): The calculation stage in which the load is active.
            sigmax (float): Base normal stress in the x-direction at the reference point. [kN/m²]
            sigmay (float): Base normal stress in the y-direction at the reference point. [kN/m²]
            sigmaz (float): Base normal stress in the z-direction at the reference point. [kN/m²]
            sigmax_inc_x (float): The rate of change (gradient) of sigmax along the x-axis. [kN/m²/m]
            sigmay_inc_x (float): The rate of change (gradient) of sigmay along the x-axis. [kN/m²/m]
            sigmaz_inc_x (float): The rate of change (gradient) of sigmaz along the x-axis. [kN/m²/m]
            x_ref (float): The x-coordinate of the reference point where base stresses are defined. [m]
            y_ref (float): The y-coordinate of the reference point where base stresses are defined. [m]
            z_ref (float): The z-coordinate of the reference point where base stresses are defined. [m]
        """
        # Map new parameter names to internal gradient keys expected by _BaseLoad
        grads = {"gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y}
        # Combine individual ref point parameters into a tuple for _BaseLoad
        ref_point_tuple = (x_ref, y_ref, z_ref)
        super().__init__(
            name,
            comment,
            surface,
            distribution=DistributionType.Y_ALIGNED_INC,
            stage=stage,
            sigmax=sigmax,
            sigmay=sigmay,
            sigmaz=sigmaz,
            gradients=grads,
            ref_point=ref_point_tuple, # Pass the constructed tuple
        )

# Dynamic version of YAlignedIncrementSurfaceLoad
class DynYAlignedIncrementSurfaceLoad(YAlignedIncrementSurfaceLoad, _DynSurfaceLoadBase):
    """Dynamic Y-aligned increment surface load."""
    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC # Ensure dynamic stage
        super().__init__(*args, **kwargs)


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
        # Renamed parameters to match image terminology
        sigmax_inc_z: float = 0.0,
        sigmay_inc_z: float = 0.0,
        sigmaz_inc_z: float = 0.0,
        # Added individual reference point parameters
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
    ) -> None:
        """
        Args:
            surface (Any): The 3D geometry surface on which the load is applied.
            name (str): The name of the load object.
            distribution (DistributionType): The distribution type of the stress.
            stage (LoadStage): The calculation stage in which the load is active.
            sigmax (float): Base normal stress in the x-direction at the reference point. [kN/m²]
            sigmay (float): Base normal stress in the y-direction at the reference point. [kN/m²]
            sigmaz (float): Base normal stress in the z-direction at the reference point. [kN/m²]
            sigmax_inc_x (float): The rate of change (gradient) of sigmax along the x-axis. [kN/m²/m]
            sigmay_inc_x (float): The rate of change (gradient) of sigmay along the x-axis. [kN/m²/m]
            sigmaz_inc_x (float): The rate of change (gradient) of sigmaz along the x-axis. [kN/m²/m]
            x_ref (float): The x-coordinate of the reference point where base stresses are defined. [m]
            y_ref (float): The y-coordinate of the reference point where base stresses are defined. [m]
            z_ref (float): The z-coordinate of the reference point where base stresses are defined. [m]
        """
        # Map new parameter names to internal gradient keys expected by _BaseLoad
        grads = {"gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z}
        # Combine individual ref point parameters into a tuple for _BaseLoad
        ref_point_tuple = (x_ref, y_ref, z_ref)
        super().__init__(
            name,
            comment,
            surface,
            distribution=DistributionType.Z_ALIGNED_INC,
            stage=stage,
            sigmax=sigmax,
            sigmay=sigmay,
            sigmaz=sigmaz,
            gradients=grads,
            ref_point=ref_point_tuple, # Pass the constructed tuple
        )

# Dynamic version of ZAlignedIncrementSurfaceLoad
class DynZAlignedIncrementSurfaceLoad(ZAlignedIncrementSurfaceLoad, _DynSurfaceLoadBase):
    """Dynamic Z-aligned increment surface load."""
    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC # Ensure dynamic stage
        super().__init__(*args, **kwargs)


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
        # Reference point parameters
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        # Vector components
        vector_x: float = 0.0,
        vector_y: float = 0.0,
        vector_z: float = 0.0,
        # Increment stresses aligned with vector
        sigmax_inc_v: float = 0.0,
        sigmay_inc_v: float = 0.0,
        sigmaz_inc_v: float = 0.0,
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
    ) -> None:
        """
        Initializes a surface load with a vector-aligned stress gradient.

        Args:
            surface (Any): The 3D geometry surface on which the load is applied.
            name (str): The unique name of the load object.
            sigmax (float): The base normal stress in the x-direction at the reference point. **[kN/m²]**
            sigmay (float): The base normal stress in the y-direction at the reference point. **[kN/m²]**
            sigmaz (float): The base normal stress in the z-direction at the reference point. **[kN/m²]**
            x_ref (float): The x-coordinate of the reference point where the base stresses (sigmax, sigmay, sigmaz) are defined. **[m]**
            y_ref (float): The y-coordinate of the reference point. **[m]**
            z_ref (float): The z-coordinate of the reference point. **[m]**
            vector_x (float): The x-component of the direction vector for the gradient. This defines the 'v' direction. **[Dimensionless]**
            vector_y (float): The y-component of the direction vector. **[Dimensionless]**
            vector_z (float): The z-component of the direction vector. **[Dimensionless]**
            sigmax_inc_v (float): The rate of change (gradient) of sigmax per unit distance along the defined vector 'v'. **[kN/m²/m]**
            sigmay_inc_v (float): The rate of change (gradient) of sigmay per unit distance along the defined vector 'v'. **[kN/m²/m]**
            sigmaz_inc_v (float): The rate of change (gradient) of sigmaz per unit distance along the defined vector 'v'. **[kN/m²/m]**
            stage (LoadStage): The calculation stage in which the load is active (e.g., STATIC, INITIAL).
        """
        # Map increment parameters to internal gradient keys for vector alignment
        grads = {
            "gx_v": sigmax_inc_v,
            "gy_v": sigmay_inc_v,
            "gz_v": sigmaz_inc_v,
            # Store vector components in gradients dict as well for _BaseLoad to handle
            # This assumes _BaseLoad's _grad can handle these keys for representation
            "vector_x": vector_x,
            "vector_y": vector_y,
            "vector_z": vector_z,
        }
        ref_point_tuple = (x_ref, y_ref, z_ref)
        super().__init__(
            name,
            comment,
            surface,
            distribution=DistributionType.VECTOR_ALIGNED_INC,
            stage=stage,
            sigmax=sigmax,
            sigmay=sigmay,
            sigmaz=sigmaz,
            gradients=grads, # Pass the combined gradients including vector
            ref_point=ref_point_tuple,
        )

# Dynamic version of VectorAlignedIncrementSurfaceLoad
class DynVectorAlignedIncrementSurfaceLoad(VectorAlignedIncrementSurfaceLoad, _DynSurfaceLoadBase):
    """Dynamic vector-aligned increment surface load."""
    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC # Ensure dynamic stage
        super().__init__(*args, **kwargs)


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
        # Added individual reference point parameters
        x_ref: float = 0.0,
        y_ref: float = 0.0,
        z_ref: float = 0.0,
        # Gradients in X direction
        sigmax_inc_x: float = 0.0,
        sigmay_inc_x: float = 0.0,
        sigmaz_inc_x: float = 0.0,
        # Gradients in Y direction
        sigmax_inc_y: float = 0.0,
        sigmay_inc_y: float = 0.0,
        sigmaz_inc_y: float = 0.0,
        # Gradients in Z direction
        sigmax_inc_z: float = 0.0,
        sigmay_inc_z: float = 0.0,
        sigmaz_inc_z: float = 0.0,
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
    ) -> None:
        """
        Initializes a surface load with fully independent stress gradients.

        Args:
            surface (Any): The 3D geometry surface on which the load is applied.
            name (str): The unique name of the load object.
            sigmax (float): The base normal stress in the x-direction at the reference point. **[kN/m²]**
            sigmay (float): The base normal stress in the y-direction at the reference point. **[kN/m²]**
            sigmaz (float): The base normal stress in the z-direction at the reference point. **[kN/m²]**
            x_ref (float): The x-coordinate of the reference point where base stresses are defined. **[m]**
            y_ref (float): The y-coordinate of the reference point. **[m]**
            z_ref (float): The z-coordinate of the reference point. **[m]**
            sigmax_inc_x (float): The rate of change of sigmax along the x-axis. **[kN/m²/m]**
            sigmay_inc_x (float): The rate of change of sigmay along the x-axis. **[kN/m²/m]**
            sigmaz_inc_x (float): The rate of change of sigmaz along the x-axis. **[kN/m²/m]**
            sigmax_inc_y (float): The rate of change of sigmax along the y-axis. **[kN/m²/m]**
            sigmay_inc_y (float): The rate of change of sigmay along the y-axis. **[kN/m²/m]**
            sigmaz_inc_y (float): The rate of change of sigmaz along the y-axis. **[kN/m²/m]**
            sigmax_inc_z (float): The rate of change of sigmax along the z-axis. **[kN/m²/m]**
            sigmay_inc_z (float): The rate of change of sigmay along the z-axis. **[kN/m²/m]**
            sigmaz_inc_z (float): The rate of change of sigmaz along the z-axis. **[kN/m²/m]**
            stage (LoadStage): The calculation stage in which the load is active.
        """
        # Combine individual ref point parameters into a tuple for _BaseLoad
        ref_point_tuple = (x_ref, y_ref, z_ref)
        # Combine all gradient components into a single dictionary
        gradients = {
            "gx_x": sigmax_inc_x, "gy_x": sigmay_inc_x, "gz_x": sigmaz_inc_x,
            "gx_y": sigmax_inc_y, "gy_y": sigmay_inc_y, "gz_y": sigmaz_inc_y,
            "gx_z": sigmax_inc_z, "gy_z": sigmay_inc_z, "gz_z": sigmaz_inc_z,
        }
        super().__init__(
            name,
            comment,
            surface,
            distribution=DistributionType.FREE_INCREMENT,
            stage=stage,
            sigmax=sigmax,
            sigmay=sigmay,
            sigmaz=sigmaz,
            gradients=gradients, # Pass the constructed gradients dictionary
            ref_point=ref_point_tuple, # Pass the constructed tuple
        )

# Dynamic version of FreeIncrementSurfaceLoad
class DynFreeIncrementSurfaceLoad(FreeIncrementSurfaceLoad, _DynSurfaceLoadBase):
    """Dynamic free increment surface load."""
    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC # Ensure dynamic stage
        super().__init__(*args, **kwargs)


class PerpendicularSurfaceLoad(SurfaceLoad):
    """Pressure applied normal to the surface (Perpendicular distribution)."""

    def __init__(
        self,
        name: str,
        comment: str,
        surface: Polygon3D,
        pressure: float,
        stage: LoadStage = LoadStage.STATIC, # Default to STATIC
    ) -> None:
        super().__init__(
            name,
            comment,
            surface,
            distribution=DistributionType.PERPENDICULAR,
            stage=stage,
            sigmaz=pressure,  # Plaxis uses σn; map to σz (convention)
        )

# Dynamic version of PerpendicularSurfaceLoad
class DynPerpendicularSurfaceLoad(PerpendicularSurfaceLoad, _DynSurfaceLoadBase):
    """Dynamic perpendicular surface load."""
    def __init__(self, *args, **kwargs):
        kwargs["stage"] = LoadStage.DYNAMIC # Ensure dynamic stage
        super().__init__(*args, **kwargs)
