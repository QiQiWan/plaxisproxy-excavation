import time
from typing import List, Dict, Any, Optional, Union, Iterable, Sequence, Tuple

# Import the core Plaxis server and its required components from the local source
# This ensures we are using the exact version provided in the codebase.
from config.plaxis_config import HOST, PORT, PASSWORD

# Import the main data structure and the mapper class from your library
# from ..plaxisexcavation import *  # noqa: F401,F403  (keep existing usage)
# Legacy facade (optional). Some codebases import PlaxisRunner.PlaxisMapper for DI.
# Avoid hard failure if the old facade module is not present.
try:
    from .plaxismapper import PlaxisMapper  # type: ignore  # noqa: F401
except Exception:
    PlaxisMapper = None  # type: ignore
from plxscripting.server import new_server, Server, PlxProxyFactory
from ..geometry import *  # Point, PointSet, Line3D, Polygon3D, etc.

# NEW: use the static geometry mapper that wraps g_i.point/line/surface
from .geometrymapper import GeometryMapper

# Mapper imports (thin wrappers exposed below)
from .projectinfomapper import ProjectInformationMapper
from .boreholemapper import BoreholeSetMapper
from .materialmapper import (
    SoilMaterialMapper, PlateMaterialMapper, BeamMaterialMapper,
    PileMaterialMapper, AnchorMaterialMapper,
)
from .structuremapper import (
    RetainingWallMapper as _RetainingWallMapper,
    BeamMapper as _BeamStructMapper,
    EmbeddedPileMapper as _EmbeddedPileMapper,
    AnchorMapper as _AnchorStructMapper,
    WellMapper as _WellMapper,
    SoilBlockMapper as _SoilBlockMapper,
)
from .loadmapper import LoadMapper, LoadMultiplierMapper
from .watertablemapper import WaterTableMapper
from .meshmapper import MeshMapper
from .monitormapper import MonitorMapper
from .phasemapper import PhaseMapper

# Domain class imports for type hints / isinstance dispatch
from ..borehole import BoreholeSet
from ..materials.soilmaterial import BaseSoilMaterial
from ..materials.platematerial import ElasticPlate, ElastoplasticPlate
from ..materials.beammaterial import ElasticBeam, ElastoplasticBeam
from ..materials.pilematerial import ElasticPile, ElastoplasticPile
from ..materials.anchormaterial import (
    ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor,
)
from ..structures.retainingwall import RetainingWall
from ..structures.beam import Beam
from ..structures.embeddedpile import EmbeddedPile
from ..structures.anchor import Anchor
from ..structures.well import Well
from ..structures.soilblock import SoilBlock
from ..structures.load import (
    _BaseLoad, LoadMultiplier,
)
from ..components.projectinformation import ProjectInformation
from ..components.watertable import WaterLevelTable
from ..components.mesh import Mesh
from ..components.phase import Phase


class PlaxisRunner:
    """
    Manages the remote session with Plaxis 3D for a complete workflow.

    This class handles connecting to the Plaxis remote scripting server,
    orchestrating the model creation via the PlaxisMapper, initiating mesh
    generation and calculations, saving the project, and retrieving results.

    Geometry creation (points/lines/surfaces) is delegated to the static
    utility class GeometryPlaxisMapper to improve separation of concerns.
    """

    def __init__(self, input_port: int = PORT, password: str = PASSWORD, host: str = HOST):
        """
        Initializes the runner with connection details for the Plaxis server.

        Args:
            input_port (int): The port number for the Plaxis Input server.
            password (str): The password configured for the server connection.
            host (str, optional): The host where Plaxis is running. Defaults to HOST.
        """
        self.host = host
        self.input_port = input_port
        self.output_port = None  # reserved for Output server if needed
        self.password = password

        self.is_connected = False

        self.input_server: Optional[Server] = None  # The Server object for the Input application
        self.output_server: Optional[Server] = None  # The Server object for the Output application
        self.g_i: Optional[PlxProxyFactory] = None  # The global input object (for modeling)
        self.g_o: Optional[PlxProxyFactory] = None  # The global output object (for results)

    # ------------------------- connection management -------------------------

    def connect(self) -> bool:
        """
        Establishes connections with the Plaxis Input server.
        """
        try:
            print(f"Attempting to connect to Plaxis Input at {self.host}:{self.input_port}...")
            # Note: new_server returns (server, g_i)
            self.input_server, self.g_i = new_server(HOST, PORT, password=PASSWORD)
            print("Successfully connected to Plaxis Input.")
            time.sleep(1)
            self.is_connected = True
            return True

        except ConnectionRefusedError:
            print("-----------------------------------------------------------------------")
            print(f"[Error] Connection was refused. Please confirm:")
            print("1. The Plaxis software has been launched.")
            print("2. The remote script service has been enabled.")
            print(f"3. The port number ({PORT}) and password settings are correct.")
            print("-----------------------------------------------------------------------")
            print("Please check for any errors and then re-execute the 'connect()' function.")
            return False

        except Exception as e:
            print(f"[Error] An unexpected error occurred during connection: {e}")
            return False

    def new(self) -> bool:
        """
        Creates a new project in the Plaxis Input application.

        Returns:
            bool: True if the new project was created successfully, False otherwise.
        """
        if not self.is_connected or self.g_i is None or self.input_server is None:
            print("[Error] Cannot create a new project: Not connected to Plaxis server.")
            print("Please call the connect() method first.")
            return False

        try:
            print("Sending 'new' command to Plaxis...")
            result = self.input_server.new()
            if result == "OK":
                time.sleep(1)
                print("New project successfully created.")
                return True
            else:
                raise Exception(f"Bad result: {result}!")
        except Exception as e:
            print(f"[Error] An unexpected error occurred while creating a new project: {e}")
            print("Please check the Plaxis application for any error messages or alerts.")
            return False

    # ===================== convenience wrappers to GeometryPlaxisMapper =====================

    # ---- Points ----
    def create_point(self, point: Point) -> Any:
        """
        Thin wrapper: create a PLAXIS point via GeometryPlaxisMapper and return the handle.
        """
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return GeometryMapper.create_point(self.g_i, point)

    def create_points(
        self,
        data: Union[
            PointSet,
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
        ],
        stop_on_error: bool = False,
    ) -> List[Any]:
        """
        Thin wrapper: batch create PLAXIS points via GeometryPlaxisMapper and return handles.
        """
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return GeometryMapper.create_points(self.g_i, data, stop_on_error=stop_on_error)

    # ---- Lines ----
    def create_line(
        self,
        data: Union[
            # points-based inputs
            PointSet,
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
            Tuple[Point, Point],
            Tuple[Tuple[float, float, float], Tuple[float, float, float]],
            # line-based inputs
            Line3D,
            Sequence[Line3D],
            Iterable[Line3D],
        ],
        name: Optional[str] = None,
        stop_on_error: bool = False,
    ) -> Union[Any, List[Any], Line3D, List[Line3D], None]:
        """
        Thin wrapper: create PLAXIS line(s) via GeometryPlaxisMapper.
        See GeometryPlaxisMapper.create_line for detailed behavior.
        """
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return GeometryMapper.create_line(self.g_i, data, name=name, stop_on_error=stop_on_error)

    # ---- Surfaces ----
    def create_surface(
        self,
        data: Union[
            Polygon3D,
            PointSet,
            Iterable[Point],
            Iterable[Tuple[float, float, float]],
            Sequence[Point],
            Sequence[Tuple[float, float, float]],
        ],
        name: Optional[str] = None,
        auto_close: bool = True,
        stop_on_error: bool = False,
        return_polygon: bool = False,
    ) -> Any:
        """
        Thin wrapper: create a PLAXIS surface via GeometryPlaxisMapper.
        See GeometryPlaxisMapper.create_surface for detailed behavior.
        """
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        # GeometryMapper.create_surface does not take a validate_ring flag;
        # closure and basic validity are handled internally.
        return GeometryMapper.create_surface(
            self.g_i,
            data,
            name=name,
            auto_close=auto_close,
            stop_on_error=stop_on_error,
            return_polygon=return_polygon,
        )

    # ===================== Project information =====================
    def apply_project_information(self, proj: ProjectInformation) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return ProjectInformationMapper.create(self.g_i, proj)

    # ===================== Boreholes & layers =====================
    def create_boreholes(self, bhset: BoreholeSet, *, normalize: bool = True, set_name_on_objects: bool = True) -> Dict[str, List[Tuple[float, float]]]:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return BoreholeSetMapper.create(self.g_i, bhset, normalize=normalize, set_name_on_objects=set_name_on_objects)

    # ===================== Materials =====================
    def create_material(self, mat: Any) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        if isinstance(mat, BaseSoilMaterial):
            return SoilMaterialMapper.create_material(self.g_i, mat)
        if isinstance(mat, (ElasticPlate, ElastoplasticPlate)):
            return PlateMaterialMapper.create_material(self.g_i, mat)  # type: ignore[arg-type]
        if isinstance(mat, (ElasticBeam, ElastoplasticBeam)):
            return BeamMaterialMapper.create_material(self.g_i, mat)   # type: ignore[arg-type]
        if isinstance(mat, (ElasticPile, ElastoplasticPile)):
            return PileMaterialMapper.create_material(self.g_i, mat)   # type: ignore[arg-type]
        if isinstance(mat, (ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor)):
            return AnchorMaterialMapper.create_material(self.g_i, mat) # type: ignore[arg-type]
        raise TypeError(f"Unsupported material type: {type(mat).__name__}")

    def delete_material(self, mat_or_handle: Any) -> bool:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        # Dispatch by domain type when possible, otherwise try each mapper (safe no-ops)
        if isinstance(mat_or_handle, BaseSoilMaterial):
            return SoilMaterialMapper.delete_material(self.g_i, mat_or_handle)
        if isinstance(mat_or_handle, (ElasticPlate, ElastoplasticPlate)):
            return PlateMaterialMapper.delete_material(self.g_i, mat_or_handle)  # type: ignore[arg-type]
        if isinstance(mat_or_handle, (ElasticBeam, ElastoplasticBeam)):
            return BeamMaterialMapper.delete_material(self.g_i, mat_or_handle)   # type: ignore[arg-type]
        if isinstance(mat_or_handle, (ElasticPile, ElastoplasticPile)):
            return PileMaterialMapper.delete_material(self.g_i, mat_or_handle)   # type: ignore[arg-type]
        if isinstance(mat_or_handle, (ElasticAnchor, ElastoplasticAnchor, ElastoPlasticResidualAnchor)):
            return AnchorMaterialMapper.delete_material(self.g_i, mat_or_handle) # type: ignore[arg-type]
        # Fallback: try all and OR results
        ok = False
        for mapper in (SoilMaterialMapper, PlateMaterialMapper, BeamMaterialMapper, PileMaterialMapper, AnchorMaterialMapper):
            try:
                ok = mapper.delete_material(self.g_i, mat_or_handle) or ok
            except Exception:
                continue
        return ok

    # ===================== Structures =====================
    def create_retaining_wall(self, wall: RetainingWall) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return _RetainingWallMapper.create(self.g_i, wall)

    def create_beam(self, beam: Beam) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return _BeamStructMapper.create(self.g_i, beam)

    def create_anchor(self, anchor: Anchor) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return _AnchorStructMapper.create(self.g_i, anchor)

    def create_embedded_pile(self, pile: EmbeddedPile) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return _EmbeddedPileMapper.create(self.g_i, pile)

    def create_well(self, well: Well) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return _WellMapper.create(self.g_i, well)

    def create_soil_block(self, block: SoilBlock) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return _SoilBlockMapper.create(self.g_i, block)

    # ===================== Loads =====================
    def create_load(self, load: Any) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return LoadMapper.create(self.g_i, load)

    def delete_load(self, load_or_handle: Any) -> None:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return LoadMapper.delete(self.g_i, load_or_handle)

    def create_load_multiplier(self, mul: LoadMultiplier) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return LoadMultiplierMapper.create(self.g_i, mul)

    # ===================== Water table =====================
    def create_water_table(self, table: WaterLevelTable, *, goto_flow: bool = True) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return WaterTableMapper.create_table(self.g_i, table, goto_flow=goto_flow)

    def update_water_table(self, table: WaterLevelTable, *, rebuild_if_needed: bool = True) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return WaterTableMapper.update_table(self.g_i, table, rebuild_if_needed=rebuild_if_needed)

    def delete_water_table(self, table: WaterLevelTable) -> None:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return WaterTableMapper.delete_table(self.g_i, table)

    # ===================== Mesh =====================
    def mesh(self, mesh: Mesh) -> str:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        if hasattr(self.g_i, "gotomesh"):
            goto = getattr(self.g_i, "gotomesh")
            goto()
        return MeshMapper.generate(self.g_i, mesh)

    # ===================== Monitors =====================
    # def create_monitors(self, monitors: List["CurvePoint"]) -> None:
    #     if self.g_i is None:
    #         raise RuntimeError("Not connected: g_i is None.")
    #     return MonitorMapper.create_monitors(self.g_i, monitors)

    # ===================== Phases =====================
    def goto_stages(self) -> bool:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return PhaseMapper.goto_stages(self.g_i)

    def get_initial_phase(self) -> Phase:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return PhaseMapper.get_initial_phase(self.g_i)

    def create_phase(self, phase: Phase, inherits: Optional[Any] = None) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        base = inherits if inherits is not None else (getattr(getattr(phase, "inherits", None), "plx_id", None))
        return PhaseMapper.create(self.g_i, phase, inherits=base)

    def apply_phase(self, phase_handle: Any, phase: Phase, *, warn_on_missing: bool = False) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return PhaseMapper.apply_phase(self.g_i, phase_handle, phase, warn_on_missing=warn_on_missing)

    def update_phase(
        self,
        phase: Phase,
        *,
        warn_on_missing: bool = False,
        allow_recreate: bool = False,
        sync_meta: bool = True,
    ) -> Any:
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return PhaseMapper.update(
            self.g_i,
            phase,
            warn_on_missing=warn_on_missing,
            allow_recreate=allow_recreate,
            sync_meta=sync_meta,
        )
