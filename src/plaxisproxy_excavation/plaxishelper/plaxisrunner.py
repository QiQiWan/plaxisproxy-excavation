# src/plaxisproxy_excavation/plaxishelper/plaxisrunner.py

import time
from typing import List, Dict, Any, Optional, Union, Iterable, Sequence, Tuple

# Import the core Plaxis server and its required components from the local source
# This ensures we are using the exact version provided in the codebase.
from config.plaxis_config import HOST, PORT, PASSWORD

# Import the main data structure and the mapper class from your library
from ..plaxisexcavation import *  # noqa: F401,F403  (keep existing usage)
from .plaxismapper import PlaxisMapper  # noqa: F401
from plxscripting.server import new_server, Server, PlxProxyFactory
from ..geometry import *  # Point, PointSet, Line3D, Polygon3D, etc.

# NEW: use the static geometry mapper that wraps g_i.point/line/surface
from .geometrymapper import GeometryMapper


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
        validate_ring: bool = True,
        stop_on_error: bool = False,
        return_polygon: bool = False,
    ) -> Any:
        """
        Thin wrapper: create a PLAXIS surface via GeometryPlaxisMapper.
        See GeometryPlaxisMapper.create_surface for detailed behavior.
        """
        if self.g_i is None:
            raise RuntimeError("Not connected: g_i is None.")
        return GeometryMapper.create_surface(
            self.g_i,
            data,
            name=name,
            auto_close=auto_close,
            validate_ring=validate_ring,
            stop_on_error=stop_on_error,
            return_polygon=return_polygon,
        )
