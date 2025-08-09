# src/plaxisproxy_excavation/plaxishelper/plaxisrunner.py

import time
from typing import List, Dict, Any, Optional

# Import the core Plaxis server and its required components from the local source
# This ensures we are using the exact version provided in the codebase.
# CORRECTED: Relative import path changed from 3 dots to 4 dots to correctly
# navigate from src/plaxisproxy_excavation/plaxishelper/ up to the root and then
# into third_party/.
from ....third_party.plxscripting.server import Server
from ....third_party.plxscripting.connection import HttpConnection
from ....third_party.plxscripting.plxproxyfactory import PlxProxyFactory
from ....third_party.plxscripting.tokenizer import PlxTokenizer

# Import the main data structure and the mapper class from your library
from ..plaxisexcavation import PlaxisFoundationPit
from .plaxismapper import PlaxisMapper


class PlaxisRunner:
    """
    Manages the remote session with Plaxis 3D for a complete workflow.
    
    This class handles connecting to the Plaxis remote scripting server,
    orchestrating the model creation via the PlaxisMapper, initiating mesh
    generation and calculations, saving the project, and retrieving results.
    It is designed based on the provided plxscripting source code for full compatibility.
    """
    
    def __init__(self, input_port: int, output_port: int, password: str, host: str = 'localhost'):
        """
        Initializes the runner with connection details for the Plaxis server.
        
        Args:
            input_port (int): The port number for the Plaxis Input server.
            output_port (int): The port number for the Plaxis Output server.
            password (str): The password configured for the server connection.
            host (str, optional): The host where Plaxis is running. Defaults to 'localhost'.
        """
        self.host = host
        self.input_port = input_port
        self.output_port = output_port
        self.password = password
        
        self.input_server: Optional[Server] = None  # The Server object for the Input application
        self.output_server: Optional[Server] = None # The Server object for the Output application
        self.g_i: Optional[Any] = None             # The global input object (for modeling)
        self.g_o: Optional[Any] = None             # The global output object (for results)

    def connect(self, start_new_project: bool = True) -> bool:
        """
        Establishes connections with both the Plaxis Input and Output servers.
        
        This method manually constructs the necessary components required by the
        core `plxscripting.server.Server` constructor, as revealed by the source code.
        
        Args:
            start_new_project (bool): If True, a new Plaxis project will be created upon connection.
        
        Returns:
            bool: True if both connections were successful, False otherwise.
        """
        try:
            # --- Connect to Input Server ---
            print(f"Attempting to connect to Plaxis Input at {self.host}:{self.input_port}...")
            input_connection = HttpConnection(self.host, self.input_port, password=self.password)
            self.input_server = Server(input_connection, PlxProxyFactory(), PlxTokenizer())
            self.g_i = self.input_server.get_global_input()
            print("Successfully connected to Plaxis Input.")

            # --- Connect to Output Server ---
            print(f"Attempting to connect to Plaxis Output at {self.host}:{self.output_port}...")
            output_connection = HttpConnection(self.host, self.output_port, password=self.password)
            self.output_server = Server(output_connection, PlxProxyFactory(), PlxTokenizer())
            
            # IMPORTANT: The method is still called get_global_input(), but because it's
            # connected to the output port, it correctly returns the global OUTPUT object (g_o).
            self.g_o = self.output_server.get_global_input()
            print("Successfully connected to Plaxis Output.")

            if start_new_project and self.g_i:
                print("Starting a new project...")
                self.g_i.new()
            
            return True
            
        except ConnectionRefusedError:
            print(f"ERROR: Connection refused. Please ensure Plaxis is running and the scripting server is enabled on both ports ({self.input_port} and {self.output_port}) with the correct password.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during connection: {e}")
            raise

    def build_model(self, model: PlaxisFoundationPit):
        """
        Builds the entire geotechnical model in Plaxis using the PlaxisMapper.
        
        Args:
            model (PlaxisFoundationPit): The complete Python object model to be translated into Plaxis.
        """
        if not self.g_i:
            print("Cannot build model: Not connected to Plaxis. Please call connect() first.")
            return
            
        print("\n" + "="*25)
        print("STEP 1: Building Model Geometry and Phases")
        print("="*25)
        mapper = PlaxisMapper(self.g_i)
        mapper.run(model)
        print("Model construction finished.")

    def generate_mesh(self):
        """
        Switches to the mesh mode and generates the finite element mesh.
        """
        if not self.g_i:
            print("Cannot generate mesh: Not connected to Plaxis.")
            return
        
        print("\n" + "="*25)
        print("STEP 2: Generating Finite Element Mesh")
        print("="*25)
        try:
            self.g_i.gotomesh()
            print("Switched to mesh mode. Starting mesh generation...")
            self.g_i.generatemesh()
            print("Mesh generation successful.")
        except Exception as e:
            print(f"ERROR during mesh generation: {e}")

    def calculate(self, phases_to_calculate: Optional[List[str]] = None):
        """
        Switches to staged construction mode and starts the calculation process.
        
        Args:
            phases_to_calculate (Optional[List[str]]): 
                A list of phase names to calculate. If None, all phases marked for
                calculation in the model will be run.
        """
        if not self.g_i:
            print("Cannot calculate: Not connected to Plaxis.")
            return
            
        print("\n" + "="*25)
        print("STEP 3: Starting Calculation")
        print("="*25)
        try:
            self.g_i.gotostages()
            print("Switched to staged construction mode.")

            if phases_to_calculate:
                print(f"Setting specific phases to calculate: {phases_to_calculate}")
                for phase in self.g_i.Phases:
                    phase.ShouldCalculate = False
                for phase_name in phases_to_calculate:
                    try:
                        target_phase = getattr(self.g_i.Phases, phase_name)
                        target_phase.ShouldCalculate = True
                    except AttributeError:
                        print(f"Warning: Phase '{phase_name}' not found in the model. It will be skipped.")

            print("Sending calculation command to Plaxis...")
            self.g_i.calculate()
            print("Calculation command sent. Plaxis is now processing.")
        except Exception as e:
            print(f"ERROR during calculation setup: {e}")
            
    def save(self, file_path: Optional[str] = None):
        """
        Saves the Plaxis project.
        
        Args:
            file_path (Optional[str]): The full path where the project should be saved.
                                       If None, a standard "Save" command is issued.
        """
        if not self.g_i:
            print("Cannot save: Not connected to Plaxis.")
            return
        
        print("\n" + "="*25)
        print("STEP 4: Saving Project")
        print("="*25)
        try:
            if file_path:
                self.g_i.save(file_path)
                print(f"Project saved to {file_path}")
            else:
                self.g_i.save()
                print("Project saved.")
        except Exception as e:
            print(f"ERROR while saving project: {e}")

    def get_results(self, monitor_points: List[str]) -> Dict[str, Any]:
        """
        Retrieves results for specified curve points from the Plaxis Output.
        
        Args:
            monitor_points (List[str]): A list of monitor point labels (names) to retrieve results for.
            
        Returns:
            Dict[str, Any]: A dictionary where keys are monitor point labels and values are the results.
        """
        if not self.g_o:
            print("Cannot get results: Not connected to Plaxis Output.")
            return {}
            
        print("\n" + "="*25)
        print("STEP 5: Retrieving Results")
        print("="*25)
        results = {}
        
        last_phase = self.g_o.Phases[-1]
        print(f"Retrieving results for the last phase: '{last_phase.Name}'")
        
        for point_label in monitor_points:
            try:
                # This is a hypothetical command and needs to be adapted to the actual Plaxis API
                # result_value = self.g_o.getcurveresults(point_label, last_phase, self.g_o.ResultTypes.Displacement.Utot)
                # results[point_label] = result_value[-1]
                print(f"  - (Placeholder) Retrieving total displacement for '{point_label}'.")
                results[point_label] = "Some result value"
            except Exception as e:
                print(f"  - ERROR retrieving results for '{point_label}': {e}")
                results[point_label] = None
        
        return results

    def close(self):
        """
        Closes the connections to the servers.
        Note: This does NOT close the Plaxis application itself.
        """
        if self.input_server or self.output_server:
            print("\nClosing connections to Plaxis servers.")
            self.input_server = None
            self.output_server = None
            self.g_i = None
            self.g_o = None
