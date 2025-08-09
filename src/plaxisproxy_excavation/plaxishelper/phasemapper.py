from typing import Any, List, Optional
from ..components.phase import ConstructionStage
from ..core.plaxisobject import PlaxisObject

class PhaseMapper:
    """Creates and configures construction phases, including activation/deactivation of objects."""

    @classmethod
    def create_phases(cls, g_i: Any, phases: List[ConstructionStage]) -> None:
        """
        Creates construction phases in Plaxis and applies activation/deactivation of objects and excavation.

        Args:
            g_i (Any): The Plaxis global input object.
            phases (List[ConstructionStage]): List of construction stage objects to create.
        """
        print("Mapping Construction Phases...")
        if not phases:
            return  # No phases to create

        previous_plaxis_phase = g_i.Phases[0]  # Reference to the initial phase

        for py_phase in phases:
            try:
                # 1) Create new phase
                new_plaxis_phase = g_i.phase(previous_plaxis_phase)
                if new_plaxis_phase is None:
                    print(f"  - WARNING: Failed to create phase '{py_phase.name}' in Plaxis.")
                    continue

                new_plaxis_phase.Name = py_phase.name

                # 2) Set calculate parameters of the phase
                for setting_name, setting_value in py_phase.settings.settings.items():
                    if hasattr(new_plaxis_phase, setting_name):
                        setattr(new_plaxis_phase, setting_name, setting_value)

                # 3) Activate / Deactivate Object —— Pass in g_i here to avoid insecure access like getattr(phase, obj_ref)
                cls._set_phase_activity(g_i, new_plaxis_phase, py_phase._activate, True)
                cls._set_phase_activity(g_i, new_plaxis_phase, py_phase._deactivate, False)

                # 4) Excavation treatment (removal of soil)
                for soil_block in getattr(py_phase, "_excavate", []):
                    obj = getattr(soil_block, "plx_id", None)
                    if obj is not None:
                        try:
                            # Excavation treatment (removal of soil)
                            g_i.deactivate(obj, new_plaxis_phase)
                            print(f"    - Excavation: Deactivated '{soil_block.name}' in phase '{py_phase.name}'.")
                        except Exception:
                            # Rollback: Directly set "Active" (some objects/versions may not support this)
                            try:
                                obj.Active = False
                                print(f"    - Excavation (fallback): Deactivated '{soil_block.name}' in phase '{py_phase.name}'.")
                            except Exception:
                                print(f"    - WARNING: Could not deactivate soil block '{soil_block.name}' in phase '{py_phase.name}'.")
                    else:
                        print(f"    - WARNING: Soil block '{soil_block.name}' has no plx_id; cannot excavate in phase '{py_phase.name}'.")

                # 5) Reference of the object in the return stage
                py_phase.plx_id = new_plaxis_phase
                if py_phase.plx_id is None:
                    print(f"  - ERROR: Phase '{py_phase.name}' plx_id not assigned!")
                else:
                    print(f"  - Phase '{py_phase.name}' created.")
                previous_plaxis_phase = new_plaxis_phase

            except Exception as e:
                print(f"  - ERROR creating phase '{py_phase.name}': {e}")

    @classmethod
    def _set_phase_activity(
        cls,
        g_i: Any,
        plaxis_phase: Any,
        object_list: List[Any],
        active_state: bool
    ) -> None:
        """
        Helper to activate or deactivate a list of objects in a given phase.

        Args:
            g_i (Any): The Plaxis global input object.
            plaxis_phase (Any): The Plaxis phase object.
            object_list (List[Any]): List of model-layer objects to set active state.
            active_state (bool): True to activate objects, False to deactivate.
        """
        action_word = "activate" if active_state else "deactivate"

        for item in object_list:
            # 1) Firstly, use plx_id (object reference); secondly, parse from g_i by name.
            obj = getattr(item, "plx_id", None)
            if obj is None:
                # Backtrack: Attempt to parse under g_i using the name.
                name = getattr(item, "name", None)
                if isinstance(name, str) and hasattr(g_i, name):
                    try:
                        obj = getattr(g_i, name)
                    except Exception:
                        obj = None

            if obj is None:
                human_name = getattr(item, "name", str(item))
                print(f"    - WARNING: Cannot {action_word} '{human_name}' in phase '{plaxis_phase.Name}': object not found.")
                continue

            # 2) Give priority to using the Plaxis API: g_i.activate/g_i.deactivate
            try:
                if active_state:
                    g_i.activate(obj, plaxis_phase)
                else:
                    g_i.deactivate(obj, plaxis_phase)
                continue  
            except Exception:
                # 3) Rollback: Directly set the Active property (some objects/versions may allow this)
                try:
                    obj.Active = active_state
                except Exception:
                    # 4) Finally, as a fallback: Attempt to set the value by name within the "phase" scope (to avoid passing object references to getattr)
                    try:
                        name_in_plaxis = getattr(obj, "Name", getattr(item, "name", None))
                        if isinstance(name_in_plaxis, str):
                            getattr(plaxis_phase, name_in_plaxis).Active = active_state
                        else:
                            raise AttributeError("No valid name for phase-scope attribute access")
                    except Exception:
                        human_name = getattr(item, "name", str(item))
                        print(f"    - WARNING: Failed to {action_word} '{human_name}' in phase '{plaxis_phase.Name}'.")
