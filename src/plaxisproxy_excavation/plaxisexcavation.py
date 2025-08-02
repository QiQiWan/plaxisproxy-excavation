from __future__ import annotations

"""High‑level wrapper representing a deep‑excavation (foundation pit) system.

Revision 2025‑07‑30
-------------------
*   Added ``ProjectInformation`` linkage so each pit is aware of its parent
    metadata block.
*   Updated import paths to reflect new package layout (components/, structures/).
*   ``finalize()`` validates ring closure **and** presence of project info.
"""

from typing import List, Optional, Dict, Any
import uuid

# --- updated import map -----------------------------------------------------
from geometry import * 
from components.mesh import Mesh
from components.watertable import WaterLevelTable
from components.projectinformation import ProjectInformation
from structures.retainingwall import RetainingWall
from structures.anchor import Anchor
from structures.embeddedpile import EmbeddedPile
from structures.well import Well
from borehole import BoreholeSet
from components.phase import ConstructionStage
from structures.basestructure import BaseStructure


class PlaxisFoundationPit(BaseStructure):
    pass