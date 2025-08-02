from __future__ import annotations

"""plx.remote – thin OO wrapper for PLAXIS 3D Remote Scripting API
===================================================================
This module *translates* high‑level domain objects (\*FoundationPit, Retaining
Wall, Mesh …) into PLAXIS Remote Scripting commands and executes them through
the official *plxscripting* socket interface.

Key design goals
----------------
1. **Zero‑boilerplate** – user scripts call domain objects; mapping layer handles
   creation of geometry / materials / phases automatically.
2. **Lazy command queue** – commands are batched and flushed with *session.sync()*
   for performance.
3. **ID registry** – every :class:`basestructure.BaseStructure` carries a
   ``_plx_id`` that stores the handle returned by the g_i object.  Registry
   prevents duplicate creation and enables updates.
4. **Composable mappers** – each domain class owns a corresponding *Mapper*
   responsible for *create / update / delete*.

Usage example
-------------
>>> from geometry import PointSet, Polygon3D
>>> from plx.remote import PlaxisSession, FoundationPitMapper
>>> sess = PlaxisSession("localhost", 10000, password="abc")
>>> pit = my_foundation_pit_factory()   # returns PlaxisFoundationPit instance
>>> FoundationPitMapper(sess).create(pit)
>>> sess.sync()   # flush → PLAXIS 3D executes all commands
"""

import contextlib
import logging
from typing import Dict, List, Any, Optional

import plxscripting.easy as plx  # official PLAXIS helper – pip install plxscripting

from structures.basestructure import BaseStructure
from plaxis_foundationpit import PlaxisFoundationPit
from structures.retainingwall import RetainingWall
from components.mesh import Mesh
from geometry import Polygon3D

__all__ = [
    "PlaxisSession",
    "BaseMapper",
    "FoundationPitMapper",
    "RetainingWallMapper",
]

LOGGER = logging.getLogger("plx.remote")

# ----------------------------------------------------------------------------
# 1. Session / command queue --------------------------------------------------
# ----------------------------------------------------------------------------

class PlaxisSession:
    """Encapsulates g_i / s_i handles & a deferred command queue."""

    def __init__(self, host: str, port: int, password: str | None = None):
        self.host = host
        self.port = port
        self.password = password or ""

        # connect -----------------------------------------------------------------
        self.s_i, self.g_i = plx.new_server(port, password=password) if host == "localhost" \
            else plx.connect_to_remote(host, port, password)

        self.cmd_buffer: List[str] = []  # raw command strings
        self.object_registry: Dict[int, BaseStructure] = {}  # plx_id → object

    # ---------------- command helpers ------------------------------------------
    def queue(self, cmd: str) -> None:
        self.cmd_buffer.append(cmd)
        LOGGER.debug(cmd)

    def eval(self, cmd: str):
        """Immediate evaluation – avoid when batching large jobs."""
        LOGGER.debug(cmd)
        return self.s_i.calc(cmd)

    def sync(self):
        """Flush queued commands in a single multi‑line script."""
        if not self.cmd_buffer:
            return
        script = "\n".join(self.cmd_buffer)
        self.s_i.calc(script)
        self.cmd_buffer.clear()

    # ---------------- registry helpers -----------------------------------------
    def register(self, obj: BaseStructure, plx_handle):
        obj._plx_id = plx_handle
        self.object_registry[id(obj)] = obj

    def is_registered(self, obj: BaseStructure) -> bool:
        return obj._plx_id is not None

    # ---------------- context manager -----------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # auto‑flush & close socket if no exception occurred
        if exc_type is None:
            self.sync()
        self.s_i.close()

# ----------------------------------------------------------------------------
# 2. BaseMapper – parent for all translators ----------------------------------
# ----------------------------------------------------------------------------

class BaseMapper:
    """Abstract translator between *domain object* and *PLAXIS commands*."""

    def __init__(self, session: PlaxisSession):
        self.sess = session
        self.q = session.queue  # shorthand
        self.g = session.g_i    # geometry side handle

    # these should be overridden -------------------------------------------------
    def create(self, obj: BaseStructure):
        raise NotImplementedError

    def update(self, obj: BaseStructure):
        raise NotImplementedError

    # helper: ensure object only created once ------------------------------------
    def _require_new(self, obj: BaseStructure):
        if self.sess.is_registered(obj):
            raise RuntimeError(f"{obj} already registered with PLAXIS (id={obj._plx_id})")

# ----------------------------------------------------------------------------
# 3. Concrete mappers ----------------------------------------------------------
# ----------------------------------------------------------------------------

class FoundationPitMapper(BaseMapper):
    """Create geometry surfaces + excavation phase for a PlaxisFoundationPit."""

    def create(self, pit: PlaxisFoundationPit):
        self._require_new(pit)

        # 1) create soil volume box ------------------------------------------------
        self.q("soil_volume = g_i.soilvolumes.create()  # placeholder")
        self.q("# TODO: set correct dimensions based on pit footprint + buffer")

        # 2) create excavation region surface -------------------------------------
        outer_pts_cmd = ", ".join([f"({p.x},{p.y},0)" for p in pit.footprint.get_lines()[0].get_points()])
        self.q(f"pit_polygon = g_i.Polygon({outer_pts_cmd})")
        self.q("pit_excavation = g_i.soilvolumes.create(pit_polygon, 0, -{:.3f})".format(pit.depth))

        # register & store handle (fake handle here; real handle captured via _gev())
        self.sess.register(pit, plx_handle="pit_excavation")

        # 3) loop retaining walls --------------------------------------------------
        wall_mapper = RetainingWallMapper(self.sess)
        for wall in pit.retaining_walls:
            wall_mapper.create(wall)

        # 4) optional: mesh settings ----------------------------------------------
        if pit.mesh:
            # Example: g_i.mesh.generate(...)
            self.q("g_i.mesh_1.CoarsenessFactor = {}".format(pit.mesh.coarseness))

class RetainingWallMapper(BaseMapper):
    """Translate RetainingWall → plate/extrusion in PLAXIS."""

    def create(self, wall: RetainingWall):
        self._require_new(wall)

        poly: Polygon3D = getattr(wall, "surface", None) or getattr(wall, "outline")
        pts = poly.get_lines()[0].get_points()
        cmd_pts = ", ".join([f"({p.x},{p.y},{p.z})" for p in pts])

        self.q(f"wall_poly = g_i.Polygon({cmd_pts})")
        self.q("wall_plate = g_i.plates.create(wall_poly)")
        self.q("wall_plate.Thickness = {:.3f}".format(wall.thickness))
        self.q("wall_plate.Material = globals()['{}']".format(wall.material_name))

        self.sess.register(wall, plx_handle="wall_plate")

    # TODO: update() for geometry/material changes
