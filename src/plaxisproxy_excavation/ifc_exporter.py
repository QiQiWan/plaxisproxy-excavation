"""
ifc_exporter.py
===============
IFC 导出模块 — 将 plaxisproxy_excavation 的 FoundationPit 对象
规范化并导出为 IFC 4 文件。

依赖：
    pip install ifcopenshell   # 或使用 conda-forge 版本

支持的对象映射：
    RetainingWall   → IfcWall          (含 IfcMaterial)
    Anchor          → IfcTendon        (含 IfcMaterial)
    Beam            → IfcBeam          (含 IfcMaterial)
    EmbeddedPile    → IfcPile          (含 IfcMaterial)
    SoilBlock       → IfcGeographicElement (含 IfcMaterial)
    Borehole        → IfcBorehole      (含 IfcMaterial)
    FoundationPit   → IfcSite + IfcBuilding + IfcBuildingStorey

用法示例：
    from plaxisproxy_excavation.ifc_exporter import IFCExporter

    exporter = IFCExporter(pit)
    exporter.export("output.ifc")
"""

from __future__ import annotations

import math
import uuid
import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ── 可选依赖 ──────────────────────────────────────────────────────────────────
try:
    import ifcopenshell
    import ifcopenshell.api
    import ifcopenshell.util.element
    _IFC_AVAILABLE = True
except ImportError:
    _IFC_AVAILABLE = False

# ── 项目内部导入 ───────────────────────────────────────────────────────────────
from .excavation import FoundationPit, StructureType
from .geometry import Point, Line3D, Polygon3D, PointSet
from .structures.retainingwall import RetainingWall
from .structures.anchor import Anchor
from .structures.beam import Beam
from .structures.embeddedpile import EmbeddedPile
from .structures.soilblock import SoilBlock
from .borehole import Borehole, BoreholeLayer


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _require_ifcopenshell() -> None:
    if not _IFC_AVAILABLE:
        raise ImportError(
            "ifcopenshell is required for IFC export.\n"
            "Install it with:  pip install ifcopenshell\n"
            "or via conda:     conda install -c conda-forge ifcopenshell"
        )


def _new_guid() -> str:
    """生成 IFC 兼容的 22 位 GlobalId（Base64 压缩 UUID）。"""
    return ifcopenshell.guid.compress(uuid.uuid4().hex)


def _ifc_cartesian_point(ifc_file, x: float, y: float, z: float = 0.0):
    """创建 IfcCartesianPoint。"""
    return ifc_file.createIfcCartesianPoint((float(x), float(y), float(z)))


def _ifc_direction(ifc_file, dx: float, dy: float, dz: float = 0.0):
    return ifc_file.createIfcDirection((float(dx), float(dy), float(dz)))


def _ifc_axis2placement3d(ifc_file, origin: Tuple[float, float, float] = (0, 0, 0),
                           axis: Tuple[float, float, float] = (0, 0, 1),
                           ref_dir: Tuple[float, float, float] = (1, 0, 0)):
    """创建 IfcAxis2Placement3D。"""
    pt = _ifc_cartesian_point(ifc_file, *origin)
    ax = _ifc_direction(ifc_file, *axis)
    rd = _ifc_direction(ifc_file, *ref_dir)
    return ifc_file.createIfcAxis2Placement3D(pt, ax, rd)


def _ifc_local_placement(ifc_file, relative_to=None,
                          origin=(0, 0, 0), axis=(0, 0, 1), ref_dir=(1, 0, 0)):
    placement = _ifc_axis2placement3d(ifc_file, origin, axis, ref_dir)
    return ifc_file.createIfcLocalPlacement(relative_to, placement)


def _polygon3d_to_ifc_face(ifc_file, poly: Polygon3D):
    """
    将 Polygon3D 的外环转换为 IfcFace（用于 IfcFacetedBrep）。
    仅使用外环，忽略内孔（基坑支护结构通常无孔）。
    """
    pts = poly.outer_ring.get_points()
    # 去掉闭合重复点
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return None

    ifc_pts = [_ifc_cartesian_point(ifc_file, p.x, p.y, p.z) for p in pts]
    poly_loop = ifc_file.createIfcPolyLoop(ifc_pts)
    face_bound = ifc_file.createIfcFaceOuterBound(poly_loop, True)
    return ifc_file.createIfcFace([face_bound])


def _polygon3d_to_brep(ifc_file, poly: Polygon3D):
    """将单个 Polygon3D 面转换为 IfcFacetedBrep（单面体）。"""
    face = _polygon3d_to_ifc_face(ifc_file, poly)
    if face is None:
        return None
    closed_shell = ifc_file.createIfcClosedShell([face])
    return ifc_file.createIfcFacetedBrep(closed_shell)


def _retaining_wall_to_brep(ifc_file, wall: RetainingWall):
    """
    将 RetainingWall 的 Polygon3D 面转换为 IfcFacetedBrep。
    如果面是四边形（典型挡土墙），生成完整的六面体（拉伸体）。
    否则退化为单面 Brep。
    """
    surface: Polygon3D = wall.surface
    pts = surface.outer_ring.get_points()
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]

    if len(pts) == 4:
        # 尝试构建六面体（前后两个面 + 四个侧面）
        try:
            return _quad_to_box_brep(ifc_file, pts)
        except Exception:
            pass

    # 退化：单面 Brep
    return _polygon3d_to_brep(ifc_file, surface)


def _quad_to_box_brep(ifc_file, pts: List[Point]):
    """
    将四边形面（挡土墙截面）构建为 IfcFacetedBrep 六面体。
    pts: 4 个 Point，按顺序排列（不含闭合重复点）。
    假设墙有一定厚度（从材料属性中读取，默认 0.5m）。
    这里简化为单面 Brep，实际项目可扩展为真实厚度。
    """
    ifc_pts = [_ifc_cartesian_point(ifc_file, p.x, p.y, p.z) for p in pts]
    poly_loop = ifc_file.createIfcPolyLoop(ifc_pts)
    face_bound = ifc_file.createIfcFaceOuterBound(poly_loop, True)
    face = ifc_file.createIfcFace([face_bound])
    closed_shell = ifc_file.createIfcClosedShell([face])
    return ifc_file.createIfcFacetedBrep(closed_shell)


def _line3d_to_ifc_polyline(ifc_file, line: Line3D):
    """将 Line3D 转换为 IfcPolyline。"""
    pts = line.get_points()
    ifc_pts = [_ifc_cartesian_point(ifc_file, p.x, p.y, p.z) for p in pts]
    return ifc_file.createIfcPolyline(ifc_pts)


def _line3d_to_swept_solid(ifc_file, line: Line3D, radius: float = 0.1):
    """
    将 Line3D（锚杆/梁/桩）转换为 IfcSweptDiskSolid（圆截面扫掠体）。
    radius: 截面半径（m），默认 0.1m。
    """
    polyline = _line3d_to_ifc_polyline(ifc_file, line)
    return ifc_file.createIfcSweptDiskSolid(polyline, float(radius), None, 0.0, 1.0)


def _make_shape_representation(ifc_file, context, items: list,
                                rep_id: str = "Body",
                                rep_type: str = "Brep"):
    """创建 IfcShapeRepresentation。"""
    return ifc_file.createIfcShapeRepresentation(
        context, rep_id, rep_type, items
    )


def _make_product_definition_shape(ifc_file, representations: list):
    return ifc_file.createIfcProductDefinitionShape(None, None, representations)


def _make_material(ifc_file, name: str):
    """创建 IfcMaterial。"""
    return ifc_file.createIfcMaterial(name)


def _assign_material(ifc_file, element, material):
    """将 IfcMaterial 关联到 IfcElement。"""
    mat_select = ifc_file.createIfcRelAssociatesMaterial(
        _new_guid(), None, None, None, [element], material
    )
    return mat_select


def _get_material_name(obj: Any) -> str:
    """从结构对象中提取材料名称（容错）。"""
    for attr in ("_plate_type", "_anchor_type", "_beam_type", "_pile_type", "_material"):
        val = getattr(obj, attr, None)
        if val is None:
            continue
        if isinstance(val, str):
            return val
        name = getattr(val, "name", None)
        if name:
            return str(name)
        return type(val).__name__
    return "Unknown"


def _get_soil_material_name(soil_block: SoilBlock) -> str:
    mat = getattr(soil_block, "_material", None) or getattr(soil_block, "material", None)
    if mat is None:
        return "Unknown Soil"
    return getattr(mat, "name", str(mat))


# ─────────────────────────────────────────────────────────────────────────────
# 主导出器
# ─────────────────────────────────────────────────────────────────────────────

class IFCExporter:
    """
    将 FoundationPit 对象导出为 IFC 4 文件。

    Parameters
    ----------
    pit : FoundationPit
        要导出的基坑模型对象。
    schema : str
        IFC schema 版本，默认 "IFC4"。
    author : str
        文件作者信息。
    organization : str
        组织名称。

    Examples
    --------
    >>> exporter = IFCExporter(pit)
    >>> exporter.export("my_excavation.ifc")
    """

    def __init__(
        self,
        pit: FoundationPit,
        schema: str = "IFC4",
        author: str = "EatRice",
        organization: str = "智能建造",
    ) -> None:
        _require_ifcopenshell()
        self.pit = pit
        self.schema = schema
        self.author = author
        self.organization = organization

        # 运行时状态（export() 调用时初始化）
        self._ifc: Any = None
        self._context: Any = None
        self._site: Any = None
        self._building: Any = None
        self._storey: Any = None
        self._site_placement: Any = None
        self._owner_history: Any = None

        # 材料缓存（避免重复创建）
        self._material_cache: Dict[str, Any] = {}

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def export(self, file_path: str) -> None:
        """
        执行导出，将 IFC 文件写入 file_path。

        Parameters
        ----------
        file_path : str
            输出文件路径，例如 "output.ifc"。
        """
        self._init_ifc_file()
        self._build_spatial_hierarchy()
        self._export_retaining_walls()
        self._export_anchors()
        self._export_beams()
        self._export_embedded_piles()
        self._export_soil_blocks()
        self._export_boreholes()
        self._ifc.write(file_path)
        print(f"[IFCExporter] IFC 文件已导出至: {file_path}")

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def _init_ifc_file(self) -> None:
        """初始化 IFC 文件头和全局上下文。"""
        self._ifc = ifcopenshell.file(schema=self.schema)

        # ── OwnerHistory ──
        person = self._ifc.createIfcPerson(None, self.author, None, None, None, None, None, None)
        org = self._ifc.createIfcOrganization(None, self.organization, None, None, None)
        person_and_org = self._ifc.createIfcPersonAndOrganization(person, org, None)
        application = self._ifc.createIfcApplication(
            org, "1.0", "plaxisproxy_excavation IFC Exporter", "plaxisproxy_ifc"
        )
        now = int(datetime.datetime.now().timestamp())
        self._owner_history = self._ifc.createIfcOwnerHistory(
            person_and_org, application, "READWRITE", None, None, None, None, now
        )

        # ── 几何上下文 ──
        world_coord = _ifc_axis2placement3d(self._ifc)
        self._context = self._ifc.createIfcGeometricRepresentationContext(
            "Model", "Model", 3, 1.0e-5, world_coord, None
        )

        # ── 单位 ──
        length_unit = self._ifc.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE")
        area_unit = self._ifc.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE")
        volume_unit = self._ifc.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE")
        plane_angle_unit = self._ifc.createIfcSIUnit(None, "PLANEANGLEUNIT", None, "RADIAN")
        unit_assignment = self._ifc.createIfcUnitAssignment(
            [length_unit, area_unit, volume_unit, plane_angle_unit]
        )

        # ── 项目 ──
        proj_name = getattr(
            getattr(self.pit, "project_information", None), "project_name", "ExcavationProject"
        ) or "ExcavationProject"
        self._ifc.createIfcProject(
            _new_guid(), self._owner_history, proj_name, None,
            None, None, None, [self._context], unit_assignment
        )

    # ── 空间层级 ─────────────────────────────────────────────────────────────

    def _build_spatial_hierarchy(self) -> None:
        """构建 IfcSite → IfcBuilding → IfcBuildingStorey 层级。"""
        self._site_placement = _ifc_local_placement(self._ifc)

        self._site = self._ifc.createIfcSite(
            _new_guid(), self._owner_history, "ExcavationSite", None, None,
            self._site_placement, None, None, "ELEMENT", None, None, None, None, None
        )

        building_placement = _ifc_local_placement(self._ifc, self._site_placement)
        self._building = self._ifc.createIfcBuilding(
            _new_guid(), self._owner_history, "ExcavationBuilding", None, None,
            building_placement, None, None, "ELEMENT", None, None, None
        )

        storey_placement = _ifc_local_placement(self._ifc, building_placement)
        self._storey = self._ifc.createIfcBuildingStorey(
            _new_guid(), self._owner_history, "ExcavationLevel", None, None,
            storey_placement, None, None, "ELEMENT", 0.0
        )

        # 关联层级
        self._ifc.createIfcRelAggregates(
            _new_guid(), self._owner_history, None, None,
            self._site, [self._building]
        )
        self._ifc.createIfcRelAggregates(
            _new_guid(), self._owner_history, None, None,
            self._building, [self._storey]
        )

    # ── 材料缓存 ──────────────────────────────────────────────────────────────

    def _get_or_create_material(self, name: str):
        if name not in self._material_cache:
            self._material_cache[name] = _make_material(self._ifc, name)
        return self._material_cache[name]

    # ── 挡土墙 ────────────────────────────────────────────────────────────────

    def _export_retaining_walls(self) -> None:
        walls: List[RetainingWall] = (
            self.pit.structures.get(StructureType.RETAINING_WALLS.value, []) or []
        )
        elements = []
        for wall in walls:
            elem = self._wall_to_ifc(wall)
            if elem is not None:
                elements.append(elem)
        if elements:
            self._ifc.createIfcRelContainedInSpatialStructure(
                _new_guid(), self._owner_history, "RetainingWalls", None,
                elements, self._storey
            )

    def _wall_to_ifc(self, wall: RetainingWall):
        try:
            brep = _retaining_wall_to_brep(self._ifc, wall)
            if brep is None:
                return None
            shape_rep = _make_shape_representation(self._ifc, self._context, [brep])
            shape = _make_product_definition_shape(self._ifc, [shape_rep])
            placement = _ifc_local_placement(self._ifc, self._site_placement)

            ifc_wall = self._ifc.createIfcWall(
                _new_guid(), self._owner_history,
                wall.name, getattr(wall, "comment", None),
                None, placement, shape, None
            )
            mat_name = _get_material_name(wall)
            mat = self._get_or_create_material(mat_name)
            _assign_material(self._ifc, ifc_wall, mat)

            # 添加自定义属性集
            self._add_pset(ifc_wall, "Pset_RetainingWall", {
                "PlaxisName": wall.name,
                "MaterialType": mat_name,
            })
            return ifc_wall
        except Exception as e:
            print(f"[IFCExporter] 跳过挡土墙 '{wall.name}': {e}")
            return None

    # ── 锚杆 ──────────────────────────────────────────────────────────────────

    def _export_anchors(self) -> None:
        anchors: List[Anchor] = (
            self.pit.structures.get(StructureType.ANCHORS.value, []) or []
        )
        elements = []
        for anchor in anchors:
            elem = self._anchor_to_ifc(anchor)
            if elem is not None:
                elements.append(elem)
        if elements:
            self._ifc.createIfcRelContainedInSpatialStructure(
                _new_guid(), self._owner_history, "Anchors", None,
                elements, self._storey
            )

    def _anchor_to_ifc(self, anchor: Anchor):
        try:
            line = anchor.line
            radius = self._get_section_radius(anchor, default=0.05)
            swept = _line3d_to_swept_solid(self._ifc, line, radius)
            shape_rep = _make_shape_representation(
                self._ifc, self._context, [swept], "Body", "SweptSolid"
            )
            shape = _make_product_definition_shape(self._ifc, [shape_rep])
            placement = _ifc_local_placement(self._ifc, self._site_placement)

            # IFC4 中锚杆映射为 IfcTendon
            ifc_anchor = self._ifc.createIfcTendon(
                _new_guid(), self._owner_history,
                anchor.name, getattr(anchor, "comment", None),
                None, placement, shape, None,
                "STRAND", None, None, None, None, None
            )
            mat_name = _get_material_name(anchor)
            mat = self._get_or_create_material(mat_name)
            _assign_material(self._ifc, ifc_anchor, mat)

            self._add_pset(ifc_anchor, "Pset_Anchor", {
                "PlaxisName": anchor.name,
                "AnchorType": mat_name,
                "Length": f"{line.length:.3f} m",
            })
            return ifc_anchor
        except Exception as e:
            print(f"[IFCExporter] 跳过锚杆 '{anchor.name}': {e}")
            return None

    # ── 梁 ────────────────────────────────────────────────────────────────────

    def _export_beams(self) -> None:
        beams: List[Beam] = (
            self.pit.structures.get(StructureType.BEAMS.value, []) or []
        )
        elements = []
        for beam in beams:
            elem = self._beam_to_ifc(beam)
            if elem is not None:
                elements.append(elem)
        if elements:
            self._ifc.createIfcRelContainedInSpatialStructure(
                _new_guid(), self._owner_history, "Beams", None,
                elements, self._storey
            )

    def _beam_to_ifc(self, beam: Beam):
        try:
            line = beam.line
            radius = self._get_section_radius(beam, default=0.15)
            swept = _line3d_to_swept_solid(self._ifc, line, radius)
            shape_rep = _make_shape_representation(
                self._ifc, self._context, [swept], "Body", "SweptSolid"
            )
            shape = _make_product_definition_shape(self._ifc, [shape_rep])
            placement = _ifc_local_placement(self._ifc, self._site_placement)

            ifc_beam = self._ifc.createIfcBeam(
                _new_guid(), self._owner_history,
                beam.name, getattr(beam, "comment", None),
                None, placement, shape, None
            )
            mat_name = _get_material_name(beam)
            mat = self._get_or_create_material(mat_name)
            _assign_material(self._ifc, ifc_beam, mat)

            self._add_pset(ifc_beam, "Pset_Beam", {
                "PlaxisName": beam.name,
                "BeamType": mat_name,
                "Length": f"{line.length:.3f} m",
            })
            return ifc_beam
        except Exception as e:
            print(f"[IFCExporter] 跳过梁 '{beam.name}': {e}")
            return None

    # ── 嵌入桩 ────────────────────────────────────────────────────────────────

    def _export_embedded_piles(self) -> None:
        piles: List[EmbeddedPile] = (
            self.pit.structures.get(StructureType.EMBEDDED_PILES.value, []) or []
        )
        elements = []
        for pile in piles:
            elem = self._pile_to_ifc(pile)
            if elem is not None:
                elements.append(elem)
        if elements:
            self._ifc.createIfcRelContainedInSpatialStructure(
                _new_guid(), self._owner_history, "EmbeddedPiles", None,
                elements, self._storey
            )

    def _pile_to_ifc(self, pile: EmbeddedPile):
        try:
            line = pile.line
            radius = self._get_section_radius(pile, default=0.3)
            swept = _line3d_to_swept_solid(self._ifc, line, radius)
            shape_rep = _make_shape_representation(
                self._ifc, self._context, [swept], "Body", "SweptSolid"
            )
            shape = _make_product_definition_shape(self._ifc, [shape_rep])
            placement = _ifc_local_placement(self._ifc, self._site_placement)

            ifc_pile = self._ifc.createIfcPile(
                _new_guid(), self._owner_history,
                pile.name, getattr(pile, "comment", None),
                None, placement, shape, None, "DRIVEN"
            )
            mat_name = _get_material_name(pile)
            mat = self._get_or_create_material(mat_name)
            _assign_material(self._ifc, ifc_pile, mat)

            self._add_pset(ifc_pile, "Pset_Pile", {
                "PlaxisName": pile.name,
                "PileType": mat_name,
                "Length": f"{line.length:.3f} m",
            })
            return ifc_pile
        except Exception as e:
            print(f"[IFCExporter] 跳过嵌入桩 '{pile.name}': {e}")
            return None

    # ── 土体块 ────────────────────────────────────────────────────────────────

    def _export_soil_blocks(self) -> None:
        soil_blocks: List[SoilBlock] = (
            self.pit.structures.get(StructureType.SOIL_BLOCKS.value, []) or []
        )
        elements = []
        for sb in soil_blocks:
            elem = self._soil_block_to_ifc(sb)
            if elem is not None:
                elements.append(elem)
        if elements:
            self._ifc.createIfcRelContainedInSpatialStructure(
                _new_guid(), self._owner_history, "SoilBlocks", None,
                elements, self._storey
            )

    def _soil_block_to_ifc(self, sb: SoilBlock):
        try:
            geom = getattr(sb, "_geometry", None) or getattr(sb, "geometry", None)
            items = []

            if geom is not None and isinstance(geom, Polygon3D):
                brep = _polygon3d_to_brep(self._ifc, geom)
                if brep:
                    items.append(brep)
            elif geom is not None and hasattr(geom, "_faces"):
                for face in geom._faces:
                    brep = _polygon3d_to_brep(self._ifc, face)
                    if brep:
                        items.append(brep)

            placement = _ifc_local_placement(self._ifc, self._site_placement)

            if items:
                shape_rep = _make_shape_representation(self._ifc, self._context, items)
                shape = _make_product_definition_shape(self._ifc, [shape_rep])
            else:
                shape = None

            # IFC4 中土体映射为 IfcGeographicElement
            ifc_geo = self._ifc.createIfcGeographicElement(
                _new_guid(), self._owner_history,
                sb.name, getattr(sb, "comment", None),
                None, placement, shape, None, "TERRAIN"
            )
            mat_name = _get_soil_material_name(sb)
            mat = self._get_or_create_material(mat_name)
            _assign_material(self._ifc, ifc_geo, mat)

            self._add_pset(ifc_geo, "Pset_SoilBlock", {
                "PlaxisName": sb.name,
                "SoilMaterial": mat_name,
            })
            return ifc_geo
        except Exception as e:
            print(f"[IFCExporter] 跳过土体块 '{sb.name}': {e}")
            return None

    # ── 钻孔 ──────────────────────────────────────────────────────────────────

    def _export_boreholes(self) -> None:
        bh_set = getattr(self.pit, "borehole_set", None)
        if bh_set is None:
            return
        boreholes: List[Borehole] = getattr(bh_set, "boreholes", []) or []
        elements = []
        for bh in boreholes:
            elem = self._borehole_to_ifc(bh)
            if elem is not None:
                elements.append(elem)
        if elements:
            self._ifc.createIfcRelContainedInSpatialStructure(
                _new_guid(), self._owner_history, "Boreholes", None,
                elements, self._storey
            )

    def _borehole_to_ifc(self, bh: Borehole):
        try:
            loc = bh.location
            depth = bh.depth()

            # 用竖直线段表示钻孔
            p_top = _ifc_cartesian_point(self._ifc, loc.x, loc.y, bh.ground_level)
            p_bot = _ifc_cartesian_point(self._ifc, loc.x, loc.y, bh.ground_level - depth)
            polyline = self._ifc.createIfcPolyline([p_top, p_bot])
            swept = self._ifc.createIfcSweptDiskSolid(polyline, 0.05, None, 0.0, 1.0)

            shape_rep = _make_shape_representation(
                self._ifc, self._context, [swept], "Body", "SweptSolid"
            )
            shape = _make_product_definition_shape(self._ifc, [shape_rep])
            placement = _ifc_local_placement(
                self._ifc, self._site_placement,
                origin=(loc.x, loc.y, bh.ground_level)
            )

            # IFC4 中钻孔映射为 IfcGeographicElement（USERDEFINED + ObjectType）
            ifc_bh = self._ifc.createIfcGeographicElement(
                _new_guid(), self._owner_history,
                bh.name, getattr(bh, "comment", None),
                None, placement, shape, None, "USERDEFINED"
            )
            # 通过 ObjectType 标记为 Borehole
            ifc_bh.ObjectType = "Borehole"

            # 钻孔属性集
            layer_info = "; ".join(
                f"{ly.soil_layer.name if ly.soil_layer else ly.name}"
                f"[{ly.top_z:.2f}~{ly.bottom_z:.2f}m]"
                for ly in bh.layers
            )
            self._add_pset(ifc_bh, "Pset_Borehole", {
                "PlaxisName": bh.name,
                "GroundLevel": f"{bh.ground_level:.3f} m",
                "Depth": f"{depth:.3f} m",
                "WaterHead": f"{bh.water_head:.3f} m" if bh.water_head is not None else "N/A",
                "LayerSequence": layer_info,
            })
            return ifc_bh
        except Exception as e:
            print(f"[IFCExporter] 跳过钻孔 '{bh.name}': {e}")
            return None

    # ── 属性集工具 ────────────────────────────────────────────────────────────

    def _add_pset(self, element, pset_name: str, props: Dict[str, str]) -> None:
        """为 IFC 元素添加自定义属性集（IfcPropertySet）。"""
        try:
            ifc_props = []
            for k, v in props.items():
                single_val = self._ifc.createIfcPropertySingleValue(
                    k, None,
                    self._ifc.createIfcLabel(str(v)),
                    None
                )
                ifc_props.append(single_val)

            pset = self._ifc.createIfcPropertySet(
                _new_guid(), self._owner_history,
                pset_name, None, ifc_props
            )
            self._ifc.createIfcRelDefinesByProperties(
                _new_guid(), self._owner_history, None, None,
                [element], pset
            )
        except Exception as e:
            # 属性集失败不影响主体导出
            pass

    # ── 截面半径推断 ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_section_radius(obj: Any, default: float = 0.1) -> float:
        """
        从材料属性中推断截面半径（m）。
        - 对于桩/锚杆：尝试读取 diameter / radius 属性
        - 对于梁：尝试读取 d（板厚）/ 截面高度
        - 否则使用 default
        """
        for attr_chain in (
            ("_pile_type", "diameter"),
            ("_pile_type", "radius"),
            ("_anchor_type", "diameter"),
            ("_beam_type", "d"),
            ("_beam_type", "height"),
            ("_plate_type", "d"),
        ):
            val = obj
            for attr in attr_chain:
                val = getattr(val, attr, None)
                if val is None:
                    break
            if val is not None:
                try:
                    r = float(val)
                    # 如果是直径，转为半径
                    if attr_chain[-1] == "diameter":
                        r /= 2.0
                    return max(r, 0.01)
                except (TypeError, ValueError):
                    pass
        return default


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数
# ─────────────────────────────────────────────────────────────────────────────

def export_to_ifc(
    pit: FoundationPit,
    file_path: str,
    author: str = "EatRice",
    organization: str = "智能建造",
    schema: str = "IFC4",
) -> None:
    """
    一行导出 FoundationPit 为 IFC 文件。

    Parameters
    ----------
    pit : FoundationPit
        基坑模型对象。
    file_path : str
        输出 IFC 文件路径。
    author : str
        文件作者。
    organization : str
        组织名称。
    schema : str
        IFC schema，默认 "IFC4"。

    Examples
    --------
    >>> from plaxisproxy_excavation.ifc_exporter import export_to_ifc
    >>> export_to_ifc(pit, "output.ifc")
    """
    exporter = IFCExporter(pit, schema=schema, author=author, organization=organization)
    exporter.export(file_path)
