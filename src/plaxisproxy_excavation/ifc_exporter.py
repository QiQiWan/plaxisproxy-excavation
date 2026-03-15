"""
ifc_exporter.py  v2.0
=====================
将 plaxisproxy_excavation 的 FoundationPit 对象导出为规范 IFC 4 文件。

v2.0 改动（相对 v1.0）：
  - RetainingWall  → 真实厚度六面体（沿法向偏移 plate_type.d）
  - Beam           → 矩形/正方形截面实体柱（ExtrudedAreaSolid）
  - Anchor         → 圆截面扫掠体（SweptDiskSolid，半径从材料推断）
  - EmbeddedPile   → 圆截面扫掠体（直径从 pile_type.diameter 读取）
  - SoilBlock      → IfcGeographicElement（保留）
  - Borehole       → IfcGeographicElement + 完整地层属性集
  - 完整元数据：IfcSite 含经纬度/地址，IfcBuilding 含项目信息
  - 每个构件附带详细 IfcPropertySet

依赖：
    pip install ifcopenshell

用法：
    from plaxisproxy_excavation.ifc_exporter import IFCExporter, export_to_ifc
    export_to_ifc(pit, "output.ifc")
"""

from __future__ import annotations

import math
import uuid
import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import ifcopenshell
    import ifcopenshell.api
    _IFC_AVAILABLE = True
except ImportError:
    _IFC_AVAILABLE = False

from .excavation import FoundationPit, StructureType
from .geometry import Point, Line3D, Polygon3D, PointSet
from .structures.retainingwall import RetainingWall
from .structures.anchor import Anchor
from .structures.beam import Beam
from .structures.embeddedpile import EmbeddedPile
from .structures.soilblock import SoilBlock
from .borehole import Borehole, BoreholeLayer


# ─────────────────────────────────────────────────────────────────────────────
# 基础工具
# ─────────────────────────────────────────────────────────────────────────────

def _require_ifcopenshell() -> None:
    if not _IFC_AVAILABLE:
        raise ImportError(
            "ifcopenshell is required.\n"
            "Install: pip install ifcopenshell"
        )


def _new_guid() -> str:
    return ifcopenshell.guid.compress(uuid.uuid4().hex)


def _pt(f, x, y, z=0.0):
    return f.createIfcCartesianPoint((float(x), float(y), float(z)))


def _dir(f, dx, dy, dz=0.0):
    return f.createIfcDirection((float(dx), float(dy), float(dz)))


def _axis2p3d(f, origin=(0, 0, 0), axis=(0, 0, 1), ref=(1, 0, 0)):
    return f.createIfcAxis2Placement3D(
        _pt(f, *origin), _dir(f, *axis), _dir(f, *ref)
    )


def _local_placement(f, relative_to=None, origin=(0, 0, 0),
                     axis=(0, 0, 1), ref=(1, 0, 0)):
    return f.createIfcLocalPlacement(relative_to, _axis2p3d(f, origin, axis, ref))


def _shape_rep(f, ctx, items, rep_id="Body", rep_type="SweptSolid"):
    return f.createIfcShapeRepresentation(ctx, rep_id, rep_type, items)


def _product_shape(f, reps):
    return f.createIfcProductDefinitionShape(None, None, reps)


def _material(f, name: str):
    return f.createIfcMaterial(name)


def _assign_material(f, element, mat):
    f.createIfcRelAssociatesMaterial(
        _new_guid(), None, None, None, [element], mat
    )


def _pset(f, owner, element, pset_name: str, props: Dict[str, str]):
    """为构件添加 IfcPropertySet。"""
    try:
        ifc_props = [
            f.createIfcPropertySingleValue(k, None, f.createIfcLabel(str(v)), None)
            for k, v in props.items()
        ]
        ps = f.createIfcPropertySet(_new_guid(), owner, pset_name, None, ifc_props)
        f.createIfcRelDefinesByProperties(
            _new_guid(), owner, None, None, [element], ps
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 几何构建工具
# ─────────────────────────────────────────────────────────────────────────────

def _vec3_sub(a: Point, b: Point) -> Tuple[float, float, float]:
    return (a.x - b.x, a.y - b.y, a.z - b.z)


def _vec3_cross(u, v) -> Tuple[float, float, float]:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _vec3_norm(u) -> float:
    return math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)


def _vec3_normalize(u) -> Tuple[float, float, float]:
    n = _vec3_norm(u)
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (u[0] / n, u[1] / n, u[2] / n)


def _vec3_scale(u, s) -> Tuple[float, float, float]:
    return (u[0] * s, u[1] * s, u[2] * s)


def _vec3_add(u, v) -> Tuple[float, float, float]:
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])


def _polygon3d_normal(pts: List[Point]) -> Tuple[float, float, float]:
    """计算多边形法向量（Newell 法，适用于任意平面多边形）。"""
    n = len(pts)
    nx = ny = nz = 0.0
    for i in range(n):
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        nx += (cur.y - nxt.y) * (cur.z + nxt.z)
        ny += (cur.z - nxt.z) * (cur.x + nxt.x)
        nz += (cur.x - nxt.x) * (cur.y + nxt.y)
    return _vec3_normalize((nx, ny, nz))


def _build_wall_solid(f, ctx, pts: List[Point], thickness: float):
    """
    将挡土墙面（任意平面四边形/多边形）沿法向偏移 thickness，
    构建真实厚度的 IfcFacetedBrep 六面体。

    pts: 外环顶点（不含闭合重复点），至少 3 个。
    thickness: 板厚（m），沿法向向内偏移。
    """
    n = len(pts)
    if n < 3:
        return None

    normal = _polygon3d_normal(pts)
    offset = _vec3_scale(normal, thickness)

    # 前面（原始面）和后面（偏移面）
    front = pts
    back = [
        Point(p.x + offset[0], p.y + offset[1], p.z + offset[2])
        for p in pts
    ]

    def make_face(ring_pts, reverse=False):
        if reverse:
            ring_pts = list(reversed(ring_pts))
        ifc_pts = [_pt(f, p.x, p.y, p.z) for p in ring_pts]
        loop = f.createIfcPolyLoop(ifc_pts)
        bound = f.createIfcFaceOuterBound(loop, True)
        return f.createIfcFace([bound])

    faces = []
    # 前面（法向朝外）
    faces.append(make_face(front, reverse=False))
    # 后面（法向朝外，需反转）
    faces.append(make_face(back, reverse=True))
    # 四条侧面
    for i in range(n):
        j = (i + 1) % n
        side_pts = [front[i], front[j], back[j], back[i]]
        faces.append(make_face(side_pts))

    shell = f.createIfcClosedShell(faces)
    brep = f.createIfcFacetedBrep(shell)
    return _shape_rep(f, ctx, [brep], "Body", "Brep")


def _build_beam_solid(f, ctx, p_start: Point, p_end: Point,
                      width: float, height: float):
    """
    沿梁轴线方向生成矩形截面实体（IfcExtrudedAreaSolid）。

    坐标系：
      - 挤出方向 = 梁轴向单位向量
      - 截面局部 X = 与轴向垂直的水平方向
      - 截面局部 Y = 与轴向和局部X都垂直的方向
    截面：以轴线起点为原点，width × height 矩形，居中。
    """
    dx = p_end.x - p_start.x
    dy = p_end.y - p_start.y
    dz = p_end.z - p_start.z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-9:
        return None

    # 挤出方向（局部 Z）
    extrude_dir = (dx / length, dy / length, dz / length)

    # 局部 X：与挤出方向垂直的水平方向
    if abs(extrude_dir[2]) < 0.9:
        # 轴线不接近竖直：用全局 Z 叉乘轴向
        local_x = _vec3_normalize(_vec3_cross((0, 0, 1), extrude_dir))
    else:
        # 轴线接近竖直：用全局 X 叉乘轴向
        local_x = _vec3_normalize(_vec3_cross((1, 0, 0), extrude_dir))

    # 局部 Y = 挤出方向 × 局部X
    local_y = _vec3_normalize(_vec3_cross(extrude_dir, local_x))

    # 截面矩形（居中，在局部 XY 平面内）
    hw = width / 2.0
    hh = height / 2.0
    rect_pts_2d = [
        f.createIfcCartesianPoint((-hw, -hh)),
        f.createIfcCartesianPoint(( hw, -hh)),
        f.createIfcCartesianPoint(( hw,  hh)),
        f.createIfcCartesianPoint((-hw,  hh)),
    ]
    polyline_2d = f.createIfcPolyline(
        rect_pts_2d + [rect_pts_2d[0]]  # 闭合
    )
    profile = f.createIfcArbitraryClosedProfileDef("AREA", None, polyline_2d)

    # 截面坐标系（放置在 p_start，局部 X/Y 如上）
    origin_3d = _pt(f, p_start.x, p_start.y, p_start.z)
    axis_3d = _dir(f, *extrude_dir)   # 局部 Z（挤出方向）
    ref_3d = _dir(f, *local_x)        # 局部 X
    placement_3d = f.createIfcAxis2Placement3D(origin_3d, axis_3d, ref_3d)

    extrude_vec = _dir(f, 0.0, 0.0, 1.0)  # 在截面局部坐标系中沿 Z 挤出
    solid = f.createIfcExtrudedAreaSolid(profile, placement_3d, extrude_vec, length)
    return _shape_rep(f, ctx, [solid], "Body", "SweptSolid")


def _build_circle_swept(f, ctx, p_start: Point, p_end: Point, radius: float):
    """圆截面扫掠体（锚杆/桩）。"""
    ifc_pts = [_pt(f, p_start.x, p_start.y, p_start.z),
               _pt(f, p_end.x, p_end.y, p_end.z)]
    polyline = f.createIfcPolyline(ifc_pts)
    solid = f.createIfcSweptDiskSolid(polyline, float(radius), None, 0.0, 1.0)
    return _shape_rep(f, ctx, [solid], "Body", "SweptSolid")


def _get_beam_section(obj: Any) -> Tuple[float, float]:
    """
    从 ElasticBeam 材料中推断矩形截面尺寸 (width, height)。
    优先读 width/height，其次 diameter（正方形），最后默认 0.3×0.3。
    """
    mat = getattr(obj, "_beam_type", None) or getattr(obj, "beam_type", None)
    if mat is None:
        return 0.3, 0.3
    w = getattr(mat, "_width", None) or getattr(mat, "width", None)
    h = getattr(mat, "_height", None) or getattr(mat, "height", None)
    d = getattr(mat, "_diameter", None) or getattr(mat, "diameter", None)
    if w and h:
        return float(w), float(h)
    if d:
        return float(d), float(d)
    return 0.3, 0.3


def _get_pile_radius(obj: Any) -> float:
    """从 EmbeddedPile 材料中读取桩半径（m）。"""
    for chain in (("_pile_type", "_diameter"), ("_pile_type", "diameter")):
        v = obj
        for attr in chain:
            v = getattr(v, attr, None)
            if v is None:
                break
        if v is not None:
            try:
                return float(v) / 2.0
            except (TypeError, ValueError):
                pass
    return 0.3


def _get_anchor_radius(obj: Any) -> float:
    """从 Anchor 材料中读取锚杆半径（m）。"""
    for chain in (("_anchor_type", "diameter"), ("_anchor_type", "_diameter")):
        v = obj
        for attr in chain:
            v = getattr(v, attr, None)
            if v is None:
                break
        if v is not None:
            try:
                return float(v) / 2.0
            except (TypeError, ValueError):
                pass
    return 0.05


def _get_wall_thickness(wall: RetainingWall) -> float:
    """从 RetainingWall.plate_type 读取板厚 d（m）。"""
    pt = getattr(wall, "_plate_type", None) or getattr(wall, "plate_type", None)
    if pt is None:
        return 0.8
    d = getattr(pt, "_d", None) or getattr(pt, "d", None)
    if d is not None:
        try:
            return max(float(d), 0.01)
        except (TypeError, ValueError):
            pass
    return 0.8


def _get_material_name(obj: Any) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# 主导出器
# ─────────────────────────────────────────────────────────────────────────────

class IFCExporter:
    """
    将 FoundationPit 导出为 IFC 4 文件（v2.0，真实三维几何）。

    Parameters
    ----------
    pit : FoundationPit
    schema : str        IFC schema，默认 "IFC4"
    author : str
    organization : str
    project_address : str   项目地址（写入 IfcPostalAddress）
    latitude : float        纬度（度，写入 IfcSite）
    longitude : float       经度（度，写入 IfcSite）
    elevation : float       场地高程（m）

    Examples
    --------
    >>> exporter = IFCExporter(pit, author="EatRice",
    ...     project_address="上海市浦东新区XX路1号",
    ...     latitude=31.23, longitude=121.47)
    >>> exporter.export("output.ifc")
    """

    def __init__(
        self,
        pit: FoundationPit,
        schema: str = "IFC4",
        author: str = "EatRice",
        organization: str = "智能建造",
        project_address: str = "",
        latitude: float = 0.0,
        longitude: float = 0.0,
        elevation: float = 0.0,
    ) -> None:
        _require_ifcopenshell()
        self.pit = pit
        self.schema = schema
        self.author = author
        self.organization = organization
        self.project_address = project_address
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation

        self._ifc: Any = None
        self._ctx: Any = None
        self._owner: Any = None
        self._site: Any = None
        self._building: Any = None
        self._storey: Any = None
        self._site_pl: Any = None
        self._mat_cache: Dict[str, Any] = {}

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def export(self, file_path: str) -> None:
        """导出 IFC 文件到 file_path。"""
        self._init_file()
        self._build_hierarchy()
        self._export_retaining_walls()
        self._export_anchors()
        self._export_beams()
        self._export_embedded_piles()
        self._export_soil_blocks()
        self._export_boreholes()
        self._ifc.write(file_path)
        print(f"[IFCExporter v2.0] 导出完成 → {file_path}")

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def _init_file(self) -> None:
        self._ifc = ifcopenshell.file(schema=self.schema)
        f = self._ifc

        person = f.createIfcPerson(None, self.author, None, None, None, None, None, None)
        org = f.createIfcOrganization(None, self.organization, None, None, None)
        pao = f.createIfcPersonAndOrganization(person, org, None)
        app = f.createIfcApplication(org, "2.0", "plaxisproxy_excavation IFC Exporter v2", "plaxisproxy_ifc")
        now = int(datetime.datetime.now().timestamp())
        self._owner = f.createIfcOwnerHistory(pao, app, "READWRITE", None, None, None, None, now)

        world = f.createIfcAxis2Placement3D(
            _pt(f, 0, 0, 0), _dir(f, 0, 0, 1), _dir(f, 1, 0, 0)
        )
        self._ctx = f.createIfcGeometricRepresentationContext(
            "Model", "Model", 3, 1.0e-5, world, None
        )

        lu = f.createIfcSIUnit(None, "LENGTHUNIT", None, "METRE")
        au = f.createIfcSIUnit(None, "AREAUNIT", None, "SQUARE_METRE")
        vu = f.createIfcSIUnit(None, "VOLUMEUNIT", None, "CUBIC_METRE")
        pu = f.createIfcSIUnit(None, "PLANEANGLEUNIT", None, "RADIAN")
        units = f.createIfcUnitAssignment([lu, au, vu, pu])

        proj_info = getattr(self.pit, "project_information", None)
        proj_name = getattr(proj_info, "title", None) or getattr(proj_info, "project_name", None) or "ExcavationProject"
        f.createIfcProject(
            _new_guid(), self._owner, proj_name, None,
            None, None, None, [self._ctx], units
        )

    # ── 空间层级（含完整元数据）────────────────────────────────────────────────

    def _build_hierarchy(self) -> None:
        f = self._ifc
        proj_info = getattr(self.pit, "project_information", None)

        # ── IfcPostalAddress ──
        addr = None
        if self.project_address:
            addr = f.createIfcPostalAddress(
                "OFFICE", None, None, None,
                [self.project_address], None, None, None, None, "CN"
            )

        # ── IfcSite（含经纬度/高程）──
        def _deg_to_compound(deg: float):
            """度 → IFC IfcCompoundPlaneAngleMeasure (度, 分, 秒, 微秒)"""
            sign = 1 if deg >= 0 else -1
            deg = abs(deg)
            d = int(deg)
            m = int((deg - d) * 60)
            s = int(((deg - d) * 60 - m) * 60)
            us = int((((deg - d) * 60 - m) * 60 - s) * 1e6)
            return (sign * d, sign * m, sign * s, sign * us)

        lat = _deg_to_compound(self.latitude) if self.latitude else None
        lon = _deg_to_compound(self.longitude) if self.longitude else None

        self._site_pl = _local_placement(f)
        self._site = f.createIfcSite(
            _new_guid(), self._owner,
            "ExcavationSite",
            getattr(proj_info, "comment", None) or "基坑工程场地",
            None,
            self._site_pl, None, None,
            "ELEMENT",
            lat, lon,
            float(self.elevation) if self.elevation else None,
            None, addr
        )

        # ── IfcBuilding（含项目信息属性集）──
        bld_pl = _local_placement(f, self._site_pl)
        self._building = f.createIfcBuilding(
            _new_guid(), self._owner,
            getattr(proj_info, "title", None) or "ExcavationBuilding",
            None, None,
            bld_pl, None, None,
            "ELEMENT", None, None, addr
        )

        # 项目信息属性集
        if proj_info is not None:
            _pset(f, self._owner, self._building, "Pset_ProjectInformation", {
                "ProjectTitle": getattr(proj_info, "title", "") or "",
                "Company": getattr(proj_info, "company", "") or "",
                "Model": getattr(proj_info, "model", "") or "",
                "Element": getattr(proj_info, "element", "") or "",
                "GammaWater": f"{getattr(proj_info, 'gamma_water', 9.81):.3f} kN/m³",
                "ExcavationDepth": f"{abs(getattr(self.pit, 'excava_depth', 0) or 0):.2f} m",
            })

        # ── IfcBuildingStorey ──
        st_pl = _local_placement(f, bld_pl)
        self._storey = f.createIfcBuildingStorey(
            _new_guid(), self._owner,
            "ExcavationLevel", None, None,
            st_pl, None, None,
            "ELEMENT", float(self.elevation)
        )

        # 层级关联
        f.createIfcRelAggregates(_new_guid(), self._owner, None, None, self._site, [self._building])
        f.createIfcRelAggregates(_new_guid(), self._owner, None, None, self._building, [self._storey])

    # ── 材料缓存 ──────────────────────────────────────────────────────────────

    def _mat(self, name: str):
        if name not in self._mat_cache:
            self._mat_cache[name] = _material(self._ifc, name)
        return self._mat_cache[name]

    def _contain(self, elements: list, label: str) -> None:
        if elements:
            self._ifc.createIfcRelContainedInSpatialStructure(
                _new_guid(), self._owner, label, None,
                elements, self._storey
            )

    # ── 挡土墙（真实厚度六面体）──────────────────────────────────────────────

    def _export_retaining_walls(self) -> None:
        walls: List[RetainingWall] = (
            self.pit.structures.get(StructureType.RETAINING_WALLS.value, []) or []
        )
        elems = []
        for wall in walls:
            e = self._wall_to_ifc(wall)
            if e is not None:
                elems.append(e)
        self._contain(elems, "RetainingWalls")

    def _wall_to_ifc(self, wall: RetainingWall):
        try:
            surface: Polygon3D = wall.surface
            pts = surface._ring_core_points(surface._lines[0])
            thickness = _get_wall_thickness(wall)

            rep = _build_wall_solid(self._ifc, self._ctx, pts, thickness)
            if rep is None:
                return None

            placement = _local_placement(self._ifc, self._site_pl)
            ifc_wall = self._ifc.createIfcWall(
                _new_guid(), self._owner,
                wall.name, getattr(wall, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None
            )
            mat_name = _get_material_name(wall)
            _assign_material(self._ifc, ifc_wall, self._mat(mat_name))

            pt = getattr(wall, "_plate_type", None) or getattr(wall, "plate_type", None)
            _pset(self._ifc, self._owner, ifc_wall, "Pset_RetainingWall", {
                "PlaxisName": wall.name,
                "MaterialName": mat_name,
                "Thickness_m": f"{thickness:.3f}",
                "E_kPa": str(getattr(pt, "E", "") or ""),
                "nu": str(getattr(pt, "nu", "") or ""),
                "gamma_kNm3": str(getattr(pt, "gamma", "") or ""),
            })
            return ifc_wall
        except Exception as e:
            print(f"[IFCExporter] ⚠ 跳过挡土墙 '{wall.name}': {e}")
            return None

    # ── 锚杆（圆截面扫掠）────────────────────────────────────────────────────

    def _export_anchors(self) -> None:
        anchors: List[Anchor] = (
            self.pit.structures.get(StructureType.ANCHORS.value, []) or []
        )
        elems = []
        for anchor in anchors:
            e = self._anchor_to_ifc(anchor)
            if e is not None:
                elems.append(e)
        self._contain(elems, "Anchors")

    def _anchor_to_ifc(self, anchor: Anchor):
        try:
            pts = anchor.line.get_points()
            p0, p1 = pts[0], pts[1]
            radius = _get_anchor_radius(anchor)

            rep = _build_circle_swept(self._ifc, self._ctx, p0, p1, radius)
            placement = _local_placement(self._ifc, self._site_pl)
            ifc_anchor = self._ifc.createIfcTendon(
                _new_guid(), self._owner,
                anchor.name, getattr(anchor, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None,
                "STRAND", None, None, None, None, None
            )
            mat_name = _get_material_name(anchor)
            _assign_material(self._ifc, ifc_anchor, self._mat(mat_name))

            at = getattr(anchor, "_anchor_type", None)
            length = math.sqrt(
                (p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2 + (p1.z - p0.z) ** 2
            )
            _pset(self._ifc, self._owner, ifc_anchor, "Pset_Anchor", {
                "PlaxisName": anchor.name,
                "MaterialName": mat_name,
                "Radius_m": f"{radius:.4f}",
                "Length_m": f"{length:.3f}",
                "EA_kN": str(getattr(at, "EA", "") or ""),
            })
            return ifc_anchor
        except Exception as e:
            print(f"[IFCExporter] ⚠ 跳过锚杆 '{anchor.name}': {e}")
            return None

    # ── 梁（矩形截面实体柱）──────────────────────────────────────────────────

    def _export_beams(self) -> None:
        beams: List[Beam] = (
            self.pit.structures.get(StructureType.BEAMS.value, []) or []
        )
        elems = []
        for beam in beams:
            e = self._beam_to_ifc(beam)
            if e is not None:
                elems.append(e)
        self._contain(elems, "Beams")

    def _beam_to_ifc(self, beam: Beam):
        try:
            pts = beam.line.get_points()
            p0, p1 = pts[0], pts[1]
            width, height = _get_beam_section(beam)

            rep = _build_beam_solid(self._ifc, self._ctx, p0, p1, width, height)
            if rep is None:
                return None

            placement = _local_placement(self._ifc, self._site_pl)
            ifc_beam = self._ifc.createIfcBeam(
                _new_guid(), self._owner,
                beam.name, getattr(beam, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None
            )
            mat_name = _get_material_name(beam)
            _assign_material(self._ifc, ifc_beam, self._mat(mat_name))

            bt = getattr(beam, "_beam_type", None)
            length = math.sqrt(
                (p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2 + (p1.z - p0.z) ** 2
            )
            _pset(self._ifc, self._owner, ifc_beam, "Pset_Beam", {
                "PlaxisName": beam.name,
                "MaterialName": mat_name,
                "SectionWidth_m": f"{width:.3f}",
                "SectionHeight_m": f"{height:.3f}",
                "Length_m": f"{length:.3f}",
                "E_kPa": str(getattr(bt, "E", "") or ""),
                "gamma_kNm3": str(getattr(bt, "gamma", "") or ""),
            })
            return ifc_beam
        except Exception as e:
            print(f"[IFCExporter] ⚠ 跳过梁 '{beam.name}': {e}")
            return None

    # ── 嵌入桩（圆截面扫掠）──────────────────────────────────────────────────

    def _export_embedded_piles(self) -> None:
        piles: List[EmbeddedPile] = (
            self.pit.structures.get(StructureType.EMBEDDED_PILES.value, []) or []
        )
        elems = []
        for pile in piles:
            e = self._pile_to_ifc(pile)
            if e is not None:
                elems.append(e)
        self._contain(elems, "EmbeddedPiles")

    def _pile_to_ifc(self, pile: EmbeddedPile):
        try:
            pts = pile.line.get_points()
            p0, p1 = pts[0], pts[1]
            radius = _get_pile_radius(pile)

            rep = _build_circle_swept(self._ifc, self._ctx, p0, p1, radius)
            placement = _local_placement(self._ifc, self._site_pl)
            ifc_pile = self._ifc.createIfcPile(
                _new_guid(), self._owner,
                pile.name, getattr(pile, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None, "DRIVEN"
            )
            mat_name = _get_material_name(pile)
            _assign_material(self._ifc, ifc_pile, self._mat(mat_name))

            pt = getattr(pile, "_pile_type", None)
            length = math.sqrt(
                (p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2 + (p1.z - p0.z) ** 2
            )
            _pset(self._ifc, self._owner, ifc_pile, "Pset_Pile", {
                "PlaxisName": pile.name,
                "MaterialName": mat_name,
                "Diameter_m": f"{radius * 2:.3f}",
                "Length_m": f"{length:.3f}",
                "E_kPa": str(getattr(pt, "E", "") or ""),
                "gamma_kNm3": str(getattr(pt, "gamma", "") or ""),
            })
            return ifc_pile
        except Exception as e:
            print(f"[IFCExporter] ⚠ 跳过嵌入桩 '{pile.name}': {e}")
            return None

    # ── 土体块 ────────────────────────────────────────────────────────────────

    def _export_soil_blocks(self) -> None:
        soil_blocks: List[SoilBlock] = (
            self.pit.structures.get(StructureType.SOIL_BLOCKS.value, []) or []
        )
        elems = []
        for sb in soil_blocks:
            e = self._soil_block_to_ifc(sb)
            if e is not None:
                elems.append(e)
        self._contain(elems, "SoilBlocks")

    def _soil_block_to_ifc(self, sb: SoilBlock):
        try:
            placement = _local_placement(self._ifc, self._site_pl)
            ifc_geo = self._ifc.createIfcGeographicElement(
                _new_guid(), self._owner,
                sb.name, getattr(sb, "comment", None),
                None, placement, None, None, "TERRAIN"
            )
            mat_name = str(getattr(getattr(sb, "_material", None), "name", None) or "Unknown Soil")
            _assign_material(self._ifc, ifc_geo, self._mat(mat_name))
            _pset(self._ifc, self._owner, ifc_geo, "Pset_SoilBlock", {
                "PlaxisName": sb.name,
                "SoilMaterial": mat_name,
            })
            return ifc_geo
        except Exception as e:
            print(f"[IFCExporter] ⚠ 跳过土体块 '{sb.name}': {e}")
            return None

    # ── 钻孔（竖直圆柱 + 完整地层属性集）────────────────────────────────────

    def _export_boreholes(self) -> None:
        bh_set = getattr(self.pit, "borehole_set", None)
        if bh_set is None:
            return
        boreholes: List[Borehole] = getattr(bh_set, "boreholes", []) or []
        elems = []
        for bh in boreholes:
            e = self._borehole_to_ifc(bh)
            if e is not None:
                elems.append(e)
        self._contain(elems, "Boreholes")

    def _borehole_to_ifc(self, bh: Borehole):
        try:
            loc = bh.location
            depth = bh.depth()
            gl = bh.ground_level

            p_top = _pt(self._ifc, loc.x, loc.y, gl)
            p_bot = _pt(self._ifc, loc.x, loc.y, gl - depth)
            polyline = self._ifc.createIfcPolyline([p_top, p_bot])
            solid = self._ifc.createIfcSweptDiskSolid(polyline, 0.05, None, 0.0, 1.0)
            rep = _shape_rep(self._ifc, self._ctx, [solid])
            placement = _local_placement(self._ifc, self._site_pl, origin=(loc.x, loc.y, gl))

            ifc_bh = self._ifc.createIfcGeographicElement(
                _new_guid(), self._owner,
                bh.name, getattr(bh, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None, "USERDEFINED"
            )
            ifc_bh.ObjectType = "Borehole"

            # 地层信息
            layer_info = "; ".join(
                f"{getattr(getattr(ly, 'soil_layer', None), 'name', None) or ly.name}"
                f"[{ly.top_z:.2f}~{ly.bottom_z:.2f}m]"
                for ly in bh.layers
            )
            wh = bh.water_head
            _pset(self._ifc, self._owner, ifc_bh, "Pset_Borehole", {
                "PlaxisName": bh.name,
                "GroundLevel_m": f"{gl:.3f}",
                "Depth_m": f"{depth:.3f}",
                "WaterHead_m": f"{wh:.3f}" if wh is not None else "N/A",
                "LayerCount": str(len(bh.layers)),
                "LayerSequence": layer_info,
            })
            return ifc_bh
        except Exception as e:
            print(f"[IFCExporter] ⚠ 跳过钻孔 '{bh.name}': {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数
# ─────────────────────────────────────────────────────────────────────────────

def export_to_ifc(
    pit: FoundationPit,
    file_path: str,
    author: str = "EatRice",
    organization: str = "智能建造",
    schema: str = "IFC4",
    project_address: str = "",
    latitude: float = 0.0,
    longitude: float = 0.0,
    elevation: float = 0.0,
) -> None:
    """
    一行导出 FoundationPit 为 IFC 文件（v2.0）。

    Examples
    --------
    >>> export_to_ifc(pit, "output.ifc",
    ...     project_address="上海市浦东新区XX路1号",
    ...     latitude=31.23, longitude=121.47)
    """
    IFCExporter(
        pit, schema=schema, author=author, organization=organization,
        project_address=project_address,
        latitude=latitude, longitude=longitude, elevation=elevation,
    ).export(file_path)
