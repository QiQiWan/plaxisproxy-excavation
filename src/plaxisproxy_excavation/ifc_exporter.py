"""
ifc_exporter.py  v2.1
=====================
将 plaxisproxy_excavation 的 FoundationPit 对象导出为规范 IFC 4 文件。

v2.1 修复（相对 v2.0）：
  - 实心圆柱：桩/锚杆改用 IfcExtrudedAreaSolid + IfcCircleProfileDef
    （彻底解决 SweptDiskSolid InnerRadius=$ 被部分查看器渲染为空心的问题）
  - 材质颜色：每种材料绑定 IfcSurfaceStyle + IfcColourRgb，
    通过 IfcStyledItem 直接挂载到几何体，所有主流查看器均可显示彩色
  - 预设配色方案（可自定义）：
      混凝土地连墙 → 灰色  (0.75, 0.75, 0.75)
      型钢支撑     → 蓝灰  (0.40, 0.55, 0.75)
      锚杆         → 橙色  (0.90, 0.55, 0.20)
      嵌入桩       → 深灰  (0.50, 0.50, 0.50)
      土体/钻孔    → 棕色  (0.65, 0.50, 0.35)

依赖：
    pip install ifcopenshell

用法：
    from plaxisproxy_excavation.ifc_exporter import IFCExporter, export_to_ifc
    export_to_ifc(pit, "output.ifc",
                  project_address="上海市XX路1号",
                  latitude=31.23, longitude=121.47)
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
# 预设配色方案  RGB ∈ [0,1]
# ─────────────────────────────────────────────────────────────────────────────
COLOUR_PALETTE: Dict[str, Tuple[float, float, float]] = {
    # 构件类型 → (R, G, B)
    "wall":     (0.75, 0.75, 0.75),   # 混凝土地连墙：浅灰
    "beam":     (0.40, 0.55, 0.75),   # 型钢支撑：蓝灰
    "anchor":   (0.90, 0.55, 0.20),   # 锚杆：橙色
    "pile":     (0.50, 0.50, 0.50),   # 嵌入桩：深灰
    "soil":     (0.65, 0.50, 0.35),   # 土体：棕色
    "borehole": (0.55, 0.40, 0.25),   # 钻孔：深棕
}

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


def _pt2d(f, x, y):
    return f.createIfcCartesianPoint((float(x), float(y)))


def _dir(f, dx, dy, dz=0.0):
    return f.createIfcDirection((float(dx), float(dy), float(dz)))


def _axis2p3d(f, origin=(0, 0, 0), axis=(0, 0, 1), ref=(1, 0, 0)):
    return f.createIfcAxis2Placement3D(
        _pt(f, *origin), _dir(f, *axis), _dir(f, *ref)
    )


def _axis2p2d(f, origin=(0, 0)):
    return f.createIfcAxis2Placement2D(_pt2d(f, *origin), None)


def _local_placement(f, relative_to=None, origin=(0, 0, 0),
                     axis=(0, 0, 1), ref=(1, 0, 0)):
    return f.createIfcLocalPlacement(relative_to, _axis2p3d(f, origin, axis, ref))


def _shape_rep(f, ctx, items, rep_id="Body", rep_type="SweptSolid"):
    return f.createIfcShapeRepresentation(ctx, rep_id, rep_type, items)


def _product_shape(f, reps):
    return f.createIfcProductDefinitionShape(None, None, reps)


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
# 颜色 / 材质绑定
# ─────────────────────────────────────────────────────────────────────────────

def _make_surface_style(f, style_name: str, rgb: Tuple[float, float, float],
                        transparency: float = 0.0):
    """
    创建 IfcSurfaceStyle（含 IfcSurfaceStyleRendering + IfcColourRgb）。
    返回 IfcSurfaceStyle 实体。
    """
    r, g, b = rgb
    colour = f.createIfcColourRgb(None, float(r), float(g), float(b))
    rendering = f.createIfcSurfaceStyleRendering(
        colour,               # SurfaceColour
        float(transparency),  # Transparency (0=不透明, 1=全透明)
        None, None, None,     # DiffuseColour, TransmissionColour, DiffuseTransmissionColour
        None, None, None,     # ReflectionColour, SpecularColour, SpecularHighlight
        "FLAT",               # ReflectanceMethod
    )
    return f.createIfcSurfaceStyle(style_name, "BOTH", [rendering])


def _apply_colour(f, solid_item, surface_style):
    """
    将 IfcSurfaceStyle 通过 IfcStyledItem 绑定到几何体（solid_item）。
    这是让查看器显示颜色的关键步骤。
    """
    style_assign = f.createIfcPresentationStyleAssignment([surface_style])
    f.createIfcStyledItem(solid_item, [style_assign], None)


def _assign_ifc_material(f, element, mat_name: str):
    """将 IfcMaterial（语义）关联到构件。"""
    mat = f.createIfcMaterial(mat_name)
    f.createIfcRelAssociatesMaterial(
        _new_guid(), None, None, None, [element], mat
    )
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# 向量工具
# ─────────────────────────────────────────────────────────────────────────────

def _vsub(a: Point, b: Point):
    return (a.x - b.x, a.y - b.y, a.z - b.z)

def _vcross(u, v):
    return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])

def _vnorm(u):
    return math.sqrt(u[0]**2 + u[1]**2 + u[2]**2)

def _vnormalize(u):
    n = _vnorm(u)
    return (u[0]/n, u[1]/n, u[2]/n) if n > 1e-12 else (0., 0., 1.)

def _vscale(u, s):
    return (u[0]*s, u[1]*s, u[2]*s)

def _polygon3d_normal(pts: List[Point]):
    """Newell 法计算多边形法向量。"""
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        c, nx_ = pts[i], pts[(i+1) % n]
        nx += (c.y - nx_.y) * (c.z + nx_.z)
        ny += (c.z - nx_.z) * (c.x + nx_.x)
        nz += (c.x - nx_.x) * (c.y + nx_.y)
    return _vnormalize((nx, ny, nz))

# ─────────────────────────────────────────────────────────────────────────────
# 几何构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_wall_solid(f, ctx, pts: List[Point], thickness: float, surface_style=None):
    """
    地连墙：沿法向偏移 thickness，生成真实厚度六面体（IfcFacetedBrep）。
    surface_style: 若提供则对每个面的几何体绑定颜色。
    """
    if len(pts) < 3:
        return None
    normal = _polygon3d_normal(pts)
    off = _vscale(normal, thickness)
    front = pts
    back = [Point(p.x+off[0], p.y+off[1], p.z+off[2]) for p in pts]

    def make_face(ring, reverse=False):
        if reverse:
            ring = list(reversed(ring))
        ifc_pts = [_pt(f, p.x, p.y, p.z) for p in ring]
        loop = f.createIfcPolyLoop(ifc_pts)
        bound = f.createIfcFaceOuterBound(loop, True)
        return f.createIfcFace([bound])

    n = len(pts)
    faces = [make_face(front), make_face(back, reverse=True)]
    for i in range(n):
        j = (i+1) % n
        faces.append(make_face([front[i], front[j], back[j], back[i]]))

    shell = f.createIfcClosedShell(faces)
    brep = f.createIfcFacetedBrep(shell)
    if surface_style:
        _apply_colour(f, brep, surface_style)
    return _shape_rep(f, ctx, [brep], "Body", "Brep")


def _build_circle_solid(f, ctx, p_start: Point, p_end: Point,
                        radius: float, surface_style=None):
    """
    实心圆柱（桩/锚杆）：IfcExtrudedAreaSolid + IfcCircleProfileDef。
    相比 SweptDiskSolid，所有查看器均渲染为实心。
    """
    dx = p_end.x - p_start.x
    dy = p_end.y - p_start.y
    dz = p_end.z - p_start.z
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-9:
        return None

    extrude_dir = (dx/length, dy/length, dz/length)

    # 局部坐标系
    if abs(extrude_dir[2]) < 0.9:
        local_x = _vnormalize(_vcross((0, 0, 1), extrude_dir))
    else:
        local_x = _vnormalize(_vcross((1, 0, 0), extrude_dir))

    # 圆形截面（在局部 XY 平面，原点居中）
    profile = f.createIfcCircleProfileDef(
        "AREA", None,
        _axis2p2d(f, (0, 0)),
        float(radius)
    )

    # 截面放置在 p_start，挤出方向沿轴线
    placement = f.createIfcAxis2Placement3D(
        _pt(f, p_start.x, p_start.y, p_start.z),
        _dir(f, *extrude_dir),   # 局部 Z = 挤出方向
        _dir(f, *local_x),       # 局部 X
    )
    extrude_vec = _dir(f, 0., 0., 1.)  # 在截面局部坐标系中沿 Z 挤出
    solid = f.createIfcExtrudedAreaSolid(profile, placement, extrude_vec, float(length))

    if surface_style:
        _apply_colour(f, solid, surface_style)
    return _shape_rep(f, ctx, [solid], "Body", "SweptSolid")


def _build_rect_solid(f, ctx, p_start: Point, p_end: Point,
                      width: float, height: float, surface_style=None):
    """
    矩形截面实体柱（水平支撑）：IfcExtrudedAreaSolid + 矩形截面。
    """
    dx = p_end.x - p_start.x
    dy = p_end.y - p_start.y
    dz = p_end.z - p_start.z
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-9:
        return None

    extrude_dir = (dx/length, dy/length, dz/length)
    if abs(extrude_dir[2]) < 0.9:
        local_x = _vnormalize(_vcross((0, 0, 1), extrude_dir))
    else:
        local_x = _vnormalize(_vcross((1, 0, 0), extrude_dir))

    hw, hh = width/2.0, height/2.0
    rect_pts = [
        f.createIfcCartesianPoint((-hw, -hh)),
        f.createIfcCartesianPoint(( hw, -hh)),
        f.createIfcCartesianPoint(( hw,  hh)),
        f.createIfcCartesianPoint((-hw,  hh)),
    ]
    polyline_2d = f.createIfcPolyline(rect_pts + [rect_pts[0]])
    profile = f.createIfcArbitraryClosedProfileDef("AREA", None, polyline_2d)

    placement = f.createIfcAxis2Placement3D(
        _pt(f, p_start.x, p_start.y, p_start.z),
        _dir(f, *extrude_dir),
        _dir(f, *local_x),
    )
    extrude_vec = _dir(f, 0., 0., 1.)
    solid = f.createIfcExtrudedAreaSolid(profile, placement, extrude_vec, float(length))

    if surface_style:
        _apply_colour(f, solid, surface_style)
    return _shape_rep(f, ctx, [solid], "Body", "SweptSolid")


# ─────────────────────────────────────────────────────────────────────────────
# 材料属性读取工具
# ─────────────────────────────────────────────────────────────────────────────

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


def _get_wall_thickness(wall: RetainingWall) -> float:
    pt = getattr(wall, "_plate_type", None) or getattr(wall, "plate_type", None)
    if pt is None:
        return 0.8
    d = getattr(pt, "_d", None) or getattr(pt, "d", None)
    try:
        return max(float(d), 0.01) if d is not None else 0.8
    except (TypeError, ValueError):
        return 0.8


def _get_beam_section(obj: Any) -> Tuple[float, float]:
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


# ─────────────────────────────────────────────────────────────────────────────
# 主导出器
# ─────────────────────────────────────────────────────────────────────────────

class IFCExporter:
    """
    将 FoundationPit 导出为 IFC 4 文件（v2.1）。

    v2.1 新增：
      - 实心圆柱（ExtrudedAreaSolid + CircleProfileDef）
      - 每种构件类型绑定颜色（IfcSurfaceStyle + IfcStyledItem）
      - 颜色可通过 colour_overrides 参数自定义

    Parameters
    ----------
    pit              : FoundationPit
    schema           : IFC schema，默认 "IFC4"
    author           : 作者
    organization     : 组织
    project_address  : 项目地址（写入 IfcPostalAddress）
    latitude         : 纬度（度）
    longitude        : 经度（度）
    elevation        : 场地高程（m）
    colour_overrides : 覆盖默认配色，格式 {"wall": (R,G,B), ...}
                       键名：wall / beam / anchor / pile / soil / borehole
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
        colour_overrides: Optional[Dict[str, Tuple[float, float, float]]] = None,
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

        # 合并配色
        self._palette = dict(COLOUR_PALETTE)
        if colour_overrides:
            self._palette.update(colour_overrides)

        # 运行时状态
        self._ifc: Any = None
        self._ctx: Any = None
        self._owner: Any = None
        self._site: Any = None
        self._building: Any = None
        self._storey: Any = None
        self._site_pl: Any = None
        # 样式缓存（每种类型只创建一次）
        self._style_cache: Dict[str, Any] = {}

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def export(self, file_path: str) -> None:
        """导出 IFC 文件。"""
        print(f"[IFCExporter v2.1] 开始导出...")
        self._init_file()
        print(f"  ✓ IFC 文件初始化完成")
        self._build_hierarchy()
        print(f"  ✓ 空间层级构建完成（Site → Building → Storey）")
        self._export_retaining_walls()
        self._export_anchors()
        self._export_beams()
        self._export_embedded_piles()
        self._export_soil_blocks()
        self._export_boreholes()
        self._ifc.write(file_path)
        print(f"[IFCExporter v2.1] 导出完成 → {file_path}")

    # ── 样式缓存 ──────────────────────────────────────────────────────────────

    def _get_style(self, kind: str) -> Any:
        """按构件类型获取（或创建）IfcSurfaceStyle，带缓存。"""
        if kind not in self._style_cache:
            rgb = self._palette.get(kind, (0.7, 0.7, 0.7))
            self._style_cache[kind] = _make_surface_style(
                self._ifc, f"Style_{kind}", rgb
            )
        return self._style_cache[kind]

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def _init_file(self) -> None:
        f = self._ifc = ifcopenshell.file(schema=self.schema)
        person = f.createIfcPerson(None, self.author, None, None, None, None, None, None)
        org    = f.createIfcOrganization(None, self.organization, None, None, None)
        pao    = f.createIfcPersonAndOrganization(person, org, None)
        app    = f.createIfcApplication(org, "2.1",
                     "plaxisproxy_excavation IFC Exporter v2.1", "plaxisproxy_ifc")
        now = int(datetime.datetime.now().timestamp())
        self._owner = f.createIfcOwnerHistory(pao, app, "READWRITE", None, None, None, None, now)

        world = _axis2p3d(f)
        self._ctx = f.createIfcGeometricRepresentationContext(
            "Model", "Model", 3, 1.0e-5, world, None
        )
        lu = f.createIfcSIUnit(None, "LENGTHUNIT",    None, "METRE")
        au = f.createIfcSIUnit(None, "AREAUNIT",      None, "SQUARE_METRE")
        vu = f.createIfcSIUnit(None, "VOLUMEUNIT",    None, "CUBIC_METRE")
        pu = f.createIfcSIUnit(None, "PLANEANGLEUNIT",None, "RADIAN")
        units = f.createIfcUnitAssignment([lu, au, vu, pu])

        proj_info = getattr(self.pit, "project_information", None)
        proj_name = (getattr(proj_info, "title", None)
                     or getattr(proj_info, "project_name", None)
                     or "ExcavationProject")
        f.createIfcProject(
            _new_guid(), self._owner, proj_name, None,
            None, None, None, [self._ctx], units
        )

    # ── 空间层级 ──────────────────────────────────────────────────────────────

    def _build_hierarchy(self) -> None:
        f = self._ifc
        proj_info = getattr(self.pit, "project_information", None)

        addr = None
        if self.project_address:
            addr = f.createIfcPostalAddress(
                "OFFICE", None, None, None,
                [self.project_address], None, None, None, None, "CN"
            )

        def _deg_to_compound(deg: float):
            sign = 1 if deg >= 0 else -1
            deg = abs(deg)
            d = int(deg)
            m = int((deg - d) * 60)
            s = int(((deg - d) * 60 - m) * 60)
            us = int((((deg - d) * 60 - m) * 60 - s) * 1e6)
            return (sign*d, sign*m, sign*s, sign*us)

        lat = _deg_to_compound(self.latitude)  if self.latitude  else None
        lon = _deg_to_compound(self.longitude) if self.longitude else None

        self._site_pl = _local_placement(f)
        self._site = f.createIfcSite(
            _new_guid(), self._owner,
            "ExcavationSite",
            getattr(proj_info, "comment", None) or "基坑工程场地",
            None, self._site_pl, None, None, "ELEMENT",
            lat, lon,
            float(self.elevation) if self.elevation else None,
            None, addr
        )

        bld_pl = _local_placement(f, self._site_pl)
        self._building = f.createIfcBuilding(
            _new_guid(), self._owner,
            getattr(proj_info, "title", None) or "ExcavationBuilding",
            None, None, bld_pl, None, None, "ELEMENT", None, None, addr
        )
        if proj_info is not None:
            _pset(f, self._owner, self._building, "Pset_ProjectInformation", {
                "ProjectTitle":    getattr(proj_info, "title",       "") or "",
                "Company":         getattr(proj_info, "company",     "") or "",
                "Model":           getattr(proj_info, "model",       "") or "",
                "Element":         getattr(proj_info, "element",     "") or "",
                "GammaWater":      f"{getattr(proj_info, 'gamma_water', 9.81):.3f} kN/m³",
                "ExcavationDepth": f"{abs(getattr(self.pit, 'excava_depth', 0) or 0):.2f} m",
            })

        st_pl = _local_placement(f, bld_pl)
        self._storey = f.createIfcBuildingStorey(
            _new_guid(), self._owner,
            "ExcavationLevel", None, None,
            st_pl, None, None, "ELEMENT", float(self.elevation)
        )

        f.createIfcRelAggregates(_new_guid(), self._owner, None, None,
                                 self._site, [self._building])
        f.createIfcRelAggregates(_new_guid(), self._owner, None, None,
                                 self._building, [self._storey])

    def _contain(self, elements: list, label: str) -> None:
        if elements:
            self._ifc.createIfcRelContainedInSpatialStructure(
                _new_guid(), self._owner, label, None,
                elements, self._storey
            )


    # ── 挡土墙 ────────────────────────────────────────────────────────────────

    def _export_retaining_walls(self) -> None:
        walls = self.pit.structures.get(StructureType.RETAINING_WALLS.value, []) or []
        elems = []
        for wall in walls:
            e = self._wall_to_ifc(wall)
            if e is not None:
                elems.append(e)
        self._contain(elems, "RetainingWalls")
        print(f"  ✓ 地连墙：{len(elems)} 面（厚度从 plate_type.d 读取，灰色）")

    def _wall_to_ifc(self, wall: RetainingWall):
        try:
            pts = wall.surface._ring_core_points(wall.surface._lines[0])
            thickness = _get_wall_thickness(wall)
            style = self._get_style("wall")
            rep = _build_wall_solid(self._ifc, self._ctx, pts, thickness, style)
            if rep is None:
                return None
            placement = _local_placement(self._ifc, self._site_pl)
            ifc_wall = self._ifc.createIfcWall(
                _new_guid(), self._owner,
                wall.name, getattr(wall, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None
            )
            mat_name = _get_material_name(wall)
            _assign_ifc_material(self._ifc, ifc_wall, mat_name)
            pt = getattr(wall, "_plate_type", None)
            _pset(self._ifc, self._owner, ifc_wall, "Pset_RetainingWall", {
                "PlaxisName":   wall.name,
                "MaterialName": mat_name,
                "Thickness_m":  f"{thickness:.3f}",
                "E_kPa":        str(getattr(pt, "E",     "") or ""),
                "nu":           str(getattr(pt, "nu",    "") or ""),
                "gamma_kNm3":   str(getattr(pt, "gamma", "") or ""),
            })
            return ifc_wall
        except Exception as e:
            print(f"    ⚠ 跳过挡土墙 '{wall.name}': {e}")
            return None

    # ── 锚杆 ──────────────────────────────────────────────────────────────────

    def _export_anchors(self) -> None:
        anchors = self.pit.structures.get(StructureType.ANCHORS.value, []) or []
        elems = []
        for anchor in anchors:
            e = self._anchor_to_ifc(anchor)
            if e is not None:
                elems.append(e)
        self._contain(elems, "Anchors")
        print(f"  ✓ 锚杆：{len(elems)} 根（实心圆截面，橙色）")

    def _anchor_to_ifc(self, anchor: Anchor):
        try:
            pts = anchor.line.get_points()
            p0, p1 = pts[0], pts[1]
            radius = _get_anchor_radius(anchor)
            style = self._get_style("anchor")
            rep = _build_circle_solid(self._ifc, self._ctx, p0, p1, radius, style)
            placement = _local_placement(self._ifc, self._site_pl)
            ifc_anchor = self._ifc.createIfcTendon(
                _new_guid(), self._owner,
                anchor.name, getattr(anchor, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None,
                "STRAND", None, None, None, None, None
            )
            mat_name = _get_material_name(anchor)
            _assign_ifc_material(self._ifc, ifc_anchor, mat_name)
            at = getattr(anchor, "_anchor_type", None)
            length = math.sqrt((p1.x-p0.x)**2+(p1.y-p0.y)**2+(p1.z-p0.z)**2)
            _pset(self._ifc, self._owner, ifc_anchor, "Pset_Anchor", {
                "PlaxisName":   anchor.name,
                "MaterialName": mat_name,
                "Radius_m":     f"{radius:.4f}",
                "Length_m":     f"{length:.3f}",
                "EA_kN":        str(getattr(at, "EA", "") or ""),
            })
            return ifc_anchor
        except Exception as e:
            print(f"    ⚠ 跳过锚杆 '{anchor.name}': {e}")
            return None

    # ── 梁 ────────────────────────────────────────────────────────────────────

    def _export_beams(self) -> None:
        beams = self.pit.structures.get(StructureType.BEAMS.value, []) or []
        elems = []
        for beam in beams:
            e = self._beam_to_ifc(beam)
            if e is not None:
                elems.append(e)
        self._contain(elems, "Beams")
        print(f"  ✓ 水平支撑：{len(elems)} 根（矩形截面实体，蓝灰色）")

    def _beam_to_ifc(self, beam: Beam):
        try:
            pts = beam.line.get_points()
            p0, p1 = pts[0], pts[1]
            width, height = _get_beam_section(beam)
            style = self._get_style("beam")
            rep = _build_rect_solid(self._ifc, self._ctx, p0, p1, width, height, style)
            if rep is None:
                return None
            placement = _local_placement(self._ifc, self._site_pl)
            ifc_beam = self._ifc.createIfcBeam(
                _new_guid(), self._owner,
                beam.name, getattr(beam, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None
            )
            mat_name = _get_material_name(beam)
            _assign_ifc_material(self._ifc, ifc_beam, mat_name)
            bt = getattr(beam, "_beam_type", None)
            length = math.sqrt((p1.x-p0.x)**2+(p1.y-p0.y)**2+(p1.z-p0.z)**2)
            _pset(self._ifc, self._owner, ifc_beam, "Pset_Beam", {
                "PlaxisName":      beam.name,
                "MaterialName":    mat_name,
                "SectionWidth_m":  f"{width:.3f}",
                "SectionHeight_m": f"{height:.3f}",
                "Length_m":        f"{length:.3f}",
                "E_kPa":           str(getattr(bt, "E",     "") or ""),
                "gamma_kNm3":      str(getattr(bt, "gamma", "") or ""),
            })
            return ifc_beam
        except Exception as e:
            print(f"    ⚠ 跳过梁 '{beam.name}': {e}")
            return None

    # ── 嵌入桩 ────────────────────────────────────────────────────────────────

    def _export_embedded_piles(self) -> None:
        piles = self.pit.structures.get(StructureType.EMBEDDED_PILES.value, []) or []
        elems = []
        for pile in piles:
            e = self._pile_to_ifc(pile)
            if e is not None:
                elems.append(e)
        self._contain(elems, "EmbeddedPiles")
        print(f"  ✓ 嵌入桩：{len(elems)} 根（实心圆截面，深灰色）")

    def _pile_to_ifc(self, pile: EmbeddedPile):
        try:
            pts = pile.line.get_points()
            p0, p1 = pts[0], pts[1]
            radius = _get_pile_radius(pile)
            style = self._get_style("pile")
            rep = _build_circle_solid(self._ifc, self._ctx, p0, p1, radius, style)
            placement = _local_placement(self._ifc, self._site_pl)
            ifc_pile = self._ifc.createIfcPile(
                _new_guid(), self._owner,
                pile.name, getattr(pile, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None, "DRIVEN"
            )
            mat_name = _get_material_name(pile)
            _assign_ifc_material(self._ifc, ifc_pile, mat_name)
            pt = getattr(pile, "_pile_type", None)
            length = math.sqrt((p1.x-p0.x)**2+(p1.y-p0.y)**2+(p1.z-p0.z)**2)
            _pset(self._ifc, self._owner, ifc_pile, "Pset_Pile", {
                "PlaxisName":   pile.name,
                "MaterialName": mat_name,
                "Diameter_m":   f"{radius*2:.3f}",
                "Length_m":     f"{length:.3f}",
                "E_kPa":        str(getattr(pt, "E",     "") or ""),
                "gamma_kNm3":   str(getattr(pt, "gamma", "") or ""),
            })
            return ifc_pile
        except Exception as e:
            print(f"    ⚠ 跳过嵌入桩 '{pile.name}': {e}")
            return None

    # ── 土体块 ────────────────────────────────────────────────────────────────

    def _export_soil_blocks(self) -> None:
        soil_blocks = self.pit.structures.get(StructureType.SOIL_BLOCKS.value, []) or []
        elems = []
        for sb in soil_blocks:
            e = self._soil_block_to_ifc(sb)
            if e is not None:
                elems.append(e)
        self._contain(elems, "SoilBlocks")
        if elems:
            print(f"  ✓ 土体块：{len(elems)} 个（棕色）")

    def _soil_block_to_ifc(self, sb: SoilBlock):
        try:
            placement = _local_placement(self._ifc, self._site_pl)
            ifc_geo = self._ifc.createIfcGeographicElement(
                _new_guid(), self._owner,
                sb.name, getattr(sb, "comment", None),
                None, placement, None, None, "TERRAIN"
            )
            mat_name = str(getattr(getattr(sb, "_material", None), "name", None) or "Unknown Soil")
            _assign_ifc_material(self._ifc, ifc_geo, mat_name)
            _pset(self._ifc, self._owner, ifc_geo, "Pset_SoilBlock", {
                "PlaxisName":   sb.name,
                "SoilMaterial": mat_name,
            })
            return ifc_geo
        except Exception as e:
            print(f"    ⚠ 跳过土体块 '{sb.name}': {e}")
            return None

    # ── 钻孔 ──────────────────────────────────────────────────────────────────

    def _export_boreholes(self) -> None:
        bh_set = getattr(self.pit, "borehole_set", None)
        if bh_set is None:
            return
        boreholes = getattr(bh_set, "boreholes", []) or []
        elems = []
        for bh in boreholes:
            e = self._borehole_to_ifc(bh)
            if e is not None:
                elems.append(e)
        self._contain(elems, "Boreholes")
        print(f"  ✓ 钻孔：{len(elems)} 个（实心圆柱，深棕色）")

    def _borehole_to_ifc(self, bh: Borehole):
        try:
            loc = bh.location
            depth = bh.depth()
            gl = bh.ground_level
            style = self._get_style("borehole")

            # 实心圆柱（竖直向下）
            p_top = Point(loc.x, loc.y, gl)
            p_bot = Point(loc.x, loc.y, gl - depth)
            rep = _build_circle_solid(self._ifc, self._ctx, p_top, p_bot, 0.05, style)

            placement = _local_placement(self._ifc, self._site_pl,
                                         origin=(loc.x, loc.y, gl))
            ifc_bh = self._ifc.createIfcGeographicElement(
                _new_guid(), self._owner,
                bh.name, getattr(bh, "comment", None),
                None, placement, _product_shape(self._ifc, [rep]), None, "USERDEFINED"
            )
            ifc_bh.ObjectType = "Borehole"

            layer_info = "; ".join(
                f"{getattr(getattr(ly, 'soil_layer', None), 'name', None) or ly.name}"
                f"[{ly.top_z:.2f}~{ly.bottom_z:.2f}m]"
                for ly in bh.layers
            )
            wh = bh.water_head
            _pset(self._ifc, self._owner, ifc_bh, "Pset_Borehole", {
                "PlaxisName":    bh.name,
                "GroundLevel_m": f"{gl:.3f}",
                "Depth_m":       f"{depth:.3f}",
                "WaterHead_m":   f"{wh:.3f}" if wh is not None else "N/A",
                "LayerCount":    str(len(bh.layers)),
                "LayerSequence": layer_info,
            })
            return ifc_bh
        except Exception as e:
            print(f"    ⚠ 跳过钻孔 '{bh.name}': {e}")
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
    colour_overrides: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> None:
    """
    一行导出 FoundationPit 为 IFC 文件（v2.1）。

    colour_overrides 示例（自定义配色）：
        export_to_ifc(pit, "out.ifc",
            colour_overrides={"wall": (0.9, 0.9, 0.9), "beam": (0.2, 0.4, 0.8)})
    """
    IFCExporter(
        pit, schema=schema, author=author, organization=organization,
        project_address=project_address,
        latitude=latitude, longitude=longitude, elevation=elevation,
        colour_overrides=colour_overrides,
    ).export(file_path)
