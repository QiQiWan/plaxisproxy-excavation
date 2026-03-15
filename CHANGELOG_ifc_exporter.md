# IFC 导出模块 — 改动日志

**提交时间：** 2026-03-15  
**Git Commit：** `a5e963b`  
**提交信息：** `feat: add IFC export module (ifc_exporter.py)`

---

## 一、新增文件

### 1. `src/plaxisproxy_excavation/ifc_exporter.py`（核心模块）

**功能：** 将 `FoundationPit` 对象规范化并导出为 IFC 4 文件。

**依赖：** `ifcopenshell >= 0.8.0`（`pip install ifcopenshell`）

#### 对象映射关系

| plaxisproxy 对象 | IFC 实体 | 几何表达 | 属性集名称 |
|---|---|---|---|
| `RetainingWall` | `IfcWall` | `IfcFacetedBrep`（Polygon3D 面） | `Pset_RetainingWall` |
| `Anchor` | `IfcTendon` | `IfcSweptDiskSolid`（圆截面扫掠） | `Pset_Anchor` |
| `Beam` | `IfcBeam` | `IfcSweptDiskSolid` | `Pset_Beam` |
| `EmbeddedPile` | `IfcPile` | `IfcSweptDiskSolid` | `Pset_Pile` |
| `SoilBlock` | `IfcGeographicElement` | `IfcFacetedBrep` | `Pset_SoilBlock` |
| `Borehole` | `IfcGeographicElement` | 竖直扫掠体 | `Pset_Borehole` |

#### 空间层级

```
IfcProject
  └── IfcSite (ExcavationSite)
        └── IfcBuilding (ExcavationBuilding)
              └── IfcBuildingStorey (ExcavationLevel)
                    ├── IfcWall × N（挡土墙）
                    ├── IfcTendon × N（锚杆）
                    ├── IfcBeam × N（腰梁）
                    ├── IfcPile × N（嵌入桩）
                    ├── IfcGeographicElement × N（土体块）
                    └── IfcGeographicElement × N（钻孔）
```

#### 主要类和函数

```python
class IFCExporter:
    def __init__(self, pit: FoundationPit, schema="IFC4", author="", organization="")
    def export(self, file_path: str) -> None   # 主入口

def export_to_ifc(pit, file_path, author="", organization="", schema="IFC4") -> None
```

#### 内部工具函数（模块私有）

| 函数 | 作用 |
|---|---|
| `_new_guid()` | 生成 IFC 兼容 22 位 GlobalId |
| `_ifc_cartesian_point()` | 创建 IfcCartesianPoint |
| `_ifc_local_placement()` | 创建 IfcLocalPlacement |
| `_polygon3d_to_brep()` | Polygon3D → IfcFacetedBrep |
| `_retaining_wall_to_brep()` | RetainingWall 面 → IfcFacetedBrep |
| `_line3d_to_swept_solid()` | Line3D → IfcSweptDiskSolid（圆截面） |
| `_make_material()` | 创建 IfcMaterial |
| `_assign_material()` | 关联 IfcMaterial 到构件 |
| `_get_material_name()` | 从结构对象提取材料名（容错） |
| `_get_section_radius()` | 从材料属性推断截面半径 |

---

### 2. `example_ifc_export.py`（使用示例）

**位置：** 项目根目录  
**功能：** 完整演示如何构建 `FoundationPit` 并导出 IFC。

包含：
- 2 面挡土墙（南侧/北侧，Polygon3D 矩形面）
- 2 根锚杆（`ElasticAnchor`）
- 1 根腰梁（`ElasticBeam`）
- 1 根嵌入桩（`ElasticPile`）
- 2 个钻孔（含 3 层地层信息）

运行方式：
```bash
cd code/
PYTHONPATH=src python3 example_ifc_export.py
```

---

### 3. `example_excavation.ifc`（示例输出）

**位置：** 项目根目录  
**大小：** ~11 KB  
**内容统计：**

| IFC 实体 | 数量 |
|---|---|
| IfcWall | 2 |
| IfcTendon | 2 |
| IfcBeam | 1 |
| IfcPile | 1 |
| IfcGeographicElement（钻孔） | 2 |
| IfcPropertySet | 8 |
| IfcMaterial | 4 |

可用以下工具打开验证：
- [BIMvision](https://bimvision.eu/download/)（免费）
- [FreeCAD](https://www.freecad.org/)（免费）
- Autodesk Viewer（在线）

---

## 二、修改文件

### 1. `src/plaxisproxy_excavation/__init__.py`

**改动：** 从空文件改为暴露公开 API。

新增导出：
```python
from .ifc_exporter import IFCExporter, export_to_ifc
```

完整 `__all__` 列表包含：`FoundationPit`, `StructureType`, `MaterialType`,
`ProjectInformation`, `Phase`, `Point`, `PointSet`, `Line3D`, `Polygon3D`,
`Cube`, `Polyhedron`, `Borehole`, `BoreholeSet`, `BoreholeLayer`, `SoilLayer`,
`IFCExporter`, `export_to_ifc`

---

### 2. `pyproject.toml`

**改动：** `dependencies` 列表新增一行：

```toml
"ifcopenshell>=0.8.0",   # IFC 导出支持（可选，仅 ifc_exporter 模块需要）
```

---

## 三、未修改的文件

本次改动**不涉及**以下文件（均未修改）：
- 所有 `structures/` 下的结构类
- 所有 `materials/` 下的材料类
- `excavation.py`（`FoundationPit`）
- `geometry.py`
- `borehole.py`
- `plaxishelper/` 下的所有 mapper
- 所有测试文件

---

## 四、已知限制

| 限制 | 说明 |
|---|---|
| 挡土墙几何 | 目前仅导出单面 Brep，未生成真实厚度的六面体 |
| SoilBlock 无几何 | 若 `geometry=None`，导出无几何体的 IfcGeographicElement |
| IFC 版本 | 仅支持 IFC4，不支持 IFC2x3 |
| 钻孔类型 | IFC4 无原生 IfcBorehole，用 `IfcGeographicElement(USERDEFINED)` + `ObjectType="Borehole"` 代替 |

---

## 五、快速使用

```python
from plaxisproxy_excavation.ifc_exporter import export_to_ifc

# pit 是已构建好的 FoundationPit 对象
export_to_ifc(pit, "output.ifc")
```

或使用类接口（可自定义作者/组织）：

```python
from plaxisproxy_excavation.ifc_exporter import IFCExporter

exporter = IFCExporter(pit, author="EatRice", organization="智能建造研究院")
exporter.export("output.ifc")
```
