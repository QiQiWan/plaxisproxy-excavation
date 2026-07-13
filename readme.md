# 🧱 plaxisproxy-excavation

*A Pythonic automation framework for Plaxis 3D excavation modeling and simulation*
（一个面向 Plaxis 3D 的基坑工程建模与仿真自动化框架）

[Readme - English Version](readme_en.md)

## 🌍 项目简介 | Project Introduction

**plaxisproxy-excavation** 是一个面向 **Plaxis 3D** 的自动化仿真与建模封装框架，基于 Python 3.9.x 开发，适用于 **教学、科研与实际工程模拟**。

该项目通过对 Plaxis 的原生脚本接口（`plxscripting`）进行 **面向对象 (OOP)** 的再封装，构建了一个清晰、统一、可扩展的 API 层，实现了以下自动化流程：

* 🧩 **自动化建模**：自动生成土层、支护、降水、开挖阶段等；
* ⚙️ **自动化计算**：自动执行 Plaxis 求解流程；
* 📊 **结果提取与导出**：从 Plaxis Output 中提取关键结果（如位移、内力、应力）；
* 🚀 **可扩展框架**：便于添加新的材料类型、结构、边界条件与后处理逻辑。

> **目标**：以更清晰的 Python API 替代原生 `plxscripting` 的底层接口，
> 让 Plaxis 3D 的数值仿真变得**可编程、可维护、可复现**。

## 💡 设计初衷 | Motivation for Wrapping Plaxis API

### 🧱 原生 `plxscripting` 的痛点

Plaxis 官方提供的 `plxscripting` 库虽然能实现远程建模与控制计算，但存在明显问题：

| 问题类别                 | 原因                                                          | 影响                                                     |
| ------------------------ | ------------------------------------------------------------- | -------------------------------------------------------- |
| **命令发现困难**   | 许多对象的属性与方法只能通过 `__dir__()` 或实验调用才能得知 | 开发时缺乏补全与文档支持                                 |
| **属性调用不透明** | 参数命名、大小写与类型不一致                                  | 容易出错且调试困难                                       |
| **调用顺序依赖强** | 例如创建材料与结构的先后顺序必须符合内部逻辑                  | 一旦顺序错，脚本即失败                                   |
| **参数验证缺失**   | 无类型检查与参数验证机制                                      | 常见报错如 “unknown property”、“wrong argument type” |
| **错误提示晦涩**   | 返回通用异常信息                                              | 不利于教学与复用                                         |

> 💬 简言之，原生 API 是“能用但不友好”；它更像一个 **远程命令行接口** 而非真正的编程库。

### ✅ 显式封装 API 的优势

在 **plaxisproxy-excavation** 中，所有 Plaxis 命令均被重新封装为**显式 API**：

| 项目               | 原生 plxscripting                 | plaxisproxy-excavation 封装后          |
| ------------------ | --------------------------------- | -------------------------------------- |
| **调用方式** | 模糊命令（需 `__dir__()` 探测） | 显式类方法（IDE 可自动补全）           |
| **参数传递** | 不统一（类型/顺序依赖内部）       | 明确参数名与类型提示                   |
| **错误控制** | 无校验，抛出模糊异常              | 自动检测并抛出详细错误                 |
| **对象关系** | 全局命令式                        | 面向对象（Wall、Soil、Well、Phase 等） |
| **阶段管理** | 手动创建、关联复杂                | 自动管理 Phase 链接与激活逻辑          |
| **脚本结构** | 程序化但不可维护                  | 高层抽象清晰、模块化、可扩展           |

### ⚙️ 显式 API 能解决的核心问题

1. 🧩 **清晰调用关系**
   不再需要靠 `__dir__()` 试探命令。每个 Plaxis 对象都有专属类（如 `RetainingWall`、`SoilLayer`、`Well`），其方法与属性在 IDE 中一目了然。
2. 🧱 **参数赋值与顺序控制**
   创建材料或结构时，封装内部会自动确保正确的参数顺序与依赖关系。例如：

   * 材料属性自动检查单位与类型；
   * 结构依赖的材料对象会自动绑定；
   * 避免 “property not found” 与 “attribute missing” 错误。
3. 🧠 **逻辑防护与错误校验**
   每个 API 都内置参数校验与异常捕获。例如：

   * 若某墙体底部未闭合，将提示几何不封闭；
   * 若 `excava_depth` 超出最深墙底，则阻止生成坑底；
   * 这些检查能防止仿真失败或非物理计算。

## ✨ 功能与优势 | Features & Benefits

### 🔍 功能概览

| 功能模块                   | 描述                                            |
| -------------------------- | ----------------------------------------------- |
| 🏗️**自动化建模**   | 从几何、土层、结构到开挖阶段的全流程自动建模    |
| ⚙️**自动化计算**   | 自动触发网格生成与计算，支持阶段继承            |
| 📊**结果提取与导出** | 可提取 Ux、Uy、Uz、应力、内力等结果并导出 Excel |
| 🧱**项目检查**       | 自动检查墙体闭合、深度合理性、阶段连贯性        |
| 🧩**封装式 API**     | 提供 OOP 风格的清晰接口，统一参数命名与结构层级 |
| 🔁**可扩展性**       | 模块化设计，可自定义结构、材料、工况与导出器    |

### ✅ 封装带来的优势

* [X] **提高建模效率**：通过类方法批量创建模型要素；
* [X] **降低重复劳动**：支持多阶段、重复性模型的复用；
* [X] **改进代码可维护性**：清晰的模块与接口设计；
* [X] **便于扩展与集成**：可与优化算法、可视化工具协同；
* [X] **提升教学体验**：学生可快速理解数值建模逻辑；
* [X] **增强鲁棒性**：自动检测参数合法性与几何闭合；
* [X] **统一接口风格**：所有命令均采用 Python 风格命名；
* [X] **支持批量仿真**：可快速运行多工况对比分析。

## ⚙️ 安装与环境 | Installation & Environment

### 🧩 依赖环境

* Python **3.9.x**
* Plaxis **3D 2023 或以上版本**
* 需启用 Plaxis Remote Scripting 服务

### 📦 Python 包依赖

安装本框架之前需要先安装依赖：

```bash
pip install shapely==2.0.7
pip install plxscripting==1.0.4
```

直接运行以下命令安装本框架：

```bash
pip install plaxisproxy-excavation
```

### ⚙️ 配置 Plaxis 远程服务

修改：

```bash
config/plaxis_config.py
```

设置：

```python
HOST = "localhost"
PORT = 10000
PASSWORD = "your_plaxis_password"
```

确保 Plaxis 3D 已启动且 Remote scripting 服务处于开启状态。

## 🧩 项目结构 | Project Architecture

### 📁 目录树

```
plaxisproxy-excavation/
├─ config/
│  └─ plaxis_config.py
├─ examples/
│  └─ testmapper.py
├─ src/
│  └─ plaxisproxy_excavation/
│     ├─ core/
│     │  ├─ __init__.py
│     │  └─ plaxisobject.py           # 基类/对象抽象：统一ID/会话/属性映射
│     │
│     ├─ components/
│     │  ├─ __init__.py
│     │  ├─ curvepoint.py             # 曲线/特征点抽象（供几何/工况使用）
│     │  ├─ mesh.py                   # 网格全局设置/细化策略
│     │  ├─ phase.py                  # Phase 实体：继承、激活清单、土体覆盖等
│     │  ├─ phasesettings.py          # 塑性阶段/加载类型/步长等计算参数
│     │  ├─ projectinformation.py     # 工程信息、单位体系
│     │  └─ watertable.py             # 水位点表/水位面定义
│     │
│     ├─ materials/
│     │  ├─ __init__.py
│     │  ├─ anchormaterial.py         # 锚杆/拉索类材料
│     │  ├─ basematerial.py           # 统一材料基类（公共属性、校验）
│     │  ├─ beammaterial.py           # 梁单元材料（支撑/立柱等）
│     │  ├─ pilematerial.py           # 桩/嵌入式桩材料
│     │  ├─ platematerial.py          # 板单元材料（如地连墙）
│     │  └─ soilmaterial.py           # 土体材料与工厂（SoilMaterialFactory, 枚举）
│     │
│     ├─ plaxishelper/
│     │  ├─ __init__.py
│     │  ├─ boreholemapper.py         # 钻孔/层序 → Plaxis 映射
│     │  ├─ geometrymapper.py         # Point/Line/Polygon/Surface 映射
│     │  ├─ loadmapper.py             # 荷载与工况映射
│     │  ├─ materialmapper.py         # 各类材料属性映射/赋值顺序控制
│     │  ├─ meshmapper.py             # 网格生成与细化策略下发
│     │  ├─ monitormapper.py          # 监测点/结果监测映射
│     │  ├─ phasemapper.py            # Phase 创建/继承/激活清单映射
│     │  ├─ plaxisoutput.py           # Output 会话与结果查询封装
│     │  ├─ plaxisrunner.py           # Input/Output 客户端、会话与错误处理
│     │  ├─ projectinfomapper.py      # 工程信息/单位映射
│     │  ├─ resulttypes.py            # 结果枚举（Plate.Ux 等）
│     │  └─ structuremapper.py        # 结构体（墙/梁/井/荷载等）映射
│     │     └─ watertablemapper.py    # （同级）水位相关映射
│     │
│     ├─ structures/
│     │  ├─ __init__.py
│     │  ├─ anchor.py                 # 锚杆/拉锚结构
│     │  ├─ basestructure.py          # 结构基类：通用属性、创建协议
│     │  ├─ beam.py                   # 梁/撑（水平撑、围檩等）
│     │  ├─ embeddedpile.py           # 嵌入式桩
│     │  ├─ load.py                   # 面/线/点荷载实体
│     │  ├─ retainingwall.py          # 围护墙（地连墙/截桩墙等）
│     │  ├─ soilblock.py              # 土体块（开挖块/保留块）
│     │  └─ well.py                   # 井点（抽水/回灌，参数：流量、井深等）
│     │
│     ├─ excavation.py                # FoundationPit 容器 + StructureType 枚举
│     └─ builder/
│        └─ excavation_builder.py     # 构建器：建模/阶段/计算/结果提取的总控
│
├─ tests/
│  └─ ...
├─ requirements.txt
└─ README.md

```

### 🧱 模块职责概览

| 模块                            | 说明                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------ |
| **excavation_builder.py** | 控制整个建模与计算流程的核心模块。负责模型验证、阶段创建、计算执行、结果导出。 |
| **excavation.py**         | 定义基坑对象 `FoundationPit`，包含几何边界、结构列表、施工阶段等。           |
| **materials/**            | 封装不同类型的材料：土体、板、梁、锚杆等；提供属性映射与默认参数。             |
| **structures/**           | 定义支护、井点、墙体等结构元素类；各类均含 `create_in_plaxis()` 方法。       |
| **geometry.py**           | 几何操作辅助（如多边形生成、坐标变换、体积计算）。                             |
| **borehole.py**           | 土层与钻孔模型，支持按深度分层。                                               |
| **plaxishelper/**         | 对原生 `plxscripting` 进行轻封装，包括 Input、Output 的客户端管理。          |
| **resulttypes.py**        | 统一结果枚举（如位移、内力、应力类型），用于结果提取函数。                     |

### 🧭 架构逻辑图

```
PlaxisRunner  →  ExcavationBuilder  →  FoundationPit
                                   ↙
                        Structures / Materials / Phases
                                   ↘
                               Result Exporter
```

* **PlaxisRunner**：负责连接和会话管理；
* **ExcavationBuilder**：负责构建、计算、提取；
* **FoundationPit**：数据容器；
* **Structures**：结构与支护；
* **Exporter**：导出 Excel。

### 💬 如何贡献 | How to Contribute

1. Fork 仓库并创建新分支：

   ```bash
   git checkout -b feature/new-module
   ```
2. 在 `src/plaxisproxy_excavation/` 下添加新类或模块；
3. 编写单元测试（放入 `/tests`）；
4. 提交 PR 并附带使用示例。

## 🧠 示例解析 | Example: `examples/testmapper.py`

下面是一个完整的示例脚本（`examples/testmapper.py`），展示如何通过封装的 API 自动化完成基坑建模、计算与结果导出。

```python
# -*- coding: utf-8 -*-
"""
testmapper.py
Demonstration of automated pit assembly and calculation using Plaxis API wrapper.
"""

from src.plaxisproxy_excavation.excavation import FoundationPit
from src.plaxisproxy_excavation.excavation_builder import ExcavationBuilder
from src.plaxisproxy_excavation.plaxishelper.runner import PlaxisRunner
from src.plaxisproxy_excavation.materials import SoilMaterialFactory
from src.plaxisproxy_excavation.structures import RetainingWall, Well
from shapely.geometry import Polygon
import math

def rect_wall_x(y0=0, y1=50, x=0, z0=0, z1=-25):
    """Create rectangular wall geometry along x-axis."""
    return [[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]]

def wells_on_polygon_edges(polygon: Polygon, spacing: float = 5.0):
    """Generate well points along polygon edges at given spacing."""
    coords = list(polygon.exterior.coords)
    wells = []
    for i in range(len(coords)-1):
        p1, p2 = coords[i], coords[i+1]
        length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        n = int(length // spacing)
        for j in range(n+1):
            x = p1[0] + (p2[0]-p1[0]) * j / n
            y = p1[1] + (p2[1]-p1[1]) * j / n
            wells.append((x, y))
    return wells

def assemble_pit():
    """Create a demo excavation pit with walls, soil, and wells."""
    runner = PlaxisRunner("localhost", 10000, "password")
    pit = FoundationPit("TeachingDemoPit")

    # Define soil
    soil = SoilMaterialFactory.create_default()
    pit.add_soil(soil)

    # Define walls
    wall1 = RetainingWall(points=rect_wall_x(y0=0, y1=40, x=0))
    wall2 = RetainingWall(points=rect_wall_x(y0=0, y1=40, x=40))
    pit.add_structure(wall1)
    pit.add_structure(wall2)

    # Define wells
    polygon = Polygon([(0,0), (40,0), (40,40), (0,40)])
    wells = wells_on_polygon_edges(polygon)
    for w in wells:
        pit.add_structure(Well(x=w[0], y=w[1], depth=25))

    builder = ExcavationBuilder(runner, pit)
    builder.build()
    builder.calculate()

    # Export displacement results
    from src.plaxisproxy_excavation.utils.exporter import export_walls_horizontal_displacement_excel_2
    export_walls_horizontal_displacement_excel_2(builder, "results.xlsx")

if __name__ == "__main__":
    assemble_pit()
```

### 📖 逐行解析与对应模块

| 行号范围         | 内容                                             | 说明                                                                           |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------------------------ |
| **1–11**  | 模块导入                                         | 引入核心构建类与封装的几何、结构模块；替代了原生 `g_i.*` 命令。              |
| **13–18** | `rect_wall_x`                                  | 用于生成矩形墙体的几何点序列；返回四点封闭多边形。对应封装在 `geometry.py`。 |
| **20–32** | `wells_on_polygon_edges`                       | 利用 `shapely` 生成井点坐标；按边长度与间距自动布井。                        |
| **34–65** | `assemble_pit()` 主函数                        | 主体逻辑：连接 Plaxis → 定义土体 → 墙体 → 井点 → 构建与计算。              |
| **37**     | `PlaxisRunner`                                 | 建立与 Plaxis 3D Input 服务的通信连接（底层基于 plxscripting）。               |
| **38–39** | `FoundationPit`                                | 创建基坑对象，作为统一的模型容器。                                             |
| **42**     | `SoilMaterialFactory`                          | 自动生成默认土体材料（带物理参数），封装自 materials 模块。                    |
| **44–47** | `RetainingWall`                                | 定义墙体结构并加入基坑；自动建立墙面几何与材料关联。                           |
| **49–55** | `Well`                                         | 沿边布置井点降水系统，体现典型基坑降水布置。                                   |
| **57–61** | `ExcavationBuilder`                            | 调用构建器自动生成几何、材料、结构与阶段。                                     |
| **63–64** | `export_walls_horizontal_displacement_excel_2` | 结果导出函数，提取每面墙在各阶段的位移结果到 Excel。                           |
| **66–67** | 主函数调用                                       | 直接运行后完成整个建模、计算与导出流程。                                       |

### 🔁 自动化工作流说明

1. **连接阶段**：`PlaxisRunner` 自动打开 Input 会话；
2. **建模阶段**：`ExcavationBuilder.build()` 调用内部 `_create_materials()`、`_create_structures()`；
3. **计算阶段**：`builder.calculate()` 调用 Plaxis API 执行所有 Phase；
4. **结果阶段**：自动连接 Output，提取所需结果；
5. **导出阶段**：封装导出器输出 Excel 报表，一面墙一个 Sheet。

<!-- ## 🔮 未来展望 | Future Directions

* 📈 **结果可视化模块**：集成 matplotlib / plotly；
* 🧮 **优化算法接口**：与遗传算法/机器学习结合；
* 🧠 **智能网格划分**：基于墙深度与土层自适应剖分；
* 🌐 **Web 控制台**：支持浏览器端操作与模型预览。 -->

<!-- * ⚡ **并行计算支持**：多坑同步计算； -->

## 🤝 贡献与许可证 | Contributing & License

* 本项目遵循 **Apache-2.0**；
* 欢迎提交 Issue 或 Pull Request；
* 建议附带：

  * 示例脚本；
  * 单元测试；
  * 文档更新。

## 🏁 总结 | Summary

`plaxisproxy-excavation` 通过面向对象封装，将 Plaxis 的复杂命令行式接口转化为清晰、统一、自动化的 Python API，使得：

* 模型构建**更快、更稳、更可控**；
* 代码结构**更清晰、易维护**；
* 仿真流程**更自动化、可重复**。

无论是教学演示、科研实验，还是

工程项目的标准化分析流程，它都能极大提升 Plaxis 的使用体验。

✅ **Repository:** [QiQiWan/plaxisproxy-excavation](https://github.com/QiQiWan/plaxisproxy-excavation)

📚 **Language:** Python 3.9

🧩 **Dependencies:** shapely==2.0.7, plxscripting==1.0.4

🏗️ **Application:** Plaxis 3D Excavation Automation Framework

## 交流与讨论

<p align="center">
  <img src="doc/imgs/EatRicer_qrcode.jpg" alt="微信公众号二维码" width="30%" />
  <img src="doc/imgs/WXCode.jpg" alt="添加我的微信一起讨论与优化" width="30%" />
</p>
