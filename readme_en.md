# 🧱 plaxisproxy-excavation

*A Pythonic automation framework for Plaxis 3D excavation modeling and simulation*

###


## English TL;DR

* **Why this wrapper?** Native `plxscripting` can be opaque (discoverability via `__dir__()`), order/attribute sensitive, and error-prone. This project provides **explicit, OOP APIs** with validation and resilient workflows.
* **What you get:** Automated modeling, staged calculation, unified result extraction (Excel-ready), and a maintainable framework that’s simple to extend and integrate.
* **How to use:** See `examples/testmapper.py`—it walks through walls, braces, wells, phased runs, and Excel export. Adapt it to your project with minimal edits.

###

## 🌍 Project Introduction

**plaxisproxy-excavation** is an automation and modeling wrapper for **Plaxis 3D**, built with **Python 3.9.x** and aimed at **teaching, research, and engineering practice**.

It wraps Plaxis’ native scripting interface (`plxscripting`) with an **object-oriented (OOP)** design and a clear, unified API layer, enabling:

* 🧩 **Automated modeling**: soils, retaining systems, dewatering wells, staged excavation, etc.
* ⚙️ **Automated calculations**: mesh/run across multiple phases and construction stages.
* 📊 **Result extraction**: pull displacements, internal forces, stresses, and export (e.g., Excel).
* 🚀 **Extensible framework**: easy to add new materials, structures, boundaries, and post-processing.

**Goal:** replace the low-level, command-style experience of `plxscripting` with a **clear, maintainable, reproducible** Python API for Plaxis 3D numerical simulations.

###

## 💡 Motivation: Why an Explicit Plaxis API?

### Pain points in native `plxscripting` (typical in real projects)

* **Poor discoverability:** many object members aren’t documented or IDE-friendly; you often rely on `__dir__()` probing and trial-and-error.
* **Opaque properties/arguments:** inconsistent names, cases, and types across versions; easy to mis-type or mis-order.
* **Strict call ordering:** e.g., *material → structure → stage*; the wrong order leads to failures.
* **Unhelpful errors:** vague messages such as “unknown property” or “wrong argument type”.

### What an explicit API gives you (this project’s approach)

1. **Readable, IDE-friendly API**
   We expose **classes + methods** (`RetainingWall`, `Beam`, `Well`, `Phase`, …). Your IDE autocompletes, and code becomes self-documenting.

2. **Fixes common “attribute assignment failed” issues**
   Wrapper-level **parameter validation**, **naming normalization**, and **ordering guarantees** eliminate mistakes like wrong order, wrong attribute name, or missing/extra args when defining materials and structures.

3. **Automated workflow + Output binding**
   From building to calculating to extracting, the wrapper handles **phase switching, Output binding, and node/gauss-point fallbacks**—so full-phase batch extraction is robust and repeatable.

###

## ✨ Features & Benefits

### Side-by-side comparison (native vs wrapped)

| Dimension            | Native `plxscripting`               | This Wrapper (explicit API)                   |
| #################### | ################################### | ############################################# |
| **API exposure**     | Probe via `__dir__()`/trial         | Clear classes & methods (IDE friendly)        |
| **Params**           | Inconsistent names, order sensitive | Unified names, type hints, enforced order     |
| **Error handling**   | Vague exceptions                    | Validations & helpful messages                |
| **Phase management** | Manual and error-prone              | `Phase` objects with inheritance & activation |
| **Results**          | Manual Output binding & switching   | Automated binding, phase iteration, fallbacks |
| **Maintainability**  | Command-style scripts               | OOP, modular, reusable, extensible            |
| **Teaching**         | Steep learning curve                | Clear steps, friendly examples                |

### Checked advantages at a glance

* [x] **Higher modeling efficiency**: batch-create model entities with code.
* [x] **Less repetition**: reusable templates for staged construction and repeated scenarios.
* [x] **Better maintainability**: OOP modules localize changes.
* [x] **Easy to extend & integrate**: plug into optimizers, plotting, or parameter studies.
* [x] **Teaching-ready**: transparent workflow, approachable examples.
* [x] **More robust**: ordering, naming, and typing are enforced at the API layer.
* [x] **Unified Pythonic style**: consistent naming & parameters.
* [x] **Batch simulations**: quickly run comparative scenarios.

###

## ⚙️ Installation & Environment

### Requirements

* **Python 3.9.x**
* **Plaxis 3D** (Remote Scripting service enabled)

### Python packages

Before installing this framework, you need to install the dependencies first:

```bash
pip install shapely==2.0.7
# Only PLAXIS 3D 2023.2: pip install 'plxscripting==1.0.4'
# For V22, import the client shipped with the installed PLAXIS release
```

Run the following command directly to install this framework:

```bash
pip install plaxisproxy-excavation
```

### Plaxis remote configuration

Edit (example): `config/plaxis_config.py`

```python
HOST = "localhost"
PORT = 10000
PASSWORD = "your_plaxis_password"
```

Ensure Plaxis 3D is running and Remote Scripting is enabled.

###

## 🧩 Project Architecture

### Suggested directory layout

```
plaxisproxy-excavation/
├─ config/
│  └─ plaxis_config.py                # Plaxis remote connection settings
├─ examples/
│  └─ testmapper.py                   # Full demo: walls, braces, wells, phases, Excel export
├─ src/
│  └─ plaxisproxy_excavation/
│     ├─ core/
│     │  ├─ __init__.py
│     │  └─ plaxisobject.py           # Base object abstraction (IDs/session/property mapping)
│     │
│     ├─ components/
│     │  ├─ __init__.py
│     │  ├─ curvepoint.py             # Feature/curve points for geometry & loads
│     │  ├─ mesh.py                   # Global meshing and refinement settings
│     │  ├─ phase.py                  # Phase entity: inheritance, activation lists, soil overrides
│     │  ├─ phasesettings.py          # Plastic stage / load type / stepping parameters
│     │  ├─ projectinformation.py     # Project meta + unit system
│     │  └─ watertable.py             # Water-level points & surfaces
│     │
│     ├─ materials/
│     │  ├─ __init__.py
│     │  ├─ anchormaterial.py         # Materials for anchors/tiebacks
│     │  ├─ basematerial.py           # Common material base class (validation/defaults)
│     │  ├─ beammaterial.py           # Beam material (struts, walers, etc.)
│     │  ├─ pilematerial.py           # Pile / embedded pile material
│     │  ├─ platematerial.py          # Plate material (e.g., diaphragm walls)
│     │  └─ soilmaterial.py           # Soil materials + factory (SoilMaterialFactory, enums)
│     │
│     ├─ plaxishelper/
│     │  ├─ __init__.py
│     │  ├─ boreholemapper.py         # Boreholes/layer sequences → Plaxis mapping
│     │  ├─ geometrymapper.py         # Point/Line/Polygon/Surface mapping
│     │  ├─ loadmapper.py             # Loads & boundary conditions mapping
│     │  ├─ materialmapper.py         # Material property mapping & set-order control
│     │  ├─ meshmapper.py             # Meshing and refinement dispatch
│     │  ├─ monitormapper.py          # Monitoring points / result probes
│     │  ├─ phasemapper.py            # Phase creation/inheritance/activations mapping
│     │  ├─ plaxisoutput.py           # Output session & result queries
│     │  ├─ plaxisrunner.py           # Input/Output clients, session mgmt, error handling
│     │  ├─ projectinfomapper.py      # Project info & units mapping
│     │  ├─ resulttypes.py            # Result enums (e.g., Plate.Ux)
│     │  └─ structuremapper.py        # Structures (walls/beams/wells/loads) mapping
│     │     # watertablemapper.py is at the same level as structuremapper.py
│     │
│     ├─ structures/
│     │  ├─ __init__.py
│     │  ├─ anchor.py                 # Anchors/tiebacks
│     │  ├─ basestructure.py          # Base class for structures (common properties, create protocol)
│     │  ├─ beam.py                   # Beams/struts/walers
│     │  ├─ embeddedpile.py           # Embedded piles
│     │  ├─ load.py                   # Surface/line/point loads
│     │  ├─ retainingwall.py          # Retaining walls (D-walls, secant piles, etc.)
│     │  ├─ soilblock.py              # Soil blocks (excavation blocks / retained blocks)
│     │  └─ well.py                   # Wells (extraction/recharge: flow rate, depth, heads)
│     │
│     ├─ excavation.py                # FoundationPit container + StructureType enum
│     └─ builder/
│        └─ excavation_builder.py     # Orchestrator: build/phases/calc/result extraction
│
├─ tests/
│  └─ ...
├─ requirements.txt
└─ README.md

```

### Module responsibilities

* **builder/excavation_builder.py** – Core orchestrator for model validation, creation, phase management, calculation, and result extraction.
* **excavation.py** – `FoundationPit` container with geometry bounds, structure lists, phases, etc.
* **materials/** – Soil/plate/beam/… classes; property mapping and defaults.
* **structures/** – Retaining walls, braces/anchors, wells, etc., each with `create_in_plaxis()` logic.
* **geometry.py** – Helpers for constructing consistent 3D geometry.
* **borehole.py** – Boreholes and layered soils by depth.
* **plaxishelper/** – Light wrappers on `plxscripting` for robust Input/Output sessions.
* **resulttypes.py** – Unified result “leaves” (e.g., Ux/Uy/Uz, stresses), used by the extractor APIs.

### How it fits together

```
PlaxisRunner  →  ExcavationBuilder  →  FoundationPit
                                   ↙           ↘
                         Structures / Materials   Phases & WaterTable
                                   ↘           ↙
                                Results Export (Excel/CSV)
```

* **PlaxisRunner** manages sessions;
* **ExcavationBuilder** maps your pit description into Plaxis, runs phases, and extracts results;
* **FoundationPit** aggregates geometry, materials, structures, and phases;
* **Exporter** turns results into Excel/CSV for downstream analysis.

### Contributing

1. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature
   ```
2. Add/modify modules under `src/plaxisproxy_excavation/`.
3. Include a minimal example in `examples/`.
4. Add/update tests under `tests/`.
5. Open a PR with motivation, API notes, and usage example.

###

## 🧠 Example Walkthrough — `examples/testmapper.py`

This end-to-end demo assembles a pit with **retaining walls, three brace levels, dewatering wells, four phases**, and **Excel export** of wall displacements.

**Key ideas showcased:**

* **Geometry helpers** (`rect_wall_x/rect_wall_y`, `line_2pts`) to build robust walls/lines.
* **Well placement** along polygon edges and inside grids, with hard caps and deduplication.
* **Soil materials & layered boreholes** assembled into a `BoreholeSet`.
* **Structures**: diaphragm walls + horizontal braces (three levels).
* **Global elevations**: surface, excavation bottom, wall toe.
* **Phases**: initial → dewatering → excavation L1 → excavation L2 (with inheritance and activations).
* **Per-phase water table** example.
* **Build → Calculate → Export**: fully automated pipeline with robust Output binding and phase iteration.

> Tip: Treat the example as a template—swap in your geometry/materials/phases, and keep the build→compute→export rhythm.

###

## 🔮 Future Directions

* 📈 Visualization module (matplotlib/plotly).
* ⚡ Parallel/multi-case orchestration for batch studies.
* 🧮 Parameterization & optimization interfaces (e.g., heuristic/Bayesian loops).
* 🧠 Adaptive meshing strategies by depth/wall stiffness.
* 🌐 Web console for remote previews and job queues.

###

## 📜 License & Acknowledgments

* Recommended: **Apache-2.0** (or your project’s existing choice).
* Thanks to the Plaxis team for `plxscripting`, and to contributors/users for feedback that shaped these abstractions.

## Communication & Discussion

<p align="center">
  <img src="doc/imgs/EatRicer_qrcode.jpg" alt="WeChat Official Account QR Code" width="30%" />
  <img src="doc/imgs/WXCode.jpg" alt="Add me on WeChat to discuss and optimize together" width="30%" />
</p>
