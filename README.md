# Text-to-Papercraft Pipeline

A complete C++17 pipeline that converts a 3D mesh (`.obj`) into printable papercraft sheets (`.png`). The pipeline implements two research contributions: **fold-aware mesh simplification** and **papercraft-aware spectral segmentation**.

## Table of Contents

- [System Requirements](#system-requirements)
- [Build Instructions](#build-instructions)
- [CLI Usage](#cli-usage)
- [Preparing Your .obj File](#preparing-your-obj-file)
- [Interpreting Outputs](#interpreting-outputs)
- [Research Contributions](#research-contributions)

---

## System Requirements

| Requirement | Minimum Version |
|---|---|
| GCC | 10+ (C++17 support required) |
| *or* Clang | 12+ |
| CMake | 3.20+ |
| Git | Any recent version (for FetchContent) |
| OS | Linux: Ubuntu 20.04+ or Debian 11+, or Google Colab |
| OpenMP | Usually bundled with GCC |

### Additional System Libraries

- `libopenmesh-dev` — half-edge mesh data structure
- `libeigen3-dev` — linear algebra (spectral decomposition, LSCM)
- `libpng-dev` — PNG writing support

For **Google Colab**, install system dependencies with:

```bash
!apt-get install -y cmake libopenmesh-dev libeigen3-dev libpng-dev
```

---

## Build Instructions

### Local Linux

```bash
# 1. Install system dependencies
sudo apt-get install -y cmake libopenmesh-dev libeigen3-dev libpng-dev

# 2. Clone the repository
git clone <repo-url> papercraft_pipeline
cd papercraft_pipeline

# 3. Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 4. Run (from project root)
cd ..
./build/papercraft inputs/model.obj
```

### Google Colab

```python
# Cell 1: Install system dependencies
!apt-get install -y cmake libopenmesh-dev libeigen3-dev libpng-dev
```

```python
# Cell 2: Clone and build
!git clone <repo-url> papercraft_pipeline && \
 cd papercraft_pipeline && \
 mkdir -p build && cd build && \
 cmake .. -DCMAKE_BUILD_TYPE=Release && \
 make -j4
```

```python
# Cell 3: Run the pipeline
!cd papercraft_pipeline && ./build/papercraft inputs/model.obj
```

---

## CLI Usage

```
./papercraft model.obj [options]
```

### Required Arguments

| Argument | Description |
|---|---|
| `model.obj` | Input mesh file in Wavefront OBJ format |

### Optional Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output DIR` | string | `./outputs` | Output directory for all generated files |
| `--faces N` | int | `2000` | Target face count after simplification |
| `--patches N` | int | `0` | Number of patches. `0` = automatic selection via elbow method |
| `--fold-threshold DEG` | float | `30.0` | Dihedral angle threshold (degrees) for classifying fold edges |
| `--fold-preserve RATIO` | float | `0.80` | Minimum fold edge preservation ratio after simplification |
| `--tab-width MM` | float | `3.0` | Width of gluing tabs in millimeters |
| `--distortion-warn THRESH` | float | `0.25` | ARAP proxy distortion warning threshold |
| `--threads N` | int | `0` | OpenMP thread count. `0` = use all available cores |
| `--config PATH` | string | `config.json` | Path to JSON configuration file |
| `--verbose` | flag | `false` | Print per-patch metrics during processing |
| `--dry-run` | flag | `false` | Validate mesh and report stats without processing |

### Examples

```bash
# Basic usage with defaults
./papercraft bunny.obj

# High-detail papercraft with more patches
./papercraft dragon.obj --faces 4000 --patches 25 --verbose

# Quick preview with low face count
./papercraft teapot.obj --faces 500 --output ./preview

# Dry run to check mesh quality before committing to a full run
./papercraft unknown_mesh.obj --dry-run

# Custom config file with all parameters
./papercraft model.obj --config my_settings.json
```

---

## Preparing Your .obj File

### Recommended Input Characteristics

- **Face count:** 5,000–50,000 faces. Below 5,000, simplification has little to do. Above 50,000, spectral decomposition becomes slow.
- **Topology:** Single connected component. The pipeline will extract the largest component and discard others with a warning.
- **Materials/textures:** Ignored. All material and texture coordinate data is stripped on load.

### Automatic Repairs

The pipeline automatically handles:

- **Non-manifold edges:** Repaired via edge splitting
- **Duplicate vertices:** Merged on load
- **Degenerate faces:** Faces with area < 1e-12 are removed
- **Small holes:** Boundary loops up to 10 edges are closed
- **Multiple components:** Largest component by face count is kept; others are discarded with a logged warning

### Pre-cleaning in MeshLab (Recommended)

For best results, clean your mesh in MeshLab before running:

1. **Remove small disconnected pieces:**
   `Filters → Cleaning and Repairing → Remove Isolated Pieces (wrt Face Num)`
   Set threshold to `100`.

2. **Remove duplicate vertices:**
   `Filters → Cleaning and Repairing → Remove Duplicate Vertices`

3. **Export clean mesh:**
   `File → Export Mesh As → .obj`
   In the export dialog, **uncheck** "Tex Coords" and "Normals" — the pipeline recomputes normals and ignores texture coordinates.

---

## Interpreting Outputs

### `stage1_preview.png`

**What it shows:** An orthographic rendering of the loaded and repaired mesh from the (1,1,1) viewing direction.

| Quality | What you see | What to do |
|---|---|---|
| **Good** | Clean, recognizable silhouette. No holes or floating fragments. | Proceed — mesh is clean. |
| **Bad** | Holes visible, thin spikes, or disconnected floating pieces. | Go back to MeshLab and clean the mesh. Check the stage1_stats.json for specifics. |

### `stage2_foldlines.png`

**What it shows:** The simplified mesh with detected fold edges highlighted in red.

| Quality | What you see | What to do |
|---|---|---|
| **Good** | Red lines trace the major creases and geometric features of the model. The overall shape is preserved. | Proceed. |
| **Bad — too few red lines** | The model looks blobby; fold lines are missing from important creases. | Lower `--fold-threshold` (e.g., from 30 to 20 degrees). |
| **Bad — too many red lines** | Almost every edge is red; the mesh is noisy. | Raise `--fold-threshold` (e.g., from 30 to 45 degrees) or increase `--faces` for a smoother mesh. |
| **Bad — shape lost** | The simplified mesh doesn't resemble the original. | Increase `--faces` to retain more geometry. |

### `stage3_elbow.png`

**What it shows:** A plot of segmentation distortion proxy (y-axis) vs. number of patches k (x-axis). A vertical marker indicates the automatically chosen k at the "elbow" of the curve.

| Quality | What you see | What to do |
|---|---|---|
| **Good** | Clear L-shaped curve with a visible elbow. The chosen k is at the bend. | Automatic selection is working well. |
| **Bad — flat curve** | Distortion barely changes across k values. | The mesh may be too simple or too uniform. Try manually setting `--patches N`. |
| **Bad — noisy curve** | Distortion jumps erratically. | The mesh may have numerical issues. Try increasing `--faces` for a cleaner simplified mesh. |

### `stage3_patches.png`

**What it shows:** The simplified mesh with each patch colored differently.

| Quality | What you see | What to do |
|---|---|---|
| **Good** | Patches correspond to logical parts of the model (head, body, limbs, etc.). Patch boundaries follow fold lines. | Proceed. |
| **Bad — patches are fragmented** | Many tiny patches scattered across the surface. | Reduce `--patches` or let auto-selection choose a smaller k. |
| **Bad — patches span sharp folds** | A single patch wraps around a sharp crease. | Increase the dihedral weight in config.json (`dihedral_weight`) to make spectral cuts prefer fold lines more strongly. |

### `stage3_uv_layout.png`

**What it shows:** All unfolded patches laid out flat, showing the 2D shapes that will be printed.

| Quality | What you see | What to do |
|---|---|---|
| **Good** | Patches are recognizable flat shapes with moderate aspect ratios. No extreme stretching. | Proceed. |
| **Bad — extreme stretching** | Some patches are stretched into thin slivers. | The patch wraps too much curvature. Increase `--patches` to subdivide further. |
| **Bad — overlapping UV** | Patches overlap in the UV layout (check stage3_unfolding.json for `arap_proxy_distortion` values). | Increase `--patches`. Patches that aren't topological disks cause fallback to planar projection. |

### `papercraft_sheet.png` (and `papercraft_sheet_p{N}.png`)

**What it shows:** The final printable papercraft sheet(s) at the configured DPI. Cut lines are solid black, valley folds are dashed blue, mountain folds are dash-dot red, and glue tabs are dashed gray with numbered labels.

| Quality | What you see | What to do |
|---|---|---|
| **Good** | All pieces fit on the page(s) with no overlaps. Tabs are visible and labeled. Line types are distinguishable. | Print it! |
| **Bad — overlaps** | Pieces overlap on the sheet. The pipeline exits with a non-zero return code. | Increase `--faces` (fewer but larger patches) or increase `--patches` (more but smaller patches). Also check `--tab-width`. |
| **Bad — too many pages** | Pieces are spread across many pages. | Decrease `--faces` for fewer total pieces, or decrease `--tab-width`. |

### `evaluation_report.txt`

**What it shows:** A summary table of all pipeline metrics with pass/warn/fail status for each stage.

Rows highlighted as **WARN** or **FAIL** indicate:

- `fold_preservation_ratio < 0.80` → **WARN**: Simplification lost too many fold edges. Increase `--faces`.
- `mean_arap_distortion > 0.25` → **WARN**: Unfolding introduced significant distortion. Increase `--patches`.
- `max_arap_distortion > 0.40` → **WARN**: At least one patch is severely distorted. Check which patch in stage3_unfolding.json.
- `overlap_count > 0` → **FAIL**: Sheet layout has overlaps. Pipeline output is not printable.
- `patches_using_fallback > k/2` → **WARN**: Most patches failed LSCM unfolding. Mesh topology may have issues.

---

## Research Contributions

### Contribution 1: Fold-Aware Mesh Simplification

#### The Problem

When you simplify a 3D mesh (reduce its triangle count to make it manageable), standard algorithms treat every edge the same. The most widely used method — Quadric Error Metrics (QEM) by Garland and Heckbert — assigns a cost to collapsing each edge based on how much the surface shape would change geometrically. It then collapses the cheapest edges first.

This works beautifully for rendering, where you just want the model to *look* the same from a distance. But for papercraft, there's a critical difference: **some edges aren't just geometric details — they're fold lines.** A fold line is an edge where two faces meet at a sharp angle, forming a physical crease in the paper model. These are the edges you'll actually fold along when assembling the papercraft.

Standard QEM doesn't know this. It might collapse a fold edge because the local geometric error is small, even though that edge represents a critical physical crease. The result: a simplified mesh that looks smooth but has lost the structural information needed to build the paper model.

#### Our Solution

We keep QEM's speed and simplicity but add a verification step. After simplification, we check: *how many of the original fold edges survived?* We compute a **fold preservation ratio** — the fraction of original fold-line edges that still exist (or have a close equivalent) in the simplified mesh.

If this ratio falls below a threshold (default: 80%), we retry with a higher face budget. Each retry multiplies the target face count by 1.5, giving more room for fold edges to survive. We allow up to 3 retries.

This approach is deliberately simple. We don't modify QEM's internal cost function (which would complicate the algorithm and slow it down). Instead, we treat simplification as a black box and verify its output meets our papercraft-specific quality criterion. If it doesn't, we give it more budget. This is robust, easy to understand, and works with any QEM implementation.

#### Why It Matters

Without fold awareness, a model simplified from 50,000 to 2,000 faces might lose 40–60% of its fold edges. The resulting papercraft looks like a faceted blob rather than a recognizable object. With our verification loop, fold preservation stays above 80%, and the papercraft retains the sharp creases that define its shape.

---

### Contribution 2: Papercraft-Aware Spectral Segmentation

#### The Problem

To make a papercraft model, you need to cut the 3D surface into flat pieces (patches) that can be unfolded onto paper without too much stretching or distortion. This requires segmenting the mesh — dividing it into groups of faces that will each become one flat piece.

Existing segmentation methods in computer graphics are designed for **texture mapping**, not papercraft. Methods like LSCM (Least Squares Conformal Maps) and ABF++ (Angle-Based Flattening) segment meshes to minimize angular distortion when projecting textures. They optimize for even texel density across the surface. Their seam placement algorithms don't consider physical fold lines at all — they might place a seam right through the middle of a flat region (creating an unnecessary cut in the paper) while keeping a sharp 90-degree fold within a single patch (making it impossible to flatten).

#### Our Solution

We use **spectral segmentation** — a technique from graph theory where you represent the mesh as a graph (faces are nodes, shared edges are connections) and use the eigenvectors of the graph's Laplacian matrix to find natural clusters.

The key innovation is in how we weight the graph edges. For each edge shared between two faces, we assign a weight based on the dihedral angle between those faces:

- **Small dihedral angle** (faces are nearly coplanar) → **high weight** (keep these faces together — they'll flatten easily)
- **Large dihedral angle** (faces meet at a sharp crease) → **low weight** (encourage the segmentation to cut here — this is a natural fold line)

For edges that were detected as fold lines in Stage 2, we further reduce the weight by a factor of 20 (multiply by 0.05). This strongly biases the spectral cut to place patch boundaries along fold lines.

The number of patches (k) is either specified by the user or chosen automatically using the **elbow method**: we compute a distortion proxy (how much normal variation exists within each patch) for k = 4 through 20, plot the curve, and pick the k at the "elbow" — the point where adding more patches stops significantly reducing distortion.

#### Why It Matters

Texture-map segmentation methods produce patches optimized for rendering — they minimize angular distortion in the UV mapping but create awkward, hard-to-assemble papercraft pieces. Our method produces patches that:

1. **Follow fold lines** — patch boundaries are placed at sharp creases, which are natural places to cut paper
2. **Are individually flat** — each patch has low internal curvature variation, so it can be unfolded with minimal distortion
3. **Have practical shapes** — because the spectral method respects the global geometry, patches tend to be connected, compact regions rather than fragmented slivers

The result is a papercraft model where the cuts make physical sense: you cut along the creases, fold along the creases, and each flat piece corresponds to a recognizable part of the 3D model.