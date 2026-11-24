# Terrain-prior weakly supervised extraction of ancient mountain roads under closed canopy

Code accompanying the manuscript:

> **Airborne LiDAR extraction of forested ancient roads under terrain-physical constraints: a dual-branch weakly supervised framework, road-preserving filtering and adaptive-resolution DEM**

The pipeline has two computational branches:

* **Branch A — macro path finding.** A minimum-cost path is solved on a state-augmented graph (8 headings per node). The cost combines open-ness, a direction-dependent turn penalty and a **directional-cosine penalty** `λc · |cos φ|` that suppresses contour-cut shortcuts. Endpoints come from field survey. Branch A produces pseudo-labels and a distance field.
* **Branch B — weakly supervised segmentation.** A compact CNN (≈1.5 k parameters) is trained from scratch on 96 × 96 patches with four terrain channels (hillshade, normalised cross-section sign-change count, distance to the pseudo-label corridor, open-ness corridor response), and outputs a corridor probability map. The delivered axis is the skeleton of `P ≥ 0.5`.

The same corridor is then reused to relax the ground threshold locally (**road-preserving filter**) and to concentrate DEM resolution near the road (**adaptive-resolution DEM**).

Reference corridors (SHP) are used **only** for the final scoring step; they never enter feature construction, path search, pseudo-labelling, training or delivery.

## Repository layout

```
tools/            all processing and evaluation scripts
data/             placeholder — raw and derived data are NOT distributed
results/          created at run time (metrics, JSON, npz)
figures/          created at run time
tests/            regression checks for the no-leakage rules
```

### Entry points for the three contributions

| Contribution | Entry point | Supporting modules |
|---|---|---|
| Dual-branch framework (path prior → weakly supervised segmentation) | `tools/cold_start_transfer.py` | `path_extraction.py`, `path_lambda_cv.py`, `p2_terrain.py`, `train_p2.py`, `p2_dem.py`, `eval_freeaxis.py`, `eval_line_met.py` |
| Road-preserving filtering | `tools/corridor_filter.py` | `shudao_preservation_filter.py`, `csf_filter.py`, `3d_tube_filter.py` |
| Adaptive-resolution DEM | `tools/dem_rebuild.py` | `p2_adaptive_gsd.py`, `adaptive_gsd_render.py`, `p2_adaptive_validate.py` |

## Installation

```bash
python -m pip install -r requirements.txt
```

Developed and tested with Python 3.14, NumPy 2.4, SciPy 1.18, scikit-image 0.26, PyTorch 2.11 (CUDA 12.8). CPU-only runs work but are slower.

## Data (not included)

The airborne LiDAR point clouds were collected and are owned by **Chang'an University** and are not redistributable. Every script resolves its inputs from a single root, configurable with the environment variable `SHUDAO_DATA_ROOT` (default: `./data`):

```
$SHUDAO_DATA_ROOT/
├── 长安大学采集-shp/
│   ├── 杨家坝/
│   │   ├── yangjiaba.las                  # full-waveform returns
│   │   ├── 杨家河坝碥道-dem.tif           # original 5 cm DEM (EPSG:32648)
│   │   ├── 杨家坝_蜀道_中轴线.shp         # reference axis (scoring only)
│   │   └── 杨家坝_蜀道_本体范围.shp       # reference corridor polygon (scoring only)
│   └── 西坝村/
│       ├── 西坝村栈道-lidar.las
│       ├── 西坝村栈道-dem.tif             # original 10 cm DEM
│       └── XBC_orig_clean.shp             # reference corridor polygon (scoring only)
└── data_real/                             # 1 m derived grids and truth masks
```

All terrain attributes are computed on a **1 m grid using the minimum elevation per cell** (`dem_min`), which already suppresses residual vegetation without any prior filtering.

Derived products (probability maps, delivered point indices, adaptive DEM) can be obtained from the corresponding author on reasonable request, subject to the data provider's permission.

## Key parameters

| Parameter | Value | Role |
|---|---|---|
| `λc` (directional-cosine penalty) | 6.0 | derived as `k·tanβ`, `k = 10 m/m`, `β = 32–38°`; does **not** enter scoring |
| `d3` (3-D tube radius) | 3.5 m, both sites | half passable width 2 m + 1.5 m positioning tolerance; **enters scoring** |
| `AGL` cap | 2.0 m | low-vegetation layer; **enters scoring** |
| pseudo-label half-width `gate` | 3 px (Yangjiaba) / 5 px (Xibacun) | training only |
| line-metric buffer `r` | 2 m | **enters scoring**; `r = 1/3 m` reported as sensitivity |
| patch / optimiser | 96 × 96, BCE with positive weighting, Adam | early stopping on held-out patches |

## Reproduction

```bash
export SHUDAO_DATA_ROOT=/path/to/data          # Windows: set SHUDAO_DATA_ROOT=...

python tools/cold_start_transfer.py --site 杨家坝 --gate 3
python tools/eval_freeaxis.py --site 杨家坝 --axis skel --truth old --dist3 3.5 --agl 2
python tools/eval_new_shp.py --site 西坝村
python tools/eval_line_met.py                  # line metrics from the P>=0.5 skeleton
python tools/corridor_filter.py                # road-preserving filter, both sites
python tools/csf_filter.py                     # real CSF baseline (optional dependency)
python tools/dem_rebuild.py                    # adaptive-resolution DEM
python tools/lambda_c_sens_fair.py             # λc sweep under the final scoring protocol
python tools/seg_bootstrap.py                  # bootstrap confidence intervals
```

Reported values (fair point-level protocol, `d3 = 3.5 m`, `AGL = 2.0 m`):

| Site | P | R | IoU | skeleton F1 (2 px) | Buffer IoU (1 px) | line IoU (r = 2 m) |
|---|---|---|---|---|---|---|
| Yangjiaba | 0.7175 | 0.8264 | 0.6235 | 0.5274 | 0.6277 | 0.6453 |
| Xibacun | 0.9426 | 0.7709 | 0.7363 | 0.6610 | 0.7047 | 0.8049 |

Per-site metrics are reported side by side; they are **not** compared across sites, because the two reference corridors differ in width (median 4.47 m vs 7.21 m) and IoU is systematically more lenient for wider references.

## Tests

```bash
python tests/test_no_leak_guard.py
```

Two regression checks: (i) the adaptive-GSD builder refuses to run without an
explicit `--cloud` or `--prob` argument, so it can never silently fall back to
the reference corridor; (ii) `extract_path()` defaults to `allow_gt_roi=False`.
The first check is skipped when the derived grids are absent (data are not
distributed).

## Notes

* All scripts are runnable from the repository root; each inserts `tools/` on `sys.path` so that sibling modules import cleanly.
* Figures are regenerated into `figures/`; no bitmap data are shipped.
* Data availability statement of the paper: code open (MIT), raw point clouds and reference corridors restricted by the data provider.

## License

MIT — see `LICENSE`. The licence covers the source code only, not the LiDAR data or reference vectors.
