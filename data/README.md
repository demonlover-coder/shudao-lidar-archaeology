# data/ — not distributed

Raw and derived data are **not** part of this repository.

* Airborne LiDAR point clouds (`*.las`) and the original 5–10 cm DEMs are not
  included and are not redistributable.
* Reference corridors (SHP) were digitised from historical records and
  archaeological expert knowledge; they are used for scoring only and are not
  distributed either.
* Derived products (1 m `dem_min` grids, probability maps, delivered point
  indices, adaptive-resolution DEM) can be obtained from the corresponding
  author on reasonable request.

To run the code, either:

1. put the data under this directory following the layout below, or
2. set the environment variable `SHUDAO_DATA_ROOT` to any directory with the
   same layout.

```
<DATA_ROOT>/
├── survey-shp/
│   ├── 杨家坝/{yangjiaba.las, 杨家河坝碥道-dem.tif,
│   │            杨家坝_蜀道_中轴线.shp, 杨家坝_蜀道_本体范围.shp}
│   └── 西坝村/{西坝村栈道-lidar.las, 西坝村栈道-dem.tif,
│              XBC_orig_clean.shp}
└── data_real/        # 1 m grids, truth masks ({site}_label.npy), etc.
```

Reference vectors (SHP) are read **only** by the evaluation scripts.
