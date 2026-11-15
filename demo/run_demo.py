"""Run the whole pipeline on synthetic data.

The real inputs (LG U+ flow population, KoROAD crash records) can only be opened inside
the Data Safe Zone, so this script builds a fake city with the same columns and runs
every step end to end. The numbers it prints say nothing about Seoul.

    python -m demo.run_demo            # quick: 999 permutations, 2 seeds
    python -m demo.run_demo --full     # same settings as the submission: 9,999 x 5
"""
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from walkrisk.forecast import model_frame, monthly_retrain, scores
from walkrisk.hotspots import common_hh
from walkrisk.join import ELDERLY_COLS, assign_cells, dong_month_panel

CRS = "EPSG:5186"  # Korea central belt TM, metres (the boundary file's CRS)
DONG_SIZE, CELL = 400.0, 50.0
GRID = 8  # 8 x 8 dongs, 64 cells each
MONTHS = list(range(202401, 202413))


def fake_city(rng: np.random.Generator):
    dong = gpd.GeoDataFrame(
        {"EMD_CD": [f"D{r}{c}" for r in range(GRID) for c in range(GRID)]},
        geometry=[box(c * DONG_SIZE, r * DONG_SIZE, (c + 1) * DONG_SIZE, (r + 1) * DONG_SIZE) for r in range(GRID) for c in range(GRID)],
        crs=CRS,
    )
    n = int(GRID * DONG_SIZE / CELL)
    xs, ys = np.meshgrid((np.arange(n) + 0.5) * CELL, (np.arange(n) + 0.5) * CELL)
    xs, ys = xs.ravel(), ys.ravel()
    # two walking hotspots, one of them more dangerous per step
    centres = [(900, 900, 1.0, 1.0), (2300, 2200, 0.8, 1.8)]
    walk = sum(a * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 450**2)) for cx, cy, a, _ in centres) + 0.05
    danger = 1 + sum((d - 1) * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 450**2)) for cx, cy, _, d in centres)

    rows = []
    for month in MONTHS:
        seen = rng.random(len(xs)) > 0.03  # ~3% of cells missing each month
        for flow_type in (1, 2):
            level = walk[seen] * (1.0 if flow_type == 1 else 0.4) * rng.lognormal(0, 0.1, seen.sum())
            frame = pd.DataFrame({
                "month": month,
                "cell_id": np.flatnonzero(seen),
                "x": xs[seen] + rng.normal(0, 1e-9, seen.sum()),
                "y": ys[seen],
                "flow_type": flow_type,
            })
            for share, col in zip((0.3, 0.2, 0.3, 0.2), ELDERLY_COLS):
                frame[col] = level * share * 10
            rows.append(frame)
    cells = pd.concat(rows, ignore_index=True)

    cell_dong = dong.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=CRS), predicate="contains")
    rate = pd.Series(walk * danger, index=range(len(xs))).groupby(cell_dong.set_index("index_right")["EMD_CD"]).sum()
    crashes = []
    for month in MONTHS:
        counts = rng.poisson(rate.to_numpy() * 0.06)
        crashes += [{"EMD_CD": d, "month": month} for d, k in zip(rate.index, counts) for _ in range(k)]
    return dong, cells, pd.DataFrame(crashes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="9,999 permutations x 5 seeds")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    dong, cells, crashes = fake_city(rng)
    print(f"fake city: {len(dong)} dongs, {cells['cell_id'].nunique()} cells, {len(cells):,} cell-month rows, {len(crashes)} crashes")

    cell_to_dong = assign_cells(cells, dong, cell_col="cell_id", x_col="x", y_col="y", cell_crs=CRS)
    panel = dong_month_panel(cells, crashes, cell_to_dong, cell_col="cell_id")
    print(f"step 1  joined {len(cell_to_dong)} cells -> panel of {len(panel)} dong-month rows, mean obs rate {panel['obs_rate'].mean():.3f}")

    analysis = panel[panel["month"] >= 202404]
    per_dong = analysis.groupby("EMD_CD").agg(expo=("exposure", "mean"), crash=("crash", "sum"))
    geo = dong.merge(per_dong, left_on="EMD_CD", right_index=True)
    perms, seeds = (9999, range(5)) if args.full else (999, range(2))
    hh = common_hh(geo, permutations=perms, seeds=seeds)
    print(f"step 2  {int((hh['expo_hh'] == len(seeds)).sum())} walking HH, {int((hh['crash_hh'] == len(seeds)).sum())} crash HH, "
          f"{int(hh['stable_common_hh'].sum())} common HH ({perms} permutations x {len(seeds)} seeds)")

    # January-March only feed the cumulative crash mean; training targets start in May
    frame = model_frame(panel)
    evaluation = monthly_retrain(frame, range(202407, 202413), train_from=202405)
    s = scores(evaluation)
    print("step 3  monthly retrain, July-December:")
    for key, value in s.items():
        print(f"        {key:<20} {value:.3f}")


if __name__ == "__main__":
    main()
