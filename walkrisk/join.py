"""Step 1. Join 50 m flow-population cells to legal-dong polygons and build the dong x month panel.

Cells are assigned by the dong that contains their centre point. A cell whose centre
falls in no dong, or on a shared boundary, is assigned by the largest overlap of its
50 m square instead. Cells that overlap no dong are dropped.

Missing months are not filled with zero: a cell with no row in a month may still have
had people walking. Each dong x month keeps an observation rate instead.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ELDERLY_COLS = ["m65_69", "m70_plus", "f65_69", "f70_plus"]
WALK = 1  # flow type: 1 = walking, 2 = driving


def cell_points(cells: pd.DataFrame, *, cell_col: str, x_col: str, y_col: str, cell_crs, to_crs) -> gpd.GeoDataFrame:
    """One point per cell. Coordinates drift in the last digits between months, so use the median."""
    coords = cells.groupby(cell_col)[[x_col, y_col]].median()
    points = gpd.GeoDataFrame(
        coords,
        geometry=gpd.points_from_xy(coords[x_col], coords[y_col]),
        crs=cell_crs,
    ).to_crs(to_crs)
    points.index.name = cell_col
    return points


def assign_by_centre(points: gpd.GeoDataFrame, dong: gpd.GeoDataFrame, *, cell_col: str, dong_col: str = "EMD_CD"):
    """Assign each cell to the dong containing its centre. Returns (assigned, exception_ids)."""
    joined = gpd.sjoin(
        points.reset_index()[[cell_col, "geometry"]],
        dong[[dong_col, "geometry"]],
        how="left",
        predicate="within",
    )
    match_count = joined.groupby(cell_col)[dong_col].nunique().reindex(points.index, fill_value=0)
    single = (
        joined[joined[cell_col].isin(match_count[match_count == 1].index)]
        .drop_duplicates(cell_col)
        .set_index(cell_col)[dong_col]
    )
    exception_ids = match_count[match_count != 1].index
    return single, exception_ids


def assign_by_area(points: gpd.GeoDataFrame, dong: gpd.GeoDataFrame, ids, *, dong_col: str = "EMD_CD", size: float = 50.0) -> pd.Series:
    """Assign the exception cells by the dong that covers most of their square. CRS must be in metres."""
    half = size / 2
    sub = points.loc[ids]
    squares = gpd.GeoDataFrame(
        {"cell": sub.index},
        geometry=[box(p.x - half, p.y - half, p.x + half, p.y + half) for p in sub.geometry],
        crs=points.crs,
    )
    pieces = gpd.overlay(squares, dong[[dong_col, "geometry"]], how="intersection", keep_geom_type=True)
    if pieces.empty:
        return pd.Series(dtype=object, name=dong_col)
    pieces["area"] = pieces.area
    best = pieces.sort_values("area", ascending=False).drop_duplicates("cell")
    return best.set_index("cell")[dong_col]


def assign_cells(cells, dong, *, cell_col, x_col, y_col, cell_crs, dong_col="EMD_CD", size=50.0) -> pd.Series:
    """Cell id -> dong code, centre first and area for the rest. Unassignable cells are left out."""
    points = cell_points(cells, cell_col=cell_col, x_col=x_col, y_col=y_col, cell_crs=cell_crs, to_crs=dong.crs)
    single, exceptions = assign_by_centre(points, dong, cell_col=cell_col, dong_col=dong_col)
    by_area = assign_by_area(points, dong, exceptions, dong_col=dong_col, size=size)
    out = pd.concat([single, by_area])
    out.index.name = cell_col
    return out.rename(dong_col)


def dong_month_panel(cells, crashes, cell_to_dong, *, cell_col, month_col="month", dong_col="EMD_CD", months=None) -> pd.DataFrame:
    """Dong x month rows: elderly walking exposure, observation rate and elderly pedestrian crash count.

    exposure = sum over observed cells of the four 65+ columns (NaN if any of the four is missing)
    obs_rate = cells with a value that month / cells assigned to the dong
    crash    = crash records that month (0 when there are none)
    """
    walk = cells.loc[cells["flow_type"] == WALK].copy()
    walk["exposure"] = walk[ELDERLY_COLS].sum(axis=1, min_count=len(ELDERLY_COLS))
    walk[dong_col] = walk[cell_col].map(cell_to_dong)
    walk = walk.dropna(subset=[dong_col])

    if months is None:
        months = sorted(walk[month_col].unique())
    dongs = sorted(cell_to_dong.unique())
    grid = pd.MultiIndex.from_product([dongs, months], names=[dong_col, month_col])

    agg = walk.groupby([dong_col, month_col]).agg(
        exposure=("exposure", "sum"),
        observed=("exposure", "count"),
    )
    base = cell_to_dong.value_counts().rename("base_cells")
    panel = agg.reindex(grid).reset_index().merge(base, left_on=dong_col, right_index=True)
    panel["observed"] = panel["observed"].fillna(0)
    panel["obs_rate"] = panel["observed"] / panel["base_cells"]

    crash = crashes.groupby([dong_col, month_col]).size().rename("crash")
    panel = panel.merge(crash, left_on=[dong_col, month_col], right_index=True, how="left")
    panel["crash"] = panel["crash"].fillna(0).astype(int)
    return panel[[dong_col, month_col, "exposure", "obs_rate", "crash"]]
