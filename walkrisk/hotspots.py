"""Step 2. Where do elderly walking and crashes cluster together?

Local Moran's I on each variable, Queen contiguity with row-standardised weights,
BH-FDR at 0.05. A dong is a common HH hotspot only if it is HH and significant for
both variables under every random seed. With 999 permutations the result moved
between 0 and 8 dongs from seed to seed; with 9,999 permutations x 5 seeds it was
stable at 4.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran_Local
from libpysal.weights import Queen
from statsmodels.stats.multitest import multipletests


def hh_mask(values: np.ndarray, weights, *, permutations: int, seed: int, alpha: float = 0.05) -> np.ndarray:
    result = Moran_Local(values, weights, permutations=permutations, seed=seed, geoda_quads=True)
    significant = multipletests(result.p_sim, alpha=alpha, method="fdr_bh")[0]
    return (result.q == 1) & significant


def common_hh(dong: gpd.GeoDataFrame, *, expo_col="expo", crash_col="crash", seeds=range(5), permutations=9999, alpha=0.05) -> pd.DataFrame:
    """Per-dong counts of how many seeds flagged it HH for exposure, crashes, and both."""
    weights = Queen.from_dataframe(dong, use_index=False, silence_warnings=True)
    weights.transform = "r"
    counts = {"expo_hh": np.zeros(len(dong), int), "crash_hh": np.zeros(len(dong), int), "both_hh": np.zeros(len(dong), int)}
    seeds = list(seeds)
    for seed in seeds:
        hh = {
            col: hh_mask(dong[col].to_numpy(float), weights, permutations=permutations, seed=seed, alpha=alpha)
            for col in (expo_col, crash_col)
        }
        counts["expo_hh"] += hh[expo_col]
        counts["crash_hh"] += hh[crash_col]
        counts["both_hh"] += hh[expo_col] & hh[crash_col]
    out = pd.DataFrame(counts, index=dong.index)
    out["stable_common_hh"] = out["both_hh"] == len(seeds)
    return out
