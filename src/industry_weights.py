"""Multi-industry weight calculation via three valuation models (exp_2).

Implements:
  * Sales-based, assets-based and operating-income-based regressions
    mktcap_n = sum_i fundamental_{n,i} * beta_{m,i} + e_n
  * Equal-weighted combination of the three weight vectors
  * Single-industry fallback (weight = 1.0 to primary industry)

Custom — Context7 confirmed no library performs this BARRA-style multi-industry
segment-attribution regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INDUSTRY_COLS, N_INDUSTRIES


def _solve_nn_ols(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative OLS via least-squares with negative-clip fallback."""
    if F.shape[0] < F.shape[1] or np.all(F == 0):
        return np.zeros(F.shape[1])
    beta, *_ = np.linalg.lstsq(F, y, rcond=None)
    # Negative betas indicate model instability — set to zero per spec
    beta = np.maximum(beta, 0.0)
    return beta


def _model_weights(seg_matrix: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute per-asset industry weights from one valuation model."""
    contrib = seg_matrix * beta[None, :]
    row_sum = contrib.sum(axis=1, keepdims=True)
    safe = np.where(row_sum > 1e-9, row_sum, 1.0)
    weights = contrib / safe
    weights = np.where(row_sum > 1e-9, weights, 0.0)
    return weights


def compute_industry_weights(
    panel,
    universe_df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute the I=52 industry weight matrix for each asset in the universe.

    Parameters
    ----------
    panel : SyntheticPanel
        Source data.
    universe_df : pd.DataFrame
        Universe with columns (mktcap, industry, in_estu, ...) indexed by permno.
    target_date : pd.Timestamp
        Month-end of month T (used to apply 90-day Compustat lag).

    Returns
    -------
    pd.DataFrame
        Indexed by permno, columns IND_00..IND_51, each row sums to 1.0.
    """
    link = panel.link.set_index("permno")["gvkey"]
    segments = panel.segments.copy()
    segments["available"] = segments["datadate"] + pd.Timedelta(days=90)
    segments = segments[segments["available"] <= target_date]
    if segments.empty:
        # Fallback: 100% primary industry
        out = pd.DataFrame(0.0, index=universe_df.index, columns=INDUSTRY_COLS)
        for permno, row in universe_df.iterrows():
            out.at[permno, f"IND_{int(row['industry']):02d}"] = 1.0
        return out

    # Use the most recent year of segments per gvkey
    latest_dd = (segments.groupby("gvkey")["datadate"].max()
                 .rename("latest_dd").reset_index())
    segments = segments.merge(latest_dd, on="gvkey")
    segments = segments[segments["datadate"] == segments["latest_dd"]]

    # Build (gvkey -> sum by industry) for each fundamental
    permnos = universe_df.index
    n_assets = len(permnos)
    gv_lookup = link.reindex(permnos)

    sales_mat = np.zeros((n_assets, N_INDUSTRIES))
    assets_mat = np.zeros((n_assets, N_INDUSTRIES))
    oi_mat = np.zeros((n_assets, N_INDUSTRIES))

    seg_groups = segments.groupby(["gvkey", "industry"]).agg({
        "sales": "sum", "seg_at": "sum", "oiadp": "sum"
    }).reset_index()

    permno_to_idx = {p: i for i, p in enumerate(permnos)}
    gv_to_permno = {gv: p for p, gv in gv_lookup.items() if not pd.isna(gv)}

    for _, row in seg_groups.iterrows():
        gv = row["gvkey"]
        if gv not in gv_to_permno:
            continue
        permno = gv_to_permno[gv]
        if permno not in permno_to_idx:
            continue
        i = permno_to_idx[permno]
        j = int(row["industry"])
        sales_mat[i, j] += float(row["sales"])
        assets_mat[i, j] += float(row["seg_at"])
        oi_mat[i, j] += float(row["oiadp"])

    mktcap = universe_df["mktcap"].reindex(permnos).fillna(0.0).values

    # Filter to multi-segment companies for regression
    n_segments_per_asset = (sales_mat > 0).sum(axis=1)
    multi = n_segments_per_asset >= 2

    if multi.sum() >= 10:
        beta_sales = _solve_nn_ols(sales_mat[multi], mktcap[multi])
        beta_assets = _solve_nn_ols(assets_mat[multi], mktcap[multi])
        beta_oi = _solve_nn_ols(oi_mat[multi], mktcap[multi])
    else:
        beta_sales = np.ones(N_INDUSTRIES)
        beta_assets = np.ones(N_INDUSTRIES)
        beta_oi = np.ones(N_INDUSTRIES)

    w_sales = _model_weights(sales_mat, beta_sales)
    w_assets = _model_weights(assets_mat, beta_assets)
    w_oi = _model_weights(oi_mat, beta_oi)

    # Fallback when OI denominator <= 0 → use sales weights for that asset
    bad_oi = (w_oi.sum(axis=1) < 1e-6)
    w_oi[bad_oi] = w_sales[bad_oi]

    combined = (w_sales + w_assets + w_oi) / 3.0

    # Renormalize to enforce row-sum = 1
    row_sum = combined.sum(axis=1, keepdims=True)
    # Single-industry / no-segment assets → primary industry
    no_seg = (row_sum.flatten() < 1e-6)
    combined = np.where(row_sum > 1e-6, combined / row_sum, 0.0)
    for i in np.where(no_seg)[0]:
        permno = permnos[i]
        primary = int(universe_df.at[permno, "industry"])
        combined[i, primary] = 1.0

    out = pd.DataFrame(combined, index=permnos, columns=INDUSTRY_COLS)
    # Final QC: every row sums to 1
    rs = out.sum(axis=1)
    out = out.div(rs.where(rs > 0, 1.0), axis=0)
    return out
