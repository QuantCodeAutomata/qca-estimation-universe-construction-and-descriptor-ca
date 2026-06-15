"""End-to-end monthly pipeline glue: panel -> universe -> descriptors ->
industry weights -> exposure matrix -> factor returns.

This is the orchestration layer used by all experiment runners in `exp/`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EXPOSURE_COLS, INDUSTRY_COLS, RISK_INDICES, ModelConfig
from .descriptors.compute import (
    build_lcapcb,
    build_risk_indices,
    compute_descriptors,
    normalize_descriptors,
)
from .factor_returns import estimate_factor_returns
from .industry_weights import compute_industry_weights
from .synthetic_data import SyntheticPanel
from .universe import UniverseState, construct_universe, init_state


@dataclass
class MonthlySnapshot:
    """Output of a single month T of the pipeline."""
    date: pd.Timestamp
    universe: pd.DataFrame              # permno-indexed (mktcap, industry, ...)
    descriptors_raw: pd.DataFrame
    normalized: pd.DataFrame
    risk_indices: pd.DataFrame
    industry_weights: pd.DataFrame
    exposure_matrix: pd.DataFrame       # N x 65
    factor_returns: np.ndarray | None = None
    specific_returns: pd.Series | None = None
    t_stats: np.ndarray | None = None
    r2: float | None = None
    n_assets: int = 0
    cond_num: float | None = None


def build_exposure_matrix(
    industry_w: pd.DataFrame, risk_idx: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate industry weights (52) and risk indices (13) into 65-col X."""
    common = industry_w.index.intersection(risk_idx.index)
    X = pd.concat([industry_w.loc[common], risk_idx.loc[common]], axis=1)
    X.columns = list(industry_w.columns) + list(risk_idx.columns)
    # Ensure exactly EXPOSURE_COLS ordering
    for c in EXPOSURE_COLS:
        if c not in X.columns:
            X[c] = 0.0
    return X[EXPOSURE_COLS]


def run_month(
    panel: SyntheticPanel, target_date: pd.Timestamp, state: UniverseState,
    cfg: ModelConfig, method: str = "gls",
) -> MonthlySnapshot:
    """Run one full month of the pipeline (no factor regression yet)."""
    universe = construct_universe(panel, target_date, state, cfg.universe_size)
    if universe.empty:
        return MonthlySnapshot(
            date=target_date, universe=universe,
            descriptors_raw=pd.DataFrame(), normalized=pd.DataFrame(),
            risk_indices=pd.DataFrame(), industry_weights=pd.DataFrame(),
            exposure_matrix=pd.DataFrame(),
        )

    descriptors = compute_descriptors(panel, target_date)
    descriptors = descriptors.reindex(universe.index)

    # Normalize using cap-weighted mean on ESTU
    estu_mask = universe["in_estu"]
    mktcap = universe["mktcap"]
    normalized = normalize_descriptors(descriptors, estu_mask, mktcap)

    # LCAPCB: cube of normalized LNCAP (two-pass per spec)
    normalized["LCAPCB"] = build_lcapcb(normalized)
    # Re-normalize LCAPCB itself (so std-dev=1 cross-sectionally)
    if normalized["LCAPCB"].std(ddof=0) > 1e-8:
        normalized["LCAPCB"] = (normalized["LCAPCB"] - normalized.loc[estu_mask, "LCAPCB"].mean()) \
            / normalized.loc[estu_mask, "LCAPCB"].std(ddof=0)

    nonestu = universe["nonestu"]
    risk_idx = build_risk_indices(normalized, nonestu)

    industry_w = compute_industry_weights(panel, universe, target_date)

    X = build_exposure_matrix(industry_w, risk_idx)

    return MonthlySnapshot(
        date=target_date, universe=universe,
        descriptors_raw=descriptors, normalized=normalized,
        risk_indices=risk_idx, industry_weights=industry_w,
        exposure_matrix=X, n_assets=len(universe),
    )


def run_full_pipeline(
    panel: SyntheticPanel, cfg: ModelConfig, method: str = "gls",
) -> tuple[list[MonthlySnapshot], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full monthly pipeline + factor regression.

    Returns
    -------
    snapshots : list[MonthlySnapshot]
    factor_returns_df : DataFrame (T x 65)
    spec_returns_df : DataFrame (T x N_total) — specific returns
    diagnostics_df : DataFrame (T x ?) — per-month r2, n_assets, cond_num
    """
    state = init_state()
    snapshots: list[MonthlySnapshot] = []
    factor_rows: list[dict] = []
    spec_rows: dict[pd.Timestamp, pd.Series] = {}
    diag_rows: list[dict] = []

    months = sorted(panel.monthly["date"].unique())
    rf_series = panel.rf.set_index("date")["rf"]
    sp_series = panel.sp500.set_index("date")["ret"]

    for dt in months:
        snap = run_month(panel, pd.Timestamp(dt), state, cfg, method=method)
        if snap.exposure_matrix.empty:
            snapshots.append(snap)
            continue

        # Returns for month T (use current-month ret)
        m_ret = (panel.monthly[panel.monthly["date"] == dt]
                 .set_index("permno")["ret"]).reindex(snap.exposure_matrix.index)
        mktcap_t = snap.universe["mktcap"]
        hsigma_t = snap.descriptors_raw.get("HSIGMA",
                                            pd.Series(np.nan, index=snap.exposure_matrix.index))
        rf_t = float(rf_series.loc[dt])
        sp_t = float(sp_series.loc[dt])

        result, spec = estimate_factor_returns(
            X=snap.exposure_matrix, returns=m_ret, rf=rf_t,
            hsigma=hsigma_t, mktcap=mktcap_t, universe=snap.universe,
            sp500_ret=sp_t, method=method,
        )
        snap.factor_returns = result.f
        snap.specific_returns = spec
        snap.t_stats = result.t_stats
        snap.r2 = result.r2
        snap.cond_num = result.cond_num

        factor_rows.append({"date": dt,
                            **dict(zip(EXPOSURE_COLS, result.f))})
        spec_rows[pd.Timestamp(dt)] = spec
        diag_rows.append({
            "date": dt, "r2": result.r2, "n_assets": result.n_assets,
            "cond_num": result.cond_num, "w_truncated": result.w_truncated,
            "frac_t_gt_2": float(np.mean(np.abs(result.t_stats) > 2.0)),
        })
        snapshots.append(snap)

    factor_returns_df = (pd.DataFrame(factor_rows).set_index("date")
                         if factor_rows else pd.DataFrame(columns=EXPOSURE_COLS))
    spec_returns_df = pd.DataFrame(spec_rows).T
    spec_returns_df.index.name = "date"
    diagnostics_df = (pd.DataFrame(diag_rows).set_index("date")
                      if diag_rows else pd.DataFrame())

    return snapshots, factor_returns_df, spec_returns_df, diagnostics_df
