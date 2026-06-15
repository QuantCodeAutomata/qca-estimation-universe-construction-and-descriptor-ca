"""Validation / bias analysis (exp_6, exp_7).

Includes:
  * Capitalization-decile bias analysis (Model A with LCAPCB vs Model B without)
  * Random portfolio simulations
  * Cross-sectional R^2 summary, factor t-stat summary
  * Diebold-Mariano test for US-E3 vs US-E2 analog
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def bias_by_decile(
    sigma_hat_panel: pd.DataFrame, spec_returns_panel: pd.DataFrame,
    decile_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Bias_d for each capitalization decile.

    Parameters
    ----------
    sigma_hat_panel : DataFrame (date x permno)
        Predicted specific risk forecasts.
    spec_returns_panel : DataFrame (date x permno)
        Realized specific returns.
    decile_panel : DataFrame (date x permno)
        Capitalization decile assignment per asset per month.

    Returns
    -------
    DataFrame indexed by decile (1..10) with columns:
        bias, ci_lo, ci_hi, n_obs.
    """
    out = []
    for d in range(1, 11):
        mask = (decile_panel == d)
        sig = sigma_hat_panel.where(mask).values
        u = spec_returns_panel.where(mask).values
        ratio2 = (u ** 2) / np.where(sig > 0, sig ** 2, np.nan)
        flat = ratio2[np.isfinite(ratio2)]
        if len(flat) == 0:
            out.append({"decile": d, "bias": np.nan,
                        "ci_lo": np.nan, "ci_hi": np.nan, "n_obs": 0})
            continue
        bias = float(np.sqrt(flat.mean()))
        # Bootstrap 95% CI
        rng = np.random.default_rng(42)
        boots = [np.sqrt(rng.choice(flat, size=len(flat), replace=True).mean())
                 for _ in range(500)]
        out.append({
            "decile": d, "bias": bias,
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n_obs": int(len(flat)),
        })
    return pd.DataFrame(out).set_index("decile")


def random_portfolio_bias(
    sigma_hat_panel: pd.DataFrame, spec_returns_panel: pd.DataFrame,
    portfolio_sizes: list[int] = (5, 10, 20, 50, 100, 200),
    n_simulations: int = 100, seed: int = 42,
) -> pd.DataFrame:
    """Estimate bias as a function of portfolio size."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = sigma_hat_panel.index
    for N in portfolio_sizes:
        ratios = []
        for _ in range(n_simulations):
            for dt in dates:
                avail = sigma_hat_panel.loc[dt].dropna().index
                if len(avail) < N:
                    continue
                pick = rng.choice(avail, size=N, replace=False)
                # Equal-weighted portfolio predicted variance (specific only)
                pred_var = (sigma_hat_panel.loc[dt, pick] ** 2).mean() / N
                real_var = (spec_returns_panel.loc[dt, pick] ** 2).mean() / N
                if pred_var > 0 and np.isfinite(real_var):
                    ratios.append(real_var / pred_var)
        if ratios:
            arr = np.array(ratios)
            rows.append({
                "N": N, "mean_bias": float(np.sqrt(np.mean(arr))),
                "p05": float(np.sqrt(np.percentile(arr, 5))),
                "p95": float(np.sqrt(np.percentile(arr, 95))),
                "n_obs": len(arr),
            })
        else:
            rows.append({"N": N, "mean_bias": np.nan, "p05": np.nan,
                         "p95": np.nan, "n_obs": 0})
    return pd.DataFrame(rows).set_index("N")


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano test on squared loss differential.

    H0: equal predictive accuracy.
    """
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return {"DM": np.nan, "p_value": np.nan, "n": len(d)}
    d_bar = float(d.mean())
    # Newey-West variance with lag h
    T = len(d)
    gamma0 = float(np.var(d, ddof=1))
    var_d = gamma0
    for lag in range(1, h + 1):
        gamma = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        var_d += 2.0 * (1.0 - lag / (h + 1)) * gamma
    se = np.sqrt(max(var_d / T, 1e-12))
    dm = d_bar / se
    p = 2 * (1 - norm.cdf(abs(dm)))
    return {"DM": float(dm), "p_value": float(p), "n": int(T)}


def vif_table(X: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor per column."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    A = X.values.astype(float)
    return pd.Series(
        [variance_inflation_factor(A, i) for i in range(A.shape[1])],
        index=X.columns, name="VIF",
    )
