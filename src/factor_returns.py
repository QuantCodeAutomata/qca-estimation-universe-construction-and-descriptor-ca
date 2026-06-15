"""GLS cross-sectional factor return estimation (exp_3).

Implements:
  * Inverse-residual-variance GLS weights with 99th-percentile truncation
  * Cross-sectional regression r_t = X_t f_t + u_t
  * NONESTU factor return as cap-weighted average specific return of non-ESTU
  * OLS variant for H6 testing
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import GLS_TRUNCATION_PCT, RET_SCREEN_LOWER, RET_SCREEN_UPPER


@dataclass
class FactorRegressionResult:
    """Output of one monthly GLS regression."""
    f: np.ndarray        # K-vector of factor returns
    u: np.ndarray        # N-vector of specific returns
    t_stats: np.ndarray  # K-vector of t-statistics
    r2: float            # cross-sectional R^2
    cond_num: float      # condition number of X' W X
    n_assets: int
    w_truncated: int     # count of weight-truncated assets


def _gls(
    X: np.ndarray, r: np.ndarray, w: np.ndarray,
) -> FactorRegressionResult:
    """One GLS regression. X: NxK, r: N, w: N."""
    N, K = X.shape
    W = w
    XtW = X.T * W
    XtWX = XtW @ X
    XtWr = XtW @ r
    try:
        f = np.linalg.solve(XtWX, XtWr)
    except np.linalg.LinAlgError:
        f = np.linalg.lstsq(XtWX, XtWr, rcond=None)[0]
    u = r - X @ f
    dof = max(N - K, 1)
    sigma2 = float((u * W * u).sum() / dof)
    try:
        cov_f = sigma2 * np.linalg.inv(XtWX)
        diag = np.maximum(np.diag(cov_f), 1e-18)
        t_stats = f / np.sqrt(diag)
    except np.linalg.LinAlgError:
        t_stats = np.zeros(K)
    r_demean = r - r.mean()
    ss_tot = float(r_demean @ r_demean)
    ss_res = float(u @ u)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    cond = float(np.linalg.cond(XtWX))
    return FactorRegressionResult(f=f, u=u, t_stats=t_stats, r2=r2,
                                  cond_num=cond, n_assets=N, w_truncated=0)


def estimate_factor_returns(
    X: pd.DataFrame,
    returns: pd.Series,
    rf: float,
    hsigma: pd.Series,
    mktcap: pd.Series,
    universe: pd.DataFrame,
    sp500_ret: float,
    method: str = "gls",
) -> tuple[FactorRegressionResult, pd.Series]:
    """Run one monthly cross-sectional regression on the estimation universe.

    Parameters
    ----------
    X : DataFrame (permno x K)
        Full exposure matrix (industries + risk indices, including NONESTU).
    returns : Series (permno)
        Total monthly returns r_it.
    rf : float
        Risk-free rate for this month.
    hsigma : Series (permno)
        Residual std deviation from 60-month CAPM (for GLS weights).
    mktcap : Series (permno)
        Market capitalization.
    universe : DataFrame (permno-indexed)
        Output of construct_universe; provides in_estu / nonestu flags.
    sp500_ret : float
        S&P 500 monthly return (for screening).
    method : str
        'gls' or 'ols'.

    Returns
    -------
    FactorRegressionResult, pd.Series (full N specific returns indexed by permno)
    """
    excess = (returns - rf).rename("excess")

    # Screening (Step 3)
    lo = sp500_ret + RET_SCREEN_LOWER
    hi = sp500_ret + RET_SCREEN_UPPER
    screened = (excess >= lo) & (excess <= hi)

    estu_mask = universe["in_estu"].reindex(X.index, fill_value=False)
    valid = (estu_mask & screened
             & hsigma.reindex(X.index).notna()
             & mktcap.reindex(X.index).fillna(0).gt(0)
             & excess.reindex(X.index).notna())

    idx = X.index[valid.values]
    K = X.shape[1]

    if len(idx) < K + 5:
        # Insufficient data → empty regression; return zero factor returns
        result = FactorRegressionResult(
            f=np.zeros(K), u=np.zeros(len(X.index)),
            t_stats=np.zeros(K), r2=0.0, cond_num=np.nan,
            n_assets=len(idx), w_truncated=0,
        )
        spec = pd.Series(0.0, index=X.index, name="specific_return")
        return result, spec

    X_e = X.loc[idx].values.astype(float)
    r_e = excess.reindex(idx).values.astype(float)
    h = hsigma.reindex(idx).fillna(hsigma.median()).values.astype(float)

    # GLS weights
    if method == "gls":
        w = 1.0 / np.maximum(h ** 2, 1e-8)
        cap = float(np.percentile(w, GLS_TRUNCATION_PCT))
        w_trunc_count = int((w > cap).sum())
        w = np.minimum(w, cap)
    else:
        w = np.ones_like(h)
        w_trunc_count = 0

    result = _gls(X_e, r_e, w)
    result.w_truncated = w_trunc_count

    # Compute specific returns over ALL assets
    full_u = (excess.reindex(X.index).fillna(0.0).values
              - X.values.astype(float) @ result.f)
    spec = pd.Series(full_u, index=X.index, name="specific_return")

    # Override: NONESTU factor return as cap-weighted average u of non-ESTU
    nonestu_mask = universe["nonestu"].reindex(X.index, fill_value=0).astype(bool)
    if nonestu_mask.sum() > 0:
        mc = mktcap.reindex(X.index).fillna(0)
        u_ne = spec[nonestu_mask]
        w_ne = mc[nonestu_mask]
        if w_ne.sum() > 0:
            f_nonestu = float((u_ne * w_ne).sum() / w_ne.sum())
            # NONESTU is the last column by config; locate it
            if "NONESTU" in X.columns:
                k = list(X.columns).index("NONESTU")
                result.f[k] = f_nonestu
    return result, spec
