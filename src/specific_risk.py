"""Two-component specific risk model (exp_5).

* Time-series AR(k) model for S_t = mean cross-sectional specific risk
* Pooled cross-sectional V_it = sum_k X_ikt gamma_k + e_it
* Piece-wise linear capitalization-decile scaling
* Final forecast: sigma_hat_it = scale_it * S_hat_t * (1 + V_hat_it)

Custom — Context7 confirmed no library implements the BARRA two-component
specific-risk forecast (statsmodels is used for the underlying AR/OLS pieces).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.stats import mstats


@dataclass
class SpecificRiskModel:
    """Fitted two-component specific risk model."""
    ar_params: np.ndarray              # [const, beta_1..beta_k, beta_market]
    ar_lags: int
    gamma: np.ndarray                  # K-vector of risk-index coefficients
    risk_index_cols: list[str]
    scale_function: interp1d
    s_mean: float                      # historical mean of S_t
    one_component_S: float             # for the one-component benchmark


def compute_S_series(specific_returns: pd.DataFrame) -> pd.Series:
    """S_t = sqrt(mean_i(u_it^2)) per month."""
    return np.sqrt((specific_returns ** 2).mean(axis=1)).rename("S_t")


def select_ar_lags(s: pd.Series, r_m: pd.Series, max_lag: int = 5) -> int:
    """Select optimal AR lag length via AIC."""
    best_k, best_aic = 1, np.inf
    for k in range(1, max_lag + 1):
        try:
            Y, X = _ar_design(s, r_m, k)
            if len(Y) < k + 5:
                continue
            res = sm.OLS(Y, X).fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_k = k
        except Exception:
            continue
    return best_k


def _ar_design(s: pd.Series, r_m: pd.Series, lags: int):
    T = len(s)
    Y = s.values[lags:]
    cols = [s.shift(k).values[lags:] for k in range(1, lags + 1)]
    cols.append(r_m.shift(1).values[lags:])
    X = np.column_stack(cols)
    X = sm.add_constant(X)
    return Y, X


def fit_ar_model(s: pd.Series, r_m: pd.Series, lags: int) -> np.ndarray:
    Y, X = _ar_design(s, r_m, lags)
    res = sm.OLS(Y, X).fit()
    return res.params


def fit_cross_sectional(
    spec_returns: pd.DataFrame, S_series: pd.Series,
    exposures: dict[pd.Timestamp, pd.DataFrame],
    risk_index_cols: list[str], winsor: float = 0.025,
) -> np.ndarray:
    """Pooled cross-sectional regression V_it = X_ikt gamma_k + e_it."""
    rows_V, rows_X = [], []
    for dt, S_t in S_series.dropna().items():
        if dt not in exposures:
            continue
        u_t = spec_returns.loc[dt].dropna()
        if u_t.empty or S_t <= 0:
            continue
        V_t = (np.abs(u_t) / S_t - 1.0)
        X_t = exposures[dt].reindex(V_t.index)[risk_index_cols].dropna()
        common = V_t.index.intersection(X_t.index)
        if len(common) == 0:
            continue
        rows_V.append(V_t.loc[common].values)
        rows_X.append(X_t.loc[common].values)
    if not rows_V:
        return np.zeros(len(risk_index_cols))
    V = np.concatenate(rows_V)
    X = np.concatenate(rows_X, axis=0)
    # Winsorize
    V = mstats.winsorize(V, limits=[winsor, winsor]).data
    res = sm.OLS(V, X).fit()
    return res.params


def fit_decile_scaling(
    spec_returns: pd.DataFrame, S_series: pd.Series,
    gamma: np.ndarray, exposures: dict[pd.Timestamp, pd.DataFrame],
    risk_index_cols: list[str], universe_by_date: dict[pd.Timestamp, pd.DataFrame],
) -> interp1d:
    """Build piece-wise linear capitalization-decile scaling correction."""
    bias_acc = {d: [] for d in range(1, 11)}
    for dt, S_t in S_series.dropna().items():
        if dt not in exposures or dt not in universe_by_date:
            continue
        u_t = spec_returns.loc[dt].dropna()
        if u_t.empty or S_t <= 0:
            continue
        X_t = exposures[dt][risk_index_cols].reindex(u_t.index)
        V_hat = X_t.values @ gamma
        sigma_hat = S_t * (1.0 + V_hat)
        sigma_realized = np.abs(u_t.values)
        deciles = universe_by_date[dt]["decile"].reindex(u_t.index)
        for d in range(1, 11):
            mask = (deciles == d).values
            if mask.sum() == 0:
                continue
            ratios = (sigma_realized[mask] / np.maximum(sigma_hat[mask], 1e-6))
            ratios = ratios[np.isfinite(ratios)]
            if len(ratios) > 0:
                bias_acc[d].extend(ratios.tolist())
    midpoints = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], dtype=float)
    corrections = np.array([np.mean(bias_acc[d]) if bias_acc[d] else 1.0
                            for d in range(1, 11)])
    return interp1d(midpoints, corrections, kind="linear",
                    fill_value=(corrections[0], corrections[-1]),
                    bounds_error=False)


def forecast_specific_risk(
    model: SpecificRiskModel, exposures: pd.DataFrame,
    s_history: pd.Series, r_m_lag: float, cap_percentiles: pd.Series,
) -> pd.Series:
    """Forecast sigma_hat_it for one month."""
    p = model.ar_params
    k = model.ar_lags
    s_lags = s_history.tail(k).values[::-1]
    S_hat = p[0] + np.dot(p[1:1 + k], s_lags) + p[1 + k] * r_m_lag
    S_hat = max(float(S_hat), 1e-6)
    V_hat = exposures[model.risk_index_cols].values @ model.gamma
    scales = model.scale_function(cap_percentiles.reindex(exposures.index).values)
    sigma_hat = np.maximum(scales * S_hat * (1.0 + V_hat), 1e-6)
    return pd.Series(sigma_hat, index=exposures.index, name="sigma_hat")
