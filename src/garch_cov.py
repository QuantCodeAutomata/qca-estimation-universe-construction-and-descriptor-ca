"""Extended asymmetric GARCH(1,1) and factor covariance matrix (exp_4).

Custom — Context7 confirmed the linear-eps term theta*eps_{t-1} is NOT directly
provided by `arch`; custom MLE via scipy.optimize is used. The symmetric baseline
uses the `arch` package (Context7 confirmed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .config import HALFLIFE_MONTHS


@dataclass
class GarchFit:
    params: dict[str, float]
    loglik: float
    n_obs: int
    n_params: int
    variances: np.ndarray  # filtered conditional variances
    aic: float
    bic: float


def _ext_garch_filter(params: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Filter the extended GARCH conditional variance series.

    Var_t = omega + alpha * eps_{t-1}^2 + beta * Var_{t-1} + theta * eps_{t-1}
    """
    omega, alpha, beta, theta = params
    T = len(eps)
    h = np.empty(T)
    if abs(1.0 - alpha - beta) > 1e-6:
        h[0] = max(omega / (1.0 - alpha - beta), 1e-8)
    else:
        h[0] = float(np.var(eps))
    for t in range(1, T):
        val = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1] + theta * eps[t - 1]
        h[t] = max(val, 1e-8)
    return h


def _ext_garch_negll(params: np.ndarray, eps: np.ndarray) -> float:
    omega, alpha, beta, theta = params
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e10
    h = _ext_garch_filter(params, eps)
    return 0.5 * np.sum(np.log(2 * np.pi * h) + eps ** 2 / h)


def fit_extended_garch(eps: np.ndarray) -> GarchFit:
    """Fit the extended asymmetric GARCH(1,1) by ML."""
    eps = np.asarray(eps, dtype=float)
    v0 = float(np.var(eps))
    x0 = np.array([v0 * 0.05, 0.05, 0.90, -0.005])
    bounds = [(1e-8, None), (0.0, 0.5), (0.0, 0.999), (-0.5, 0.5)]
    res = minimize(_ext_garch_negll, x0=x0, args=(eps,),
                   method="L-BFGS-B", bounds=bounds)
    omega, alpha, beta, theta = res.x
    h = _ext_garch_filter(res.x, eps)
    ll = -res.fun
    n = len(eps)
    k = 4
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    return GarchFit(
        params={"omega": float(omega), "alpha": float(alpha),
                "beta": float(beta), "theta": float(theta)},
        loglik=float(ll), n_obs=n, n_params=k,
        variances=h, aic=float(aic), bic=float(bic),
    )


def fit_symmetric_garch(eps: np.ndarray) -> GarchFit:
    """Fit symmetric GARCH(1,1) — uses arch package. Context7 confirmed."""
    # Using arch.arch_model — Context7 confirmed
    from arch import arch_model
    eps = np.asarray(eps, dtype=float)
    # Scale to percentage for numerical stability
    am = arch_model(eps * 100, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
    res = am.fit(disp="off", show_warning=False)
    # Recover variances on original scale
    cv = res.conditional_volatility
    cv = cv.values if hasattr(cv, "values") else np.asarray(cv)
    h = (cv / 100.0) ** 2
    return GarchFit(
        params={k: float(v) for k, v in res.params.to_dict().items()},
        loglik=float(res.loglikelihood), n_obs=len(eps), n_params=3,
        variances=h, aic=float(res.aic), bic=float(res.bic),
    )


def likelihood_ratio_test(sym: GarchFit, ext: GarchFit) -> dict:
    """LR test of symmetric vs extended GARCH (1 dof)."""
    lr = 2 * (ext.loglik - sym.loglik)
    p = float(1 - chi2.cdf(max(lr, 0), df=1))
    return {"LR": float(lr), "p_value": p,
            "AIC_sym": sym.aic, "AIC_ext": ext.aic,
            "BIC_sym": sym.bic, "BIC_ext": ext.bic}


# ----------------------------------------------------------------------
# Exponentially weighted factor covariance matrix
# ----------------------------------------------------------------------
def exponential_weights(T: int, halflife: int = HALFLIFE_MONTHS) -> np.ndarray:
    """Normalized exponential weights: most-recent observation weight = max."""
    delta = 0.5 ** (1.0 / halflife)
    w = np.array([delta ** (T - t) for t in range(1, T + 1)], dtype=float)
    return w / w.sum()


def factor_covariance(
    factor_returns: pd.DataFrame,
    halflife: int = HALFLIFE_MONTHS,
) -> np.ndarray:
    """Compute exponentially weighted factor covariance F_T.

    Parameters
    ----------
    factor_returns : DataFrame (T x K)
        Time series of monthly factor returns.
    halflife : int
        Half-life in months.

    Returns
    -------
    np.ndarray (K x K)
        PSD-projected factor covariance.
    """
    A = factor_returns.values.astype(float)
    T = A.shape[0]
    w = exponential_weights(T, halflife)
    # Demean (cap-weighted mean ~0 in factor returns under correct estim.)
    Aw = A * w[:, None]
    F = Aw.T @ A
    F = 0.5 * (F + F.T)  # symmetrize
    return nearest_psd(F)


def nearest_psd(A: np.ndarray) -> np.ndarray:
    """Project to nearest positive semi-definite matrix via eigenvalue floor."""
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T


def garch_scale_factor_cov(
    F: np.ndarray, x_market: np.ndarray, var_market_garch: float,
) -> tuple[np.ndarray, float]:
    """Scale F so model-implied market variance equals GARCH forecast.

    Returns scaled F and the scaling factor lambda.
    """
    model_var = float(x_market @ F @ x_market)
    if model_var <= 0:
        return F, 1.0
    lam = var_market_garch / model_var
    return F * lam, float(lam)
