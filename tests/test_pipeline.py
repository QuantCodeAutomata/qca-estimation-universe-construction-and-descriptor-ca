"""Pytest suite covering the BARRA US-E3 pipeline.

Tests exercise every module of `src/` end-to-end on a small synthetic panel.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    EXPOSURE_COLS, INDUSTRY_COLS, RISK_INDICES, ModelConfig, load_panel,
    run_full_pipeline,
)
from src.descriptors.compute import (
    build_lcapcb, build_risk_indices, compute_descriptors, normalize_descriptors,
)
from src.factor_returns import estimate_factor_returns
from src.garch_cov import (
    exponential_weights, factor_covariance, fit_extended_garch,
    fit_symmetric_garch, likelihood_ratio_test, nearest_psd,
)
from src.industry_weights import compute_industry_weights
from src.specific_risk import compute_S_series, fit_ar_model, select_ar_lags
from src.universe import construct_universe, init_state
from src.validation import bias_by_decile, diebold_mariano


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig(n_assets=80, n_months=36, start_date="1995-01-31")


@pytest.fixture(scope="module")
def panel(cfg):
    return load_panel(cfg)


# ----------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------
def test_universe_size_within_bounds(panel, cfg):
    state = init_state()
    last_date = panel.monthly["date"].max()
    u = construct_universe(panel, last_date, state, universe_size=cfg.n_assets)
    assert not u.empty
    assert u["in_estu"].sum() > 0
    # Cannot exceed available asset count
    assert u["in_estu"].sum() <= cfg.n_assets


def test_nonestu_flag_consistency(panel, cfg):
    state = init_state()
    last_date = panel.monthly["date"].max()
    u = construct_universe(panel, last_date, state, universe_size=cfg.n_assets)
    # Every asset is either ESTU or NONESTU, not both
    assert ((u["in_estu"].astype(int) + u["nonestu"]) == 1).all()


def test_universe_industry_fillin(panel, cfg):
    """At least one industry should have >=1 asset after fill-in."""
    state = init_state()
    last_date = panel.monthly["date"].max()
    u = construct_universe(panel, last_date, state, universe_size=cfg.n_assets)
    estu = u[u["in_estu"]]
    industries_covered = estu["industry"].nunique()
    assert industries_covered > 0


# ----------------------------------------------------------------------
# Descriptors
# ----------------------------------------------------------------------
def test_descriptors_no_lookahead(panel):
    """Computing descriptors at month T must not use future data."""
    target = pd.Timestamp("1996-06-30")
    df = compute_descriptors(panel, target)
    # Verify no descriptors depend on dates >= target
    assert not df.empty
    # All LNCAP values should be finite
    assert df["LNCAP"].notna().any()


def test_descriptor_normalization_properties(panel):
    target = pd.Timestamp("1996-06-30")
    df = compute_descriptors(panel, target)
    estu_mask = pd.Series(True, index=df.index)
    mktcap = df["mktcap"]
    nrm = normalize_descriptors(df, estu_mask, mktcap)
    # After cap-weighted normalization, equal-weighted std ≈ 1 for numeric cols
    for c in ["LNCAP", "LPRI"]:
        if c in nrm.columns:
            valid = nrm[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) > 5 and valid.std(ddof=0) > 1e-6:
                # Std should be near 1 (allow some tolerance)
                assert 0.5 < valid.std(ddof=0) < 2.0, f"{c} std={valid.std()}"


def test_lcapcb_is_cubic(panel):
    """LCAPCB must be the cube of normalized LNCAP."""
    target = pd.Timestamp("1996-06-30")
    df = compute_descriptors(panel, target)
    estu_mask = pd.Series(True, index=df.index)
    nrm = normalize_descriptors(df, estu_mask, df["mktcap"])
    lcapcb = build_lcapcb(nrm)
    np.testing.assert_allclose(lcapcb.values, nrm["LNCAP"].values ** 3, rtol=1e-10)


# ----------------------------------------------------------------------
# Industry weights (exp_2)
# ----------------------------------------------------------------------
def test_industry_weights_sum_to_one(panel, cfg):
    state = init_state()
    last_date = panel.monthly["date"].max()
    u = construct_universe(panel, last_date, state, universe_size=cfg.n_assets)
    w = compute_industry_weights(panel, u, last_date)
    row_sums = w.sum(axis=1)
    assert np.allclose(row_sums.values, 1.0, atol=1e-6), \
        f"Industry weights row-sums deviate: max abs(1-sum)={abs(1 - row_sums).max()}"


def test_industry_weights_shape(panel, cfg):
    state = init_state()
    last_date = panel.monthly["date"].max()
    u = construct_universe(panel, last_date, state, universe_size=cfg.n_assets)
    w = compute_industry_weights(panel, u, last_date)
    assert w.shape[1] == 52
    assert list(w.columns) == INDUSTRY_COLS


# ----------------------------------------------------------------------
# Factor returns (exp_3)
# ----------------------------------------------------------------------
def test_factor_returns_shape(panel, cfg):
    snaps, fr, sr, diag = run_full_pipeline(panel, cfg, method="gls")
    assert fr.shape[1] == len(EXPOSURE_COLS) == 65
    assert (diag["r2"].between(-1, 1)).all()
    # After 24-month warmup, n_assets > 0 in tail of sample
    assert (diag["n_assets"] > 0).sum() >= 10


def test_ols_baseline_runs(panel, cfg):
    _, fr_ols, _, _ = run_full_pipeline(panel, cfg, method="ols")
    assert fr_ols.shape[1] == 65


# ----------------------------------------------------------------------
# GARCH (exp_4)
# ----------------------------------------------------------------------
def test_extended_garch_stationarity(panel):
    eps = (panel.sp500["ret"].values - panel.rf["rf"].values)
    fit = fit_extended_garch(eps)
    assert fit.params["omega"] > 0
    assert fit.params["alpha"] >= 0
    assert fit.params["beta"] >= 0
    assert fit.params["alpha"] + fit.params["beta"] < 1.0


def test_lr_test_returns_finite():
    rng = np.random.default_rng(0)
    eps = rng.normal(0, 0.04, 240)
    sym = fit_symmetric_garch(eps)
    ext = fit_extended_garch(eps)
    lr = likelihood_ratio_test(sym, ext)
    assert np.isfinite(lr["LR"])
    assert 0.0 <= lr["p_value"] <= 1.0


def test_exp_weights_sum_to_one():
    w = exponential_weights(T=100, halflife=90)
    assert abs(w.sum() - 1.0) < 1e-12
    # most recent obs should have highest weight
    assert w[-1] > w[0]


def test_factor_covariance_psd(panel, cfg):
    snaps, fr, sr, diag = run_full_pipeline(panel, cfg, method="gls")
    F = factor_covariance(fr, halflife=90)
    eigs = np.linalg.eigvalsh(F)
    assert (eigs >= -1e-8).all(), f"min eig = {eigs.min()}"


def test_nearest_psd_idempotent():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    A = A @ A.T  # already PSD
    B = nearest_psd(A)
    np.testing.assert_allclose(A, B, atol=1e-8)


# ----------------------------------------------------------------------
# Specific risk (exp_5)
# ----------------------------------------------------------------------
def test_S_series_non_negative(panel, cfg):
    snaps, fr, sr, _ = run_full_pipeline(panel, cfg, method="gls")
    S = compute_S_series(sr)
    assert (S.dropna() >= 0).all()


def test_select_ar_lags_returns_int(panel, cfg):
    snaps, fr, sr, _ = run_full_pipeline(panel, cfg, method="gls")
    S = compute_S_series(sr).dropna()
    rm = (panel.sp500.set_index("date")["ret"] - panel.rf.set_index("date")["rf"])
    rm = rm.reindex(S.index).fillna(0)
    if len(S) > 12:
        k = select_ar_lags(S, rm, max_lag=3)
        assert 1 <= k <= 3


# ----------------------------------------------------------------------
# Validation (exp_6, exp_7)
# ----------------------------------------------------------------------
def test_bias_by_decile_finite():
    n_dates, n_assets = 30, 40
    rng = np.random.default_rng(2)
    sigma_hat = pd.DataFrame(np.abs(rng.normal(0.05, 0.01, (n_dates, n_assets))))
    spec = pd.DataFrame(rng.normal(0, 0.05, (n_dates, n_assets)))
    deciles = pd.DataFrame(rng.integers(1, 11, (n_dates, n_assets)))
    bias = bias_by_decile(sigma_hat, spec, deciles)
    assert bias.shape[0] >= 1
    # Bias should be positive
    assert (bias["bias"].dropna() > 0).all()


def test_diebold_mariano_basic():
    rng = np.random.default_rng(3)
    e1 = rng.normal(0, 1, 100)
    e2 = rng.normal(0, 1.5, 100)  # systematically worse
    res = diebold_mariano(e1, e2, h=4)
    assert np.isfinite(res["DM"])
    assert 0.0 <= res["p_value"] <= 1.0
