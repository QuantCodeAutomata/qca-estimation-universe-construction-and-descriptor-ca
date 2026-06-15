"""Deterministic synthetic CRSP/Compustat-like panel generator.

Produces all the inputs required by the BARRA US-E3 pipeline when no live data
source (Massive / WRDS) is available. The structure of the output dataframes
matches what the rest of the code expects, so swapping in real data is a drop-in
replacement.

The simulated data is **not** real market data — it is used to exercise and
validate the full algorithmic pipeline end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ModelConfig, N_INDUSTRIES


@dataclass
class SyntheticPanel:
    """Container for all simulated input panels."""
    monthly: pd.DataFrame              # permno, date, ret, prc, shrout, vol
    daily: pd.DataFrame                # permno, date, ret, prc, vol, high, low
    fundamentals_a: pd.DataFrame       # gvkey, datadate, ceq, dltt, dlc, at, ...
    fundamentals_q: pd.DataFrame       # gvkey, datadate, epsq, cshoq
    segments: pd.DataFrame             # gvkey, datadate, sic, sales, at, oiadp
    ibes: pd.DataFrame                 # permno, statpers, medest_fy1, medest_fy2, stdev_fy1, stdev_fy2
    sp500: pd.DataFrame                # date, ret  (S&P 500 total return)
    rf: pd.DataFrame                   # date, rf
    fx: pd.DataFrame                   # date, fx_ret (trade-weighted broad index pct change)
    ratings: pd.DataFrame              # permno, rating_int
    link: pd.DataFrame                 # permno <-> gvkey
    industry_map: pd.DataFrame         # permno -> primary industry (0..51)


def _month_ends(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="ME")


def _make_industries(rng: np.random.Generator, n_assets: int) -> np.ndarray:
    """Assign each permno a primary industry index 0..51, roughly uniform."""
    return rng.integers(low=0, high=N_INDUSTRIES, size=n_assets)


def generate(cfg: ModelConfig) -> SyntheticPanel:
    """Generate a deterministic synthetic panel.

    Parameters
    ----------
    cfg : ModelConfig
        Top-level model configuration.

    Returns
    -------
    SyntheticPanel
        All input panels required by downstream modules.
    """
    rng = np.random.default_rng(cfg.random_seed)
    n_assets = cfg.n_assets
    months = _month_ends(cfg.start_date, cfg.n_months)
    permnos = np.arange(10000, 10000 + n_assets, dtype=np.int64)
    gvkeys = np.arange(20000, 20000 + n_assets, dtype=np.int64)

    industries = _make_industries(rng, n_assets)
    # Per-asset latent params
    beta = np.clip(rng.normal(1.0, 0.35, n_assets), 0.1, 2.5)
    idio_vol = np.abs(rng.normal(0.06, 0.025, n_assets)) + 0.015  # monthly idio
    init_log_cap = rng.normal(13.0, 1.6, n_assets)                # ln(mktcap)
    init_price = np.exp(rng.normal(3.2, 0.7, n_assets))           # ~ $25 typ
    shrout0 = np.exp(init_log_cap) / init_price                   # so price*shrout matches
    leverage = np.clip(rng.beta(2.0, 4.0, n_assets), 0.01, 0.9)

    # Market & risk-free monthly series
    mkt_excess = rng.normal(0.007, 0.045, len(months))
    rf_series = np.clip(rng.normal(0.004, 0.0008, len(months)), 0.0005, 0.012)
    mkt_total = mkt_excess + rf_series
    fx_ret = rng.normal(0.0, 0.018, len(months))

    # ---------- monthly CRSP-like panel ----------
    monthly_rows = []
    prices = init_price.copy()
    shrout = shrout0.copy()
    for t, dt in enumerate(months):
        # CAPM-style returns with idio noise + small size/value tilts
        idio = rng.normal(0.0, idio_vol)
        ret = beta * mkt_excess[t] + idio + rf_series[t]
        prices = prices * (1.0 + ret)
        prices = np.maximum(prices, 0.5)                # floor
        vol = np.maximum(rng.lognormal(mean=11.0, sigma=0.8, size=n_assets), 1e3)
        monthly_rows.append(pd.DataFrame({
            "permno": permnos,
            "date": dt,
            "ret": ret,
            "prc": prices,
            "shrout": shrout,
            "vol": vol,
        }))
    monthly = pd.concat(monthly_rows, ignore_index=True)

    # ---------- daily-like panel (approx 21 trading days/month) ----------
    daily_rows = []
    for t, dt in enumerate(months[-12:]):  # last 12 months are sufficient for daily descriptors
        n_days = 21
        for d in range(n_days):
            day = dt - pd.Timedelta(days=(n_days - d) * 1)
            d_ret = rng.normal(0.0, idio_vol / np.sqrt(21))
            d_ret += beta * mkt_excess[len(months) - 12 + t] / n_days
            d_prc = prices * (1.0 + d_ret * 0.1)
            high = d_prc * (1.0 + np.abs(rng.normal(0.0, 0.01, n_assets)))
            low = d_prc * (1.0 - np.abs(rng.normal(0.0, 0.01, n_assets)))
            d_vol = np.maximum(rng.lognormal(mean=10.0, sigma=0.6, size=n_assets), 1e3)
            daily_rows.append(pd.DataFrame({
                "permno": permnos,
                "date": day,
                "ret": d_ret,
                "prc": d_prc,
                "vol": d_vol,
                "high": high,
                "low": low,
            }))
    daily = pd.concat(daily_rows, ignore_index=True)

    # ---------- Compustat annual fundamentals ----------
    n_years = max(1, cfg.n_months // 12)
    fund_a_rows = []
    base_assets = np.exp(init_log_cap) * (0.8 + rng.uniform(size=n_assets))
    base_ceq = base_assets * (1 - leverage)
    base_ib = base_assets * rng.uniform(0.02, 0.10, n_assets)
    for y in range(n_years + 1):
        date = months[0] + pd.DateOffset(years=y - 1)
        growth = (1 + rng.normal(0.05, 0.10, n_assets))
        at_v = base_assets * (growth ** y)
        ceq_v = base_ceq * (growth ** y)
        ib_v = base_ib * (growth ** y)
        fund_a_rows.append(pd.DataFrame({
            "gvkey": gvkeys,
            "datadate": date,
            "ceq": ceq_v,
            "dltt": at_v * leverage * 0.6,
            "dlc": at_v * leverage * 0.4,
            "at": at_v,
            "dvc": np.maximum(ib_v * 0.3, 0.0),
            "ib": ib_v,
            "dp": at_v * 0.04,
            "txdi": at_v * 0.005,
            "xi": rng.normal(0, 0.001, n_assets) * at_v,
            "nopi": rng.normal(0, 0.002, n_assets) * at_v,
            "epspi": ib_v / np.maximum(shrout0, 1.0),
            "csho": shrout0,
            "pe": at_v * 0.05,        # preferred equity proxy
        }))
    fundamentals_a = pd.concat(fund_a_rows, ignore_index=True)

    # ---------- Compustat quarterly ----------
    fund_q_rows = []
    for t, dt in enumerate(months):
        if t % 3 == 0:
            fund_q_rows.append(pd.DataFrame({
                "gvkey": gvkeys,
                "datadate": dt,
                "epsq": base_ib / np.maximum(shrout0, 1.0) / 4 * (1 + rng.normal(0, 0.05, n_assets)),
                "cshoq": shrout0,
            }))
    fundamentals_q = pd.concat(fund_q_rows, ignore_index=True)

    # ---------- Compustat segments (multi-industry) ----------
    seg_rows = []
    for y in range(n_years + 1):
        date = months[0] + pd.DateOffset(years=y - 1)
        for gv, ind, atv in zip(gvkeys, industries, base_assets * (1.05 ** y)):
            n_seg = rng.choice([1, 1, 1, 2, 3], p=[0.3, 0.2, 0.1, 0.25, 0.15])
            if n_seg == 1:
                seg_rows.append({"gvkey": gv, "datadate": date,
                                 "industry": int(ind), "sales": atv * 0.8,
                                 "seg_at": atv, "oiadp": atv * 0.08})
            else:
                share = rng.dirichlet(np.ones(n_seg))
                inds = rng.choice(N_INDUSTRIES, size=n_seg, replace=False)
                inds[0] = ind  # ensure primary always present
                for s in range(n_seg):
                    seg_rows.append({
                        "gvkey": gv, "datadate": date, "industry": int(inds[s]),
                        "sales": atv * 0.8 * share[s],
                        "seg_at": atv * share[s],
                        "oiadp": atv * 0.08 * share[s],
                    })
    segments = pd.DataFrame(seg_rows)

    # ---------- IBES consensus ----------
    ibes_rows = []
    for t, dt in enumerate(months):
        ibes_rows.append(pd.DataFrame({
            "permno": permnos,
            "statpers": dt,
            "medest_fy1": base_ib / np.maximum(shrout0, 1.0) * (1 + rng.normal(0.03, 0.05, n_assets)),
            "medest_fy2": base_ib / np.maximum(shrout0, 1.0) * (1 + rng.normal(0.08, 0.05, n_assets)),
            "stdev_fy1": np.abs(rng.normal(0.05, 0.02, n_assets)),
            "stdev_fy2": np.abs(rng.normal(0.07, 0.03, n_assets)),
        }))
    ibes = pd.concat(ibes_rows, ignore_index=True)

    sp500 = pd.DataFrame({"date": months, "ret": mkt_total})
    rf = pd.DataFrame({"date": months, "rf": rf_series})
    fx = pd.DataFrame({"date": months, "fx_ret": fx_ret})

    # Bond ratings (1..21)
    ratings = pd.DataFrame({
        "permno": permnos,
        "rating_int": rng.integers(1, 22, n_assets),
    })

    link = pd.DataFrame({"permno": permnos, "gvkey": gvkeys})
    industry_map = pd.DataFrame({"permno": permnos, "industry": industries})

    return SyntheticPanel(
        monthly=monthly, daily=daily,
        fundamentals_a=fundamentals_a, fundamentals_q=fundamentals_q,
        segments=segments, ibes=ibes,
        sp500=sp500, rf=rf, fx=fx,
        ratings=ratings, link=link, industry_map=industry_map,
    )
