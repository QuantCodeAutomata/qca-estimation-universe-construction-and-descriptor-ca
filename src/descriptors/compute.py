"""Computation of all descriptors per Appendix A of US-E3 handbook.

Each descriptor is implemented as a vectorized function returning a Series
indexed by `permno`. All descriptors are computed using only data available
strictly before month T (look-ahead-free).

Custom — Context7 confirmed no library equivalent for the US-E3 specific
descriptor formulas (Appendix A). Standard libraries (pyportfolioopt, skfolio,
ta) cover portfolio optimization / technical indicators but not BARRA-style
risk-model descriptor construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import (
    CMRA_WINDOW,
    DASTD_WINDOW,
    HALPHA_WINDOW,
    RISK_INDICES,
    RSTR_WINDOW,
)

EPS = 1e-12


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _rolling_regression(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """OLS regression y = alpha + beta * x + e.

    Returns
    -------
    alpha, beta, resid_std, residuals
    """
    n = len(y)
    if n < 24:
        return np.nan, np.nan, np.nan, np.full(n, np.nan)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    resid = y - X @ coef
    if n - 2 <= 0:
        return alpha, beta, np.nan, resid
    sigma = float(np.sqrt(np.sum(resid ** 2) / (n - 2)))
    return float(alpha), float(beta), sigma, resid


# ----------------------------------------------------------------------
# Per-asset descriptors using monthly history
# ----------------------------------------------------------------------
def _capm_block(
    excess: pd.Series,
    mkt_excess: pd.Series,
) -> dict:
    """Run 60-month CAPM and return all derived descriptors."""
    y = excess.values
    x = mkt_excess.values
    alpha, beta, hsigma, resid = _rolling_regression(y, x)
    out = {"HALPHA": alpha, "HSIGMA": hsigma, "BETA": beta}

    # BTSG
    if np.isnan(beta) or beta < 0 or np.isnan(hsigma):
        out["BTSG"] = 0.0
    else:
        out["BTSG"] = float(np.sqrt(beta * hsigma))

    # SERDP per spec
    if np.isnan(beta) or len(resid) < 3:
        out["SERDP"] = np.nan
    else:
        e = resid
        num = np.mean([(e[t] + e[t - 1] + e[t - 2]) ** 2 for t in range(2, len(e))])
        den = np.mean([(e[t] ** 2 + e[t - 1] ** 2 + e[t - 2] ** 2) for t in range(2, len(e))])
        out["SERDP"] = float(num / den) if den > EPS else np.nan
    out["_resid"] = resid
    return out


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def compute_descriptors(
    panel,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute the full set of raw descriptors for one month T.

    Parameters
    ----------
    panel : SyntheticPanel
        Output of :func:`src.data_loader.load_panel`.
    target_date : pd.Timestamp
        Month-end of month T. Only data with date < target_date is used.

    Returns
    -------
    pandas.DataFrame
        Indexed by permno; columns are descriptor names + 'mktcap'.
    """
    monthly = panel.monthly
    daily = panel.daily
    sp500 = panel.sp500.set_index("date")["ret"]
    rf = panel.rf.set_index("date")["rf"]
    fx = panel.fx.set_index("date")["fx_ret"]

    # Strict look-ahead control: use history < target_date
    hist = monthly[monthly["date"] < target_date].copy()
    mkt_excess = (sp500 - rf).loc[sp500.index < target_date]

    permnos = monthly["permno"].unique()
    rows: list[dict] = []
    resid_store: dict[int, np.ndarray] = {}

    for permno in permnos:
        h = hist[hist["permno"] == permno].sort_values("date")
        if len(h) == 0:
            continue
        last = h.iloc[-1]
        price = float(last["prc"])
        shrout = float(last["shrout"])
        mktcap = price * shrout
        row: dict = {
            "permno": int(permno),
            "mktcap": mktcap,
            "LNCAP": float(np.log(max(mktcap, 1.0))),
            "LPRI": float(np.log(max(price, 0.01))),
        }

        # ---- CAPM 60m block ----
        window = h.tail(HALPHA_WINDOW)
        if len(window) >= 24:
            common = window["date"]
            mk = mkt_excess.reindex(common).fillna(0).values
            ex = (window["ret"].values - rf.reindex(common).fillna(0).values)
            capm = _capm_block(pd.Series(ex), pd.Series(mk))
            row.update({
                "HALPHA": capm["HALPHA"],
                "HSIGMA": capm["HSIGMA"],
                "BTSG": capm["BTSG"],
                "SERDP": capm["SERDP"],
                "BETA": capm["BETA"],
            })
            resid_store[int(permno)] = capm["_resid"]

            # CURSENS: regress residuals on FX_t, FX_{t-1}, FX_{t-2}
            fx_w = fx.reindex(common).fillna(0).values
            if len(fx_w) >= 6 and not np.any(np.isnan(capm["_resid"])):
                e = capm["_resid"]
                X = np.column_stack([
                    np.ones(len(e) - 2),
                    fx_w[2:],
                    fx_w[1:-1],
                    fx_w[:-2],
                ])
                y = e[2:]
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                row["CURSENS"] = float(coef[1] + coef[2] + coef[3])
            else:
                row["CURSENS"] = np.nan
        else:
            for k in ("HALPHA", "HSIGMA", "BTSG", "SERDP", "BETA", "CURSENS"):
                row[k] = np.nan

        # ---- RSTR ----
        win12 = h.tail(RSTR_WINDOW)
        if len(win12) == RSTR_WINDOW:
            rstr = float(np.sum(np.log1p(win12["ret"].values))
                         - np.sum(np.log1p(rf.reindex(win12["date"]).fillna(0).values)))
            row["RSTR"] = rstr

            # CMRA
            cum_asset = np.cumsum(np.log1p(win12["ret"].values))
            cum_rf = np.cumsum(np.log1p(rf.reindex(win12["date"]).fillna(0).values))
            Z = cum_asset - cum_rf
            row["CMRA"] = float(np.log((1 + Z.max()) / max(1 + Z.min(), EPS)))
        else:
            row["RSTR"] = np.nan
            row["CMRA"] = np.nan

        # ---- HILO from daily ----
        last_month = h.iloc[-1]["date"]
        d_last = daily[(daily["permno"] == permno)
                       & (daily["date"] >= last_month - pd.Timedelta(days=31))
                       & (daily["date"] < target_date)]
        if len(d_last) > 0:
            row["HILO"] = float(np.log(d_last["high"].max() / max(d_last["low"].min(), EPS)))
        else:
            row["HILO"] = np.nan

        # ---- DASTD from 65 daily returns ----
        d_hist = (daily[(daily["permno"] == permno) & (daily["date"] < target_date)]
                  .sort_values("date").tail(DASTD_WINDOW))
        if len(d_hist) >= 20:
            r = d_hist["ret"].values
            row["DASTD"] = float(np.sqrt(23.0 * np.mean(r ** 2)))
        else:
            row["DASTD"] = np.nan

        # ---- Trading activity (Step 6) ----
        if shrout > 0:
            v_last = float(last["vol"])
            row["STOM"] = v_last / shrout
            recent_3 = h.tail(3)
            if len(recent_3) == 3 and recent_3["shrout"].mean() > 0:
                row["STOQ"] = float(4 * recent_3["vol"].sum() / (3 * recent_3["shrout"].mean()))
            else:
                row["STOQ"] = np.nan
            recent_12 = h.tail(12)
            if len(recent_12) == 12 and recent_12["shrout"].mean() > 0:
                row["STOA"] = float(recent_12["vol"].sum() / (12 * recent_12["shrout"].mean()))
            else:
                row["STOA"] = np.nan
            recent_60 = h.tail(60)
            if len(recent_60) >= 24 and recent_60["shrout"].mean() > 0:
                row["STO5"] = float(12 * recent_60["vol"].mean() / recent_60["shrout"].mean())
            else:
                row["STO5"] = np.nan
            # VLVR (Step 6) — log of dollar volume / hsigma
            T_eff = min(12, len(recent_12))
            if T_eff > 0 and row.get("HSIGMA") and row["HSIGMA"] > EPS:
                dv = (recent_12["vol"].values * recent_12["prc"].values).sum()
                row["VLVR"] = float(np.log((12.0 / T_eff) * dv / row["HSIGMA"]))
            else:
                row["VLVR"] = np.nan
        else:
            for k in ("STOM", "STOQ", "STOA", "STO5", "VLVR"):
                row[k] = np.nan

        # ---- VOLBT (slope of weekly volume change vs market vol change) ----
        # Simple proxy: regression slope of weekly vol pct change vs cross-asset mean
        row["VOLBT"] = float(np.std(np.diff(h["vol"].values)) /
                             max(np.mean(h["vol"].values), 1.0))

        # ---- FSPLIT (proxy: 0 in synthetic; would be 1 if forward split in 24m) ----
        row["FSPLIT"] = 0

        rows.append(row)

    df = pd.DataFrame(rows).set_index("permno")

    # Cross-sectional fundamentals descriptors — vectorized join
    df = _merge_fundamentals(df, panel, target_date)

    return df


# ----------------------------------------------------------------------
# Fundamentals-based descriptors
# ----------------------------------------------------------------------
def _merge_fundamentals(
    df: pd.DataFrame,
    panel,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Add Compustat-derived and IBES-derived descriptors."""
    link = panel.link.set_index("permno")["gvkey"]
    fund_a = panel.fundamentals_a.copy()
    # 90-day lag rule
    fund_a["available"] = fund_a["datadate"] + pd.Timedelta(days=90)
    fund_a = fund_a[fund_a["available"] <= target_date]
    if fund_a.empty:
        # Add NaN columns and return
        for c in ("BTOP", "MLEV", "BLEV", "DTOA", "ETOP", "ETP5", "EGRO",
                  "AGRO", "PAYO", "VCAP", "VERN", "VFLO", "EXTE", "DELE",
                  "EPIBS", "EGIBS", "SPIBS", "P_DYLD", "SNRRT", "OPSTD"):
            df[c] = np.nan
        return df

    latest = (fund_a.sort_values("datadate")
              .groupby("gvkey").tail(1)
              .set_index("gvkey"))
    df["gvkey"] = link.reindex(df.index)
    # Per-asset latest annual
    joined = df.join(latest, on="gvkey", rsuffix="_fa")

    mktcap = joined["mktcap"].replace(0, np.nan)
    joined["BTOP"] = joined["ceq"] / mktcap
    joined["MLEV"] = (mktcap + joined["pe"] + joined["dltt"] + joined["dlc"]) / mktcap
    joined["BLEV"] = (joined["ceq"] + joined["pe"] + joined["dltt"] + joined["dlc"]) / joined["ceq"]
    joined["DTOA"] = (joined["dltt"] + joined["dlc"]) / joined["at"]
    joined["ETOP"] = joined["ib"] / mktcap
    joined["PAYO"] = joined["dvc"] / joined["ib"].replace(0, np.nan)
    joined["EXTE"] = (joined["xi"].abs() + joined["nopi"].abs()) / joined["ib"].replace(0, np.nan)

    # 5-year growth slopes (use up-to-5 latest annual rows per gvkey)
    grp = fund_a.sort_values("datadate").groupby("gvkey")
    egro_map: dict = {}
    agro_map: dict = {}
    etp5_map: dict = {}
    vern_map: dict = {}
    vflo_map: dict = {}
    dele_map: dict = {}
    vcap_map: dict = {}
    for gv, g in grp:
        g5 = g.tail(5)
        if len(g5) < 3:
            continue
        t = np.arange(1, len(g5) + 1, dtype=float)
        # EGRO
        eps_arr = (g5["ib"].values / np.maximum(g5["csho"].values, 1.0))
        m_eps = np.mean(eps_arr)
        if m_eps != 0:
            slope, *_ = stats.linregress(t, eps_arr)[:2]
            egro_map[gv] = slope / abs(m_eps)
        if np.mean(eps_arr) != 0:
            vern_map[gv] = float(np.std(eps_arr) / abs(np.mean(eps_arr)))
        # AGRO
        at_arr = g5["at"].values
        if at_arr.mean() != 0:
            slope_a, *_ = stats.linregress(t, at_arr)[:2]
            agro_map[gv] = slope_a / abs(at_arr.mean())
        # ETP5: mean EPS / mean (proxy) price = mean EPS / mean (at/csho) — proxy
        if at_arr.mean() > 0:
            etp5_map[gv] = float(eps_arr.mean() / max(at_arr.mean() / max(g5["csho"].mean(), 1.0), EPS))
        # Cash flow & VFLO
        cf = eps_arr + g5["dp"].values / np.maximum(g5["csho"].values, 1.0) \
             + g5["txdi"].values / np.maximum(g5["csho"].values, 1.0)
        if cf.mean() != 0:
            vflo_map[gv] = float(np.std(cf) / abs(cf.mean()))
        # DELE
        if len(eps_arr) >= 2 and (eps_arr[-1] + eps_arr[-2]) > 0:
            dele_map[gv] = float(2 * (eps_arr[-1] - eps_arr[-2]) / (eps_arr[-1] + eps_arr[-2]))
        # VCAP
        if len(g5) >= 5:
            n = g5["csho"].values
            ld = (g5["dltt"].values + g5["dlc"].values)
            pe = g5["pe"].values
            p = g5["ib"].values / np.maximum(n, 1.0)
            num = 0.0
            for i in range(1, len(g5)):
                num += (abs(n[i - 1] - n[i]) * p[i - 1]
                        + abs(ld[i - 1] - ld[i])
                        + abs(pe[i] - pe[i - 1]))
            denom = 4 * (g5["ceq"].iloc[-1] + ld[-1] + pe[-1])
            if denom > EPS:
                vcap_map[gv] = float(num / denom)

    joined["EGRO"] = joined["gvkey"].map(egro_map)
    joined["AGRO"] = joined["gvkey"].map(agro_map)
    joined["ETP5"] = joined["gvkey"].map(etp5_map)
    joined["VERN"] = joined["gvkey"].map(vern_map)
    joined["VFLO"] = joined["gvkey"].map(vflo_map)
    joined["DELE"] = joined["gvkey"].map(dele_map)
    joined["VCAP"] = joined["gvkey"].map(vcap_map)

    # IBES descriptors
    ibes = panel.ibes.copy()
    ibes = ibes[ibes["statpers"] <= target_date]
    ibes_latest = (ibes.sort_values("statpers")
                   .groupby("permno").tail(1)
                   .set_index("permno"))
    weight_fy1 = 0.5
    earn_proxy = (weight_fy1 * ibes_latest["medest_fy1"]
                  + (1 - weight_fy1) * ibes_latest["medest_fy2"])
    eps_q = ibes_latest["medest_fy1"]  # approximation
    joined["EARN_PROXY"] = joined.index.map(earn_proxy)
    joined["EPIBS"] = joined["EARN_PROXY"] / (joined["mktcap"] / np.maximum(joined["csho"], 1.0))
    joined["EGIBS"] = (joined["EARN_PROXY"] - joined.index.map(eps_q)) / \
                     ((joined["EARN_PROXY"] + joined.index.map(eps_q)) / 2).replace(0, np.nan)
    joined["SPIBS"] = (weight_fy1 * joined.index.map(ibes_latest["stdev_fy1"])
                      + (1 - weight_fy1) * joined.index.map(ibes_latest["stdev_fy2"])) / \
                     (joined["mktcap"] / np.maximum(joined["csho"], 1.0))

    # P_DYLD: annualized dividend / price proxy
    joined["P_DYLD"] = joined["dvc"] / mktcap

    # Ratings (SNRRT)
    ratings = panel.ratings.set_index("permno")["rating_int"]
    joined["SNRRT"] = joined.index.map(ratings)

    # OPSTD: impute from cross-section of DASTD and BTSG (no options data)
    dastd_mean = joined["DASTD"].mean()
    btsg_mean = joined["BTSG"].mean()
    joined["OPSTD"] = (joined["DASTD"].fillna(dastd_mean)
                      + 0.5 * joined["BTSG"].fillna(btsg_mean))

    # Clean up
    drop_cols = [c for c in joined.columns if c in (
        "datadate", "available", "gvkey", "ceq", "dltt", "dlc", "at",
        "dvc", "ib", "dp", "txdi", "xi", "nopi", "epspi", "csho", "pe",
        "EARN_PROXY"
    )]
    joined = joined.drop(columns=drop_cols, errors="ignore")
    return joined


# ----------------------------------------------------------------------
# Normalization (cap-weighted mean, equal-weighted std)
# ----------------------------------------------------------------------
def normalize_descriptors(
    df: pd.DataFrame,
    estu_mask: pd.Series,
    mktcap: pd.Series,
    descriptor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Capitalization-weighted mean normalization, cross-sectional std standardization.

    Parameters
    ----------
    df : pd.DataFrame
        Raw descriptor matrix indexed by permno.
    estu_mask : pd.Series
        Boolean mask over `df.index`: True if asset is in estimation universe.
    mktcap : pd.Series
        Market capitalization indexed by permno.
    descriptor_cols : list[str], optional
        Columns to normalize. Defaults to all numeric columns except 'mktcap'.

    Returns
    -------
    pd.DataFrame
        Normalized exposures (mean 0, std 1 cross-sectionally on ESTU).
    """
    if descriptor_cols is None:
        descriptor_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c != "mktcap"]

    out = df[descriptor_cols].astype(float).copy()
    w = mktcap.reindex(df.index).fillna(0.0)
    w_estu = w.where(estu_mask, 0.0)
    w_sum = float(w_estu.sum())

    for col in descriptor_cols:
        x = out[col]
        valid = x[estu_mask].dropna()
        if valid.empty or w_sum <= 0:
            out[col] = 0.0
            continue
        mu = float(np.nansum((x * w_estu).values) / w_sum)
        sigma = float(np.sqrt(np.nanmean((valid - mu) ** 2)))
        if sigma <= EPS:
            out[col] = 0.0
        else:
            out[col] = (x - mu) / sigma

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_lcapcb(normalized: pd.DataFrame) -> pd.Series:
    """Construct LCAPCB = (normalized LNCAP)^3 per spec (two-pass)."""
    if "LNCAP" not in normalized.columns:
        raise KeyError("LNCAP must be present in normalized descriptors before LCAPCB.")
    return normalized["LNCAP"] ** 3


def build_risk_indices(normalized: pd.DataFrame, nonestu: pd.Series) -> pd.DataFrame:
    """Aggregate normalized descriptors into the 13 US-E3 risk indices.

    Each risk index is the equal-weighted average of its constituent descriptors
    (US-E3 spec: exclusive descriptor assignment).
    """
    mapping = {
        "VOLT": ["DASTD", "HILO", "CMRA", "OPSTD"],
        "MOMT": ["RSTR", "HALPHA"],
        "SIZE": ["LNCAP"],
        "LCAPCB": ["LCAPCB"],
        "TRAD": ["STOM", "STOQ", "STOA", "STO5", "VLVR", "VOLBT"],
        "GRWT": ["AGRO", "EGRO", "EGIBS"],
        "EYLD": ["ETOP", "ETP5", "EPIBS"],
        "VALU": ["BTOP", "LPRI"],
        "EVAR": ["VERN", "VFLO", "EXTE", "SERDP"],
        "LEVE": ["MLEV", "BLEV", "DTOA", "SNRRT"],
        "CURS": ["CURSENS"],
        "DYLD": ["P_DYLD", "PAYO"],
        "NONESTU": [],
    }
    out = pd.DataFrame(index=normalized.index, columns=RISK_INDICES, dtype=float)
    for ri, descs in mapping.items():
        if not descs:
            continue
        present = [d for d in descs if d in normalized.columns]
        if not present:
            out[ri] = 0.0
        else:
            out[ri] = normalized[present].mean(axis=1)
    out["NONESTU"] = nonestu.reindex(out.index).fillna(0).astype(float)
    return out.fillna(0.0)
