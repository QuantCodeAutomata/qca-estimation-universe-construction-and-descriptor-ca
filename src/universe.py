"""Monthly estimation universe construction (Steps 4 & 5 of exp_1).

Implements the US-E3 rules-based selection:

1. S&P 500 anchor (we don't have constituent history → top-cap proxy).
2. Top-1500 by market capitalization.
3. Industry fill-in to ensure each of 52 industries has >= 20 assets.
4. Grandfathering up to 18 months for assets that drop below cutoff.
5. NONESTU flag for non-ESTU assets with fundamentals coverage.

Custom — Context7 confirmed no portfolio/risk library implements BARRA US-E3
universe rules; implemented from spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    GRANDFATHER_MAX_MONTHS,
    MIN_PER_INDUSTRY,
    MIN_PRICE,
    TOP_N_BY_CAP,
)


@dataclass
class UniverseState:
    """Mutable state carried month-to-month for grandfathering."""
    months_below_cutoff: dict[int, int]


def init_state() -> UniverseState:
    return UniverseState(months_below_cutoff={})


def construct_universe(
    panel,
    target_date: pd.Timestamp,
    state: UniverseState,
    universe_size: int = TOP_N_BY_CAP,
) -> pd.DataFrame:
    """Construct the monthly estimation universe for `target_date`.

    Parameters
    ----------
    panel : SyntheticPanel
        Input panel.
    target_date : pd.Timestamp
        Month-end of month T.
    state : UniverseState
        Carries grandfathering counters forward through time.
    universe_size : int
        Top-N by market cap (default 1500).

    Returns
    -------
    pd.DataFrame
        permno-indexed frame with columns: mktcap, industry, in_estu,
        nonestu, grandfathered, decile.
    """
    monthly = panel.monthly
    hist = monthly[monthly["date"] < target_date]
    if hist.empty:
        return pd.DataFrame(columns=["mktcap", "industry", "in_estu",
                                     "nonestu", "grandfathered", "decile"])

    latest_idx = hist.groupby("permno")["date"].idxmax()
    latest = hist.loc[latest_idx].set_index("permno")
    latest["mktcap"] = latest["prc"] * latest["shrout"]
    latest = latest[latest["prc"] >= MIN_PRICE]

    industry = panel.industry_map.set_index("permno")["industry"]
    latest = latest.join(industry, how="inner")

    # Bottom decile of market cap = excluded from fill-in
    cap_p10 = latest["mktcap"].quantile(0.10)

    # Step 4b — top-N by cap
    ranked = latest.sort_values("mktcap", ascending=False)
    top_n = ranked.head(universe_size).index

    # Step 4c — industry fill-in
    in_estu = pd.Series(False, index=latest.index)
    in_estu.loc[top_n] = True

    eligible_fill = latest[(latest["prc"] >= MIN_PRICE) & (latest["mktcap"] > cap_p10)]
    for ind, grp in eligible_fill.groupby("industry"):
        current = in_estu[grp.index].sum()
        if current >= MIN_PER_INDUSTRY:
            continue
        need = MIN_PER_INDUSTRY - current
        candidates = grp[~in_estu.reindex(grp.index, fill_value=False)] \
            .sort_values("mktcap", ascending=False).head(need)
        in_estu.loc[candidates.index] = True

    # Step 4d — grandfathering
    cap_cutoff = ranked["mktcap"].iloc[min(universe_size - 1, len(ranked) - 1)]
    grandfathered = pd.Series(False, index=latest.index)
    for permno in list(state.months_below_cutoff.keys()):
        if permno not in latest.index:
            del state.months_below_cutoff[permno]
            continue
        cap_i = latest.loc[permno, "mktcap"]
        prc_i = latest.loc[permno, "prc"]
        if cap_i >= cap_cutoff:
            state.months_below_cutoff[permno] = 0
        elif (prc_i >= MIN_PRICE) and (cap_i > cap_p10):
            state.months_below_cutoff[permno] += 1
            if state.months_below_cutoff[permno] <= GRANDFATHER_MAX_MONTHS:
                in_estu.loc[permno] = True
                grandfathered.loc[permno] = True
            else:
                in_estu.loc[permno] = False
                del state.months_below_cutoff[permno]

    # New ESTU members below cutoff → initialize counter at 0
    below = latest.index[(latest["mktcap"] < cap_cutoff) & in_estu]
    for permno in below:
        state.months_below_cutoff.setdefault(int(permno), 0)

    # Step 4e — NONESTU flag
    nonestu = (~in_estu).astype(int)

    # Cap deciles
    deciles = pd.qcut(latest["mktcap"], 10, labels=False, duplicates="drop") + 1

    out = latest[["mktcap"]].copy()
    out["industry"] = latest["industry"]
    out["in_estu"] = in_estu.astype(bool)
    out["nonestu"] = nonestu.astype(int)
    out["grandfathered"] = grandfathered.astype(bool)
    out["decile"] = deciles
    return out
