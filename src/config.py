"""Configuration constants for the US-E3 replication."""
from __future__ import annotations

import os
from dataclasses import dataclass

# 13 risk indices (US-E3 spec)
RISK_INDICES: list[str] = [
    "VOLT",    # Volatility
    "MOMT",    # Momentum
    "SIZE",    # Size (LNCAP)
    "LCAPCB",  # Size Nonlinearity (cubic of normalized LNCAP)
    "TRAD",    # Trading Activity
    "GRWT",    # Growth
    "EYLD",    # Earnings Yield
    "VALU",    # Value
    "EVAR",    # Earnings Variability
    "LEVE",    # Leverage
    "CURS",    # Currency Sensitivity
    "DYLD",    # Dividend Yield
    "NONESTU", # Non-estimation-universe indicator
]

N_INDUSTRIES: int = 52
INDUSTRY_COLS: list[str] = [f"IND_{i:02d}" for i in range(N_INDUSTRIES)]

# All exposure-matrix columns (industries first, then risk indices)
EXPOSURE_COLS: list[str] = INDUSTRY_COLS + RISK_INDICES
N_FACTORS: int = N_INDUSTRIES + len(RISK_INDICES)  # 52 + 13 = 65

# Universe rules
TOP_N_BY_CAP: int = 1500
MIN_PER_INDUSTRY: int = 20
GRANDFATHER_MAX_MONTHS: int = 18
MIN_PRICE: float = 5.00

# Estimation windows
HALPHA_WINDOW: int = 60          # months
DASTD_WINDOW: int = 65           # days
CMRA_WINDOW: int = 12            # months
RSTR_WINDOW: int = 12            # months
STO5_WINDOW: int = 60            # months

# Covariance / GARCH
HALFLIFE_MONTHS: int = 90

# Return screening (per Step 3 of exp_1)
RET_SCREEN_LOWER: float = -0.50
RET_SCREEN_UPPER: float = 1.50

# GLS weight truncation
GLS_TRUNCATION_PCT: float = 99.0

# Output paths
RESULTS_DIR: str = "results"
DATA_DIR: str = "data"

# Massive API token (optional)
MASSIVE_API_KEY: str | None = os.getenv("MASSIVE_API_KEY") or os.getenv("MASSIVE_TOKEN")


@dataclass(frozen=True)
class ModelConfig:
    """Top-level model configuration."""
    n_assets: int = 250
    n_months: int = 96            # 8 years – tractable for synthetic runs
    start_date: str = "1990-01-31"
    halflife_months: int = HALFLIFE_MONTHS
    universe_size: int = TOP_N_BY_CAP
    use_extended_garch: bool = True
    use_gls: bool = True
    use_two_component_specific_risk: bool = True
    exclusive_descriptors: bool = True
    random_seed: int = 20240601
