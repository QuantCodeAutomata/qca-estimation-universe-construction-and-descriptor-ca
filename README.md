# QCA — BARRA US-E3 Replication (Estimation Universe & Descriptor Engine)

End-to-end implementation of the BARRA US-E3 multi-factor equity risk model
covering all 7 experiments:

| # | Experiment | Module |
|---|---|---|
| 1 | Universe construction and 40+ descriptors | `src/universe.py`, `src/descriptors/*` |
| 2 | Multi-industry weights via 3-model valuation | `src/industry_weights.py` |
| 3 | GLS cross-sectional factor return regression | `src/factor_returns.py` |
| 4 | Extended asymmetric GARCH + factor covariance | `src/garch_cov.py` |
| 5 | Two-component specific risk model | `src/specific_risk.py` |
| 6 | LCAPCB / Size-Nonlinearity validation (H2) | `src/validation.py` |
| 7 | Full US-E3 vs US-E2 validation | `exp/run_all.py` |

## Data

The reference implementation requires CRSP / Compustat / IBES via WRDS, which is
not available in this environment. The repository therefore contains:

* `src/data_loader.py` — thin abstraction that **prefers the Massive API**
  (`MASSIVE_API_KEY` env var) for downloading real price data,
* `src/synthetic_data.py` — a deterministic synthetic CRSP/Compustat-like
  panel used when no API token is available, so the full pipeline runs end-to-end.

All algorithmic logic (universe rules, descriptor formulas, GLS, GARCH, two-component
specific-risk, bias analysis) is identical regardless of the data source.

## Usage

```bash
pip install -r requirements.txt
pytest -q                # run unit + integration tests
python exp/run_all.py    # end-to-end: writes results/RESULTS.md + figures
```

## Structure

```
data/        cached panels (parquet)
src/         models, descriptors, regressions, covariance, validation
exp/         experiment driver scripts
tests/       pytest suite
results/     metrics tables + figures + RESULTS.md
```
