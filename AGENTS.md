# AGENTS.md — repository memory

## Purpose
BARRA US-E3 multi-factor equity risk model replication, all 7 experiments.

## Key conventions
- `permno` is the primary asset identifier.
- All dates are month-end timestamps.
- Risk-free rate column is `rf`. Market column is `mkt`.
- Exposure matrices are stored as DataFrames indexed by `permno`.
- Factor return series stored as DataFrame indexed by `date`, columns = factor names.
- Industries are encoded `IND_00..IND_51` (52 industries).
- Risk indices encoded: VOLT, MOMT, SIZE, LCAPCB, TRAD, GRWT, EYLD, VALU, EVAR, LEVE, CURS, DYLD, NONESTU.

## Data
- Real data path: Massive API (env `MASSIVE_API_KEY`). Fallback to deterministic synthetic
  CRSP/Compustat-like panel for reproducibility in offline CI.

## Implementation notes
- Custom extended-GARCH (omega + alpha*eps^2 + beta*Var + theta*eps) — Context7
  confirmed no off-the-shelf library exposes this exact variant; implemented via
  scipy.optimize. Symmetric baseline uses `arch`.
- GLS uses inverse-residual-variance weights, truncated at 99th percentile.
- LCAPCB = (normalized LNCAP)^3.
- All experiments are wired through `exp/run_all.py` for one-shot reproducibility.
