# Fixed Income Positioning Score

A tool that measures **positioning in the US Treasury market** and distills it into a
single, point-in-time **crowding score** plus a calibrated forward-return probability.

## What the score answers

Two layered outputs (see the design decision in the plan):

1. **Descriptive crowding index** — *where* is the market positioned, in normalized DV01
   terms. A signed z-index (`+` = crowded net long duration, `−` = crowded short).
2. **Predictive overlay** — *what it implies*: a calibrated `P(bond selloff over next 4w)`,
   trained walk-forward. Extremes fade (contrarian), surfaced as a `{+1/0/−1}` stance.

## Inputs (which data feeds the score)

| Source | What | Access |
|---|---|---|
| **CFTC TFF** (`gpe5-46if`) | Leveraged Funds, Asset Managers, Dealer net positions per contract (TU/FV/TY/TN/US/UB), open interest, top-4 concentration | free, keyless |
| **NY Fed primary dealer** (FR2004) | cash Treasury net positions by tenor bucket (the spot leg) | free, keyless |
| **FRED** | CMT yields (DGS2/5/10/30) for the DV01 conversion | free, API key |

CFTC futures are the primary input (deepest history, back to 2006); cash is secondary;
options skew / swaps are Bloomberg-only (Track B).

## Methodology (hierarchical weighting)

```
net contracts ──DV01──▶ per-tenor DV01 ──Σ within category──▶ category duration
       │                                                              │
       │                                              causal z-score / percentile
       ▼                                                              ▼
  curve, concentration, cash features ───────────────▶  PCA  ──▶ PC1 = composite index
                                                          └──▶ PC2 = curve positioning
                                                                     │
                                            calibrated logistic (walk-forward) ──▶ P(selloff)
```

- **DV01-equivalent** first — contracts across tenors aren't additive (`core/dv01.py`).
- **PCA factor weighting** removes the collinearity that equal-weighting double-counts;
  an equal-weight average is kept as a transparent fallback (`core/aggregate.py`).
- **Leakage guards**: PCA/scaler/model fit on train only; the predictive label reads yields
  on the *publication* timeline (report + 3 bday lag), so features never see the future.

## Two tracks, one core

- **Track A — free/public** (test now): `python -m positioning.run_positioning [--force]`
  → `results/positioning_score.json` + `results/fig_*.png`.
- **Desk app** (Streamlit, BIG-branded): from the repo root run
  `streamlit run positioning/app/streamlit_app.py` — a crowding **thermometer**, a per-asset
  **Bull/Bear board** (positioning + contrarian signal), **extra positioning indicators**
  (retail, dealer repo, ETF demand, rate-vol), a **News & Sentiment** panel (FinBERT-scored
  headlines, 1D/1W/1M), and the full methodology **step by step**. Dark default + orange,
  light/dark toggle. Sidebar controls recompute reactively.

## Deploy (Streamlit Community Cloud)

- **Main file path:** `positioning/app/streamlit_app.py` (repo root is the working dir).
- **Requirements:** the repo-root `requirements.txt` (includes `transformers` + `torch` for
  FinBERT). If the tier can't fit torch, simply remove `transformers`/`torch` — the news panel
  **auto-falls back** to the built-in finance-lexicon scorer, and everything else is unaffected.
- **Secret:** set `FRED_API_KEY` in the app's **Secrets** (Settings → Secrets). The yields
  adapter reads env → `.env` → `st.secrets` in that order.
- **Network:** the app pulls live from CFTC, NY Fed, FRED, GDELT, RSS feeds and the Yahoo chart
  API — all keyless except FRED. First run also downloads the ~440MB FinBERT weights (cached).
- **News & Sentiment framing:** FinBERT gives generic financial tone; a rate-direction layer
  re-orients the sign to the **bond** convention (hawkish/“yields surge” → bearish; cuts/
  safe-haven bid → bullish). Positive = bullish fixed income.
- **Track B — Bloomberg/BQuant**: `bquant/positioning_bbg.ipynb` swaps in a `bql` data
  adapter and calls the **same `positioning.core`**. Falls back to the Track A cache when
  run outside BQNT, so it executes end-to-end today. Upgrade path: exact CTD DV01
  (`RISK_MID`), plus options skew and swap-dealer positioning.

Adapters are the only source-specific code; everything in `core/` is shared.

## Layout

```
positioning/
  config.py            registry: contracts, CFTC codes, categories, params
  core/                source-agnostic engine (schema, dv01, normalize,
                       aggregate+PCA, predictive, signal, score)
  adapters/            cftc_free, nyfed_free, yields_free
  run_positioning.py   Track A driver
  bquant/              Track B notebook
  data/raw/, results/  CSV cache + JSON/figure outputs
```
