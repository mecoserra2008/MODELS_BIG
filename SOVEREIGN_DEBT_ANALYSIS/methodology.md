# Debt-Issuance Probability Model — Methodology

## 1. The question and the reframing

We answer: *for a given counterparty (issuer), what is the probability it issues a
bond of a given type within a given time window?* — next week, within 1–2 months,
in a specific future week-interval, and conditional on characteristics (callable,
coupon in [4%,5%], fixed/floating, tenor, currency, sinkable), or with any
characteristic left unspecified.

The seven CSVs in this folder are a **cross-sectional snapshot** of ~29,538
currently-outstanding bonds (30k rows each, joined on `Símbolo`), **not** an
issuance time series. The unlock: every bond carries a fully-populated issue date
(`Data de emissão`) and an issuer, so we **reconstruct the historical
primary-issuance calendar** and build an `issuer × week` panel. Issuance is dense
in the recent window (2024: 4,074; 2025: 5,394), where survivorship bias is
smallest.

## 2. Data assembly (`load_merge.py`)

- De-duplicate each file on `Símbolo` (multi-exchange listings), join files (1)–(6)
  onto the no-suffix master (the only file with `ISIN`). One clean row per security.
- Normalize Portuguese categories to English, ratings to a 1–22 notch scale
  (handling the Unicode-minus quirk), parse amounts/coupons, derive `tenor_days`,
  `callable`/`puttable`/`sinkable`/`convertible`, `fx_flag`, and issuer rating.

Result: **29,541 securities, 5,006 issuers, 95 countries.** Despite the "sovereign"
filename the universe is *all issuers* (per the chosen scope) — mostly US
corporates/agencies — so this is a general primary-issuance model.

## 3. Panel construction (`panel.py`)

- **As-of = 2026-07-28**; **estimation window = 2019-01-01 → as-of** (earlier events
  only warm up the duration / frequency features).
- Reconstruct issuance events, then build the `issuer × week` at-risk grid for every
  issuer with ≥1 in-window issuance: **949,761 issuer-weeks, 4,248 issuers, 22,978
  in-window issuances** (weekly base rate 1.11%).
- Per-cell features: `weeks_since_last_issue` (duration / baseline hazard, capped
  260w), `issues_trailing_52w` (state dependence), and calendar seasonality
  (`month`, `quarter_end`). Issuer statics: type, sector, country, rating notch.

Empirical structure (validates the design): weekly hazard rises from 0.5% for
dormant issuers to 70% for those issuing 52+×/yr, and falls monotonically with
weeks-since-last — strong duration and state dependence.

## 4. Timing hazard (`timing_model.py`, `features.py`)

Discrete-time survival model:

    Y_{i,w} ~ cloglog(η),   η = β·[ spline(weeks_since_last) + log_trail + log_wsl
                                    + month + quarter_end
                                    + issuer_type + sector + country + rating ]

- **Complementary log-log link** because it is the grouped-interval form of a
  continuous **exponential / Cox proportional hazard** — the weekly hazard then
  aggregates exactly to any horizon via the survival product (this is the
  "exponential embedded in the model").
- **Rare-event fitting:** keep every issuance week, inverse-probability-weight a
  control sub-sample (IPW MLE — consistent for all coefficients, memory-frugal).
- **Regularization:** ridge only the high-cardinality dummies (they cause
  quasi-separation); intercept and core hazard terms unpenalized so the base rate
  stays calibrated.
- Between-issuer heterogeneity (FHLB 1,447 bonds vs a median of 2) is carried by
  `log_trail` + `weeks_since_last` plus country/sector/rating effects — **not** by
  5,000 issuer dummies.

**Forward prediction:** roll `weeks_since_last` forward, decay `trailing_52w` as past
events age out, read seasonality off the calendar → weekly hazards `h_j`, and

    S(k) = ∏_{j≤k}(1−h_j),   P(within k) = 1−S(k),
    P(first issuance in [a,b]) = S(a−1)−S(b),   P(≥1 in [a,b]) = 1−S(b)/S(a−1).

## 5. Conditional characteristics (`char_models.py`)

Given an issuance, the *type* of bond is dominated by issuer behaviour (agencies
issue callables; a treasury issues fixed non-callable). Each characteristic uses
**empirical-Bayes shrinkage**: issuer rate → country|sector group rate → global
rate. Binary (callable, fx, sinkable…), categorical (coupon type), and interval
queries (coupon rate, tenor) are all supported; coupon-value queries use the
issuer's fixed-coupon distribution so *any* interval — e.g. [4%,5%] — is answerable.
Examples: FHLB 90% callable / 93% fixed / 39% coupon∈[4,5]%; Argentina 16% callable
/ 63% short-tenor / 26% FX.

## 6. Query engine (`predict.py`)

    P(issue a bond with characteristics C in horizon H)
        ≈ P(issue at all in H)  ×  ∏_{c∈C} P(c | an issuance)

Unspecified characteristics drop out (factor 1). Returns the probability plus the
timing × characteristic decomposition and the forward hazard/survival curve.

## 7. Out-of-sample validation (`diagnostics.py`)

Temporal split — train on weeks before **2025-01-01**, test after (fit on history,
predict the future, exactly as used):

| metric | value |
|---|---|
| test issuer-weeks | 311,532 |
| ROC-AUC (weekly hazard) | **0.815** |
| Brier score | **0.0095** |
| test base rate | 1.14% |

Calibration is on the 45° line for both the weekly hazard and the horizon quantity
the user asks for — **P(issue within 4 weeks)** — across the full range (see
`results/fig_calibration.png`). Duration/state dependence, seasonality, and example
issuer survival curves are in the other `results/fig_*.png`.

## 8. Caveats (properties of the data, not bugs)

1. **Survivorship** — the snapshot omits matured/redeemed bonds; the reconstruction
   is reliable in the post-2019 window but under-counts deep history and
   ultra-short-dated paper even in-window.
2. **Universe** — "sovereign" is a misnomer; the modelled universe is all issuers
   (mostly US corporates/agencies).
3. **Sparsity** — the median issuer has 2 in-window bonds; "next week" is only a live
   question for the ~157 frequent issuers. Shrinkage keeps tail predictions honest
   (covariate-driven, near-zero).
4. **Independence approximation** in the timing × characteristic product.

## 9. How to run

```bash
python run_issuance_model.py          # build panel, fit, validate, pickle bundle + figures
streamlit run app/streamlit_app.py    # interactive query tool
```

Programmatic query:

```python
import pickle, config as C
from predict import IssuanceModel
m = pickle.load(open(C.MODEL_BUNDLE, "rb"))
m.probability("Federal Home Loan Bank System", ("within", 8),
              {"callable": True, "coupon_interval": (4, 5)})["probability"]
```
