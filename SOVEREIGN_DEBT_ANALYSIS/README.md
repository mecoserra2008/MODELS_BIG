# Debt-Issuance Probability Model

Panel-data model that estimates **the probability a given counterparty issues a
bond — of a given type, within a given time window** — from the 7-CSV bond
snapshot in this folder. Built as a discrete-time **cloglog issuance hazard**
(weekly) × **conditional characteristic models**. See `methodology.md` for the full
write-up.

## Quickstart

```bash
python run_issuance_model.py          # build panel → fit → validate → results/model_bundle.pkl + figures
streamlit run app/streamlit_app.py    # interactive desk tool
```

Programmatic:

```python
import pickle, config as C
from predict import IssuanceModel
m = pickle.load(open(C.MODEL_BUNDLE, "rb"))

m.probability("Government of Argentina", ("within", 8))["probability"]          # within 8 weeks
m.probability("Federal Home Loan Bank System", ("within", 8),
              {"callable": True, "coupon_interval": (4, 5)})["probability"]      # callable, 4–5% coupon
m.timing_prob("Deutsche Bank AG", ("recurrent", 5, 10))[0]                      # ≥1 issuance in weeks 5–10
```

Horizon specs: `("within", k)`, `("interval", a, b)` (first issuance lands in window),
`("recurrent", a, b)` (≥1 issuance in window).
Characteristics (any subset; omit to marginalize): `callable`, `puttable`, `sinkable`,
`convertible`, `fx` (bool); `coupon_type` ∈ {fixed, floating, zero};
`coupon_interval=(lo,hi)`; `tenor_bucket` ∈ {`<3y`,`3-7y`,`7-12y`,`12y+`}.

## File map

| file | role |
|---|---|
| `config.py` | paths, PT→EN maps, rating scale, buckets, window/as-of |
| `load_merge.py` | de-dupe + translate + join the 7 CSVs → security table |
| `panel.py` | reconstruct issuance events → issuer×week at-risk grid + features |
| `features.py` | picklable design matrix (duration spline + dummies) |
| `timing_model.py` | cloglog hazard fit (IPW + ridge) + forward hazard scoring |
| `char_models.py` | empirical-Bayes conditional characteristic models |
| `predict.py` | `IssuanceModel` query engine (timing × characteristics) |
| `diagnostics.py` | temporal OOS validation (AUC/Brier/calibration) + figures |
| `run_issuance_model.py` | end-to-end driver |
| `app/streamlit_app.py` | interactive query UI |

## Validation

Temporal hold-out (train <2025-01-01, test after): **ROC-AUC 0.815, Brier 0.0095**,
calibrated on the 45° line for both the weekly hazard and P(issue within 4 weeks).
Figures in `results/`.

**Caveats:** snapshot-based reconstruction ⇒ survivorship bias (reliable post-2019);
"sovereign" is a misnomer (universe = all issuers, mostly US corporates/agencies);
median issuer has 2 in-window bonds so "next week" is only meaningful for frequent
issuers. Details in `methodology.md`.
