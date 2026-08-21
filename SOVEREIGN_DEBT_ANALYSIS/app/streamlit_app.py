"""
Interactive issuance-probability desk tool.

Pick an issuer, a horizon (next K weeks, or a specific week interval), and any
subset of bond characteristics -> get P(this issuer issues such a bond in that
window), decomposed into the timing hazard x the conditional characteristic
factors.  Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from predict import IssuanceModel          # noqa: F401  (needed for unpickling)
from char_models import CharModels         # noqa: F401

st.set_page_config(page_title="Debt Issuance Probability", layout="wide", page_icon="📈")
INK = "#1f4e79"


def _describe_horizon(spec) -> str:
    if spec[0] == "within":
        return f"issue within next {spec[1]} week(s)"
    if spec[0] == "interval":
        return f"first issuance in weeks {spec[1]}–{spec[2]}"
    return f"≥1 issuance in weeks {spec[1]}–{spec[2]}"


@st.cache_resource
def load_model():
    with open(C.MODEL_BUNDLE, "rb") as f:
        return pickle.load(f)


if not os.path.exists(C.MODEL_BUNDLE):
    st.error(f"Model bundle not found at {C.MODEL_BUNDLE}. Run `python run_issuance_model.py` first.")
    st.stop()

model = load_model()
cm = model.chars
issuers = model.issuers()

st.title("Debt-Issuance Probability")
st.caption("Discrete-time cloglog issuance hazard (weekly) x conditional characteristic models. "
           f"As-of {C.AS_OF}. Reconstructed primary-issuance history, estimation window "
           f"{C.WINDOW_START}+.")

# --------------------------------------------------------------- controls
left, right = st.columns([1, 1.15])
with left:
    # default to a frequent, recognisable issuer if present
    default = "Federal Home Loan Bank System"
    idx = issuers.index(default) if default in issuers else 0
    issuer = st.selectbox("Counterparty (issuer)", issuers, index=idx)

    st.markdown("**Horizon**")
    mode = st.radio("Horizon type", ["Within next K weeks", "Specific week interval"],
                    label_visibility="collapsed", horizontal=True)
    if mode == "Within next K weeks":
        K = st.slider("K (weeks ahead)", 1, 104, 8,
                      help="1 = next week, 4-8 ≈ 1-2 months, 52 = one year")
        horizon_spec = ("within", K)
        hmax = K
    else:
        a, b = st.slider("Week interval [a, b] ahead", 1, 104, (5, 10))
        sem = st.radio("Interval meaning", ["First issuance lands in window",
                                            "At least one issuance in window"], horizontal=False)
        horizon_spec = ("interval", a, b) if sem.startswith("First") else ("recurrent", a, b)
        hmax = b

with right:
    st.markdown("**Bond characteristics** — leave on *Any* to marginalise")
    c1, c2 = st.columns(2)
    chars = {}
    with c1:
        v = st.selectbox("Callable", ["Any", "Yes", "No"])
        if v != "Any":
            chars["callable"] = (v == "Yes")
        ct = st.selectbox("Coupon type", ["Any", "fixed", "floating", "zero"])
        if ct != "Any":
            chars["coupon_type"] = ct
        tb = st.selectbox("Tenor", ["Any"] + C.TENOR_BUCKET_LABELS)
        if tb != "Any":
            chars["tenor_bucket"] = tb
    with c2:
        fx = st.selectbox("Currency", ["Any", "Foreign (FX)", "Local"])
        if fx != "Any":
            chars["fx"] = (fx.startswith("Foreign"))
        sk = st.selectbox("Sinkable", ["Any", "Yes", "No"])
        if sk != "Any":
            chars["sinkable"] = (sk == "Yes")
        use_cpn = st.checkbox("Constrain coupon rate")
        if use_cpn:
            lo, hi = st.slider("Coupon rate interval (%)", 0.0, 15.0, (4.0, 5.0), 0.25)
            chars["coupon_interval"] = (lo, hi)

# --------------------------------------------------------------- compute
res = model.probability(issuer, horizon_spec, chars)
fh = res["forward"]

st.divider()
m1, m2, m3 = st.columns([1.2, 1, 1])
m1.metric("Probability of issuance", f"{res['probability']*100:.1f}%",
          help="timing probability x characteristic factors")
m2.metric("Timing only", f"{res['timing_prob']*100:.1f}%")
m3.metric("Characteristic factor", f"{res['char_multiplier']*100:.1f}%"
          if chars else "— (any)")

# decomposition table
rows = [("Timing — " + _describe_horizon(horizon_spec), res["timing_prob"])]
rows += [(lbl, p) for lbl, p in res["char_breakdown"]]
rows += [("**Combined**", res["probability"])]
dec = pd.DataFrame(rows, columns=["component", "probability"])
dec["probability"] = (dec["probability"] * 100).round(2).astype(str) + "%"
st.table(dec)

# --------------------------------------------------------------- charts
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown("**Cumulative issuance probability by horizon**")
    curve = pd.DataFrame({
        "weeks ahead": fh["week_ahead"],
        "timing P(issued by then)": 1.0 - fh["survival"].values,
        "x characteristics": (1.0 - fh["survival"].values) * res["char_multiplier"],
    }).set_index("weeks ahead")
    st.line_chart(curve, height=280)
with cc2:
    st.markdown("**Weekly issuance hazard (forward)**")
    haz = pd.DataFrame({"weeks ahead": fh["week_ahead"],
                        "weekly hazard": fh["hazard"].values}).set_index("weeks ahead")
    st.line_chart(haz, height=280)

# --------------------------------------------------------------- issuer profile
st.divider()
st.markdown(f"### Issuer profile — {issuer}")
p1, p2, p3, p4 = st.columns(4)
n = cm.n_issuer.get(issuer, 0)
p1.metric("In-window issuances", f"{n:,}")
p2.metric("P(callable | issue)", f"{cm.p_binary(issuer, 'callable')*100:.0f}%")
p3.metric("P(fixed coupon)", f"{cm.p_category(issuer, 'coupon_type', 'fixed')*100:.0f}%")
p4.metric("P(foreign ccy)", f"{cm.p_binary(issuer, 'fx_flag')*100:.0f}%")

# historical issuance timeline from stored event weeks
ew = model.event_wk.get(issuer)
if ew is not None and len(ew):
    dates = model.base_start + pd.to_timedelta(np.asarray(ew) * 7, unit="D")
    hist = pd.Series(1, index=pd.DatetimeIndex(dates)).resample("QE").sum()
    st.markdown("**Historical issuance (per quarter)**")
    st.bar_chart(hist.rename("issuances"), height=200, color=INK)

st.caption("Model reconstructs issuance from a snapshot of outstanding bonds, so pre-window history "
           "and very short-dated paper are under-represented (survivorship). Characteristic factors "
           "are empirical-Bayes shrunk toward the issuer's country|sector group.")
