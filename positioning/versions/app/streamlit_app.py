"""Positioning Methodologies — one interactive tool, one tab per composite version.

Run from the repo root:
    streamlit run positioning/versions/app/streamlit_app.py

Presentation-only layer over the frozen `positioning.versions` contract. Each version tab
renders its editable knobs from `params.VERSION_PARAMS` (via `controls.render_params`),
rebuilds the composite through that version's `build(inp, params)`, and shows: the composite
with crowd bands, a responsiveness read vs the production baseline, and an honest conditional
predictive panel (publication-lagged signal vs weekly DGS10). A final Compare tab overlays all
five at their default settings. Nothing here mutates `positioning.core` or the version builds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from positioning.app import theme  # noqa: E402
from positioning.config import DEFAULT_START  # noqa: E402
from positioning.core.predictive import conditional_extreme_stats  # noqa: E402
from positioning.versions import base, params as P  # noqa: E402
from positioning.versions import (  # noqa: E402
    v0_production, v1_kaiser_pca, v2_cftc_bestpractice, v3_ewma_52w, v4_curve,
)
from positioning.versions.app import controls, viz  # noqa: E402

# predict.py + v5_kalman.py may still be filled in by the backend agent — import them
# defensively so a missing/partial module can never stop the app booting.
try:
    from positioning.versions import predict  # noqa: E402
except Exception:                             # pragma: no cover - stub not yet present
    predict = None

try:
    from positioning.versions import v5_kalman  # noqa: E402
    _V5_BUILD = getattr(v5_kalman, "build", None)
    _V5_OK = callable(_V5_BUILD)
except Exception:                             # pragma: no cover - stub not yet present
    _V5_BUILD, _V5_OK = None, False

st.set_page_config(page_title="Positioning Methodologies", page_icon="🟠", layout="wide",
                   initial_sidebar_state="expanded")

# Registry: version key -> (short display label, build fn).
VERSIONS = {
    "v0": ("Production", v0_production.build),
    "v1": ("Kaiser-PCA", v1_kaiser_pca.build),
    "v2": ("CFTC/Street", v2_cftc_bestpractice.build),
    "v3": ("EWMA/52w", v3_ewma_52w.build),
    "v4": ("Curve", v4_curve.build),
}
if _V5_OK:                                    # backend agent's Kalman version, once available
    VERSIONS["v5"] = ("Kalman", _V5_BUILD)

BASE_ORDER = ["v0", "v1", "v2", "v3", "v4"]   # the always-present per-methodology tabs
ORDER = BASE_ORDER + (["v5"] if _V5_OK else [])  # full set (Compare / registry iteration)
HORIZONS = (4, 8, 13)


# --------------------------------------------------------------------------- #
# Caching — keep args primitive; recompute rich objects inside
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load(start: str, force: bool) -> base.VersionInputs:
    """Cache the shared inputs bundle on primitive keys."""
    return base.load_inputs(force=force, start=start or None)


@st.cache_data(show_spinner=False)
def build_cached(version: str, params_items: tuple, start: str, force: bool) -> dict:
    """Rebuild a version from a hashable params tuple; return only cache-friendly pieces.

    VersionResult / pd.Series aren't good cache keys, so we cache on primitives and
    recompute inside: reload inputs (cached), rebuild the params dict, call build.
    """
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in params_items}
    inp = load(start, force)
    build = VERSIONS[version][1]
    res = build(inp, params)
    return {
        "name": res.name,
        "description": res.description,
        "composite": res.composite,
        "bounded": bool(res.bounded),
        "meta": res.meta,
    }


def _items(params: dict) -> tuple:
    """Hashable tuple of sorted items (lists -> tuples) for the cache key."""
    return tuple(sorted(
        (k, tuple(v) if isinstance(v, list) else v) for k, v in params.items()))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _regime(latest, bounded: bool, params: dict) -> str:
    """Regime label from the latest composite value (crowd_thr for bounded, +/-1.5 for z)."""
    if latest is None or (isinstance(latest, float) and np.isnan(latest)):
        return "n/a"
    if bounded:
        thr = float(params.get("crowd_thr", 80.0))
        if latest > thr:
            return "Crowded long"
        if latest < 100.0 - thr:
            return "Crowded short"
        return "Neutral"
    if latest > 1.5:
        return "Crowded long"
    if latest < -1.5:
        return "Crowded short"
    return "Neutral"


def _uncond_up(y10: pd.Series, horizons=HORIZONS) -> dict:
    """Unconditional P(yield up over H) per horizon — the honest reference line."""
    out = {}
    for H in horizons:
        f = (y10.shift(-H) - y10).dropna()
        out[str(H)] = float((f > 0).mean()) if len(f) else None
    return out


def _jsonable(obj):
    """Recursively coerce a meta dict into something st.json can render (Series -> summary)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Series):
        d = obj.dropna()
        latest = float(d.iloc[-1]) if len(d) else None
        return f"Series(n={len(obj)}, latest={latest})"
    if isinstance(obj, pd.DataFrame):
        return f"DataFrame(shape={obj.shape})"
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


def _lead_str(resp: dict, prod_ref: dict) -> str | None:
    """Weeks the version led production's first crowd flag (positive = earlier)."""
    w, pw = resp.get("weeks_since_flag"), prod_ref.get("weeks_since_flag")
    if w is None or pw is None:
        return None
    return f"{w - pw:+d}w vs prod"


# --------------------------------------------------------------------------- #
# Conditional predictive panel (pub_lag -> align to y10_w -> conditional_extreme_stats)
# --------------------------------------------------------------------------- #
def _conditional(comp: pd.Series, bounded: bool, inp: base.VersionInputs, mode: str,
                 key: str) -> None:
    # Bounded [0,100] composites are mapped to a z so the extreme test is comparable.
    sig = ((comp - 50.0) / 25.0) if bounded else comp
    sig_pub = base.pub_lag(sig.dropna())                        # no look-ahead
    y10 = inp.y10_w.dropna()
    # as-of align the published signal onto the weekly yield grid (same idiom as base.py)
    sig_on_y = (sig_pub.reindex(y10.index.union(sig_pub.index)).ffill().reindex(y10.index))
    stats = conditional_extreme_stats(sig_on_y, y10, horizons=HORIZONS, k=1.5)
    uncond = _uncond_up(y10, HORIZONS)
    st.plotly_chart(viz.conditional_bar(stats, mode, uncond=uncond,
                                        title="Does this signal predict yields?"),
                    width="stretch", key=f"cond_{key}")
    cap = ("Historical descriptor — CIs typically include 0.5; a faster flag is a better "
           "descriptor, not new alpha.")
    if bounded:
        cap += " Bounded composite mapped to z via (x−50)/25; extremes at |z|>1.5."
    st.caption(cap)


# --------------------------------------------------------------------------- #
# Coefficients & interpretation panel (forward regression + logistic read)
# --------------------------------------------------------------------------- #
def _r(v, nd: int = 2):
    """Round a possibly-None/NaN number for a desk dataframe cell."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, nd)


def _read_reg(comp: pd.Series, y10_w: pd.Series, bounded: bool) -> dict:
    if predict is None:
        return {}
    try:
        return predict.forward_regression(comp, y10_w, bounded=bounded) or {}
    except Exception:
        return {}


def _read_logit(comp: pd.Series, y10_w: pd.Series) -> dict:
    if predict is None:
        return {}
    try:
        return predict.logistic_read(comp, y10_w) or {}
    except Exception:
        return {}


def _coefficients(comp: pd.Series, bounded: bool, inp: base.VersionInputs, mode: str,
                  key: str) -> None:
    """Descriptive coefficient panel: forward-Δ10Y regression + logistic read, side by side,
    with interpretation sentences, an honesty caption and a per-horizon coefficient table."""
    st.markdown("###### Coefficients & interpretation")
    reg = _read_reg(comp, inp.y10_w, bounded)
    logit = _read_logit(comp, inp.y10_w)
    has_reg = bool(reg)
    has_logit = bool(logit and logit.get("coef"))

    if not has_reg and not has_logit:
        st.info("Coefficient reads are computing or unavailable (the predictive backend "
                "returns empty for now) — regression β and logistic coefficients will appear "
                "here once ready.")
        return

    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(viz.coef_table(reg, mode), width="stretch", key=f"coef_{key}")
    with cB:
        st.plotly_chart(viz.logit_table(logit, mode), width="stretch", key=f"logit_{key}")

    # interpretation sentences — per-horizon regression, then top ±2 logistic features
    if has_reg:
        for H in sorted(reg, key=int):
            txt = (reg[H] or {}).get("interpretation")
            if txt:
                st.markdown("- " + str(txt))
    if has_logit:
        coef = logit.get("coef", {}) or {}
        interp = logit.get("interpretation", {}) or {}
        pos = sorted((f for f in coef if _r(coef[f]) is not None and coef[f] > 0),
                     key=lambda f: coef[f], reverse=True)[:2]
        neg = sorted((f for f in coef if _r(coef[f]) is not None and coef[f] < 0),
                     key=lambda f: coef[f])[:2]
        for f in pos + neg:
            if interp.get(f):
                st.markdown("- " + str(interp[f]))

    st.caption("Coefficients are descriptive associations, not causal or tradeable alpha — "
               "positioning's OOS predictive power is near chance (see AUC).")

    if has_reg:
        rows = []
        for H in sorted(reg, key=int):
            d = reg[H] or {}
            rows.append({"H (w)": int(H), "β bp/σ": _r(d.get("beta_bp_per_sigma"), 1),
                         "t": _r(d.get("t")), "p": _r(d.get("p"), 3),
                         "R²": _r(d.get("r2"), 3), "n": d.get("n")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# V5 Kalman extras: latest slope/acceleration + optional trend chart
# --------------------------------------------------------------------------- #
def _latest_val(meta: dict, key: str):
    """Latest scalar for a meta field. V5 exposes latest values as `<key>_latest` scalars;
    accept a plain `<key>` (scalar or pd.Series) too so the read is robust to either shape."""
    meta = meta or {}
    v = meta.get(f"{key}_latest", meta.get(key))
    if isinstance(v, pd.Series):
        d = v.dropna()
        return float(d.iloc[-1]) if len(d) else None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return None if np.isnan(f) else f
    return None


def _v5_extras(meta: dict, mode: str) -> None:
    """Surface the Kalman filter's deviation / slope / acceleration (metrics row + trend chart)."""
    st.markdown("###### Kalman state (level / trend / acceleration)")
    dev = _latest_val(meta, "deviation")
    slope = _latest_val(meta, "slope")
    accel = _latest_val(meta, "acceleration")
    level = _latest_val(meta, "level")

    def _f(x, nd=2):
        return "n/a" if x is None else f"{x:+.{nd}f}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Deviation (σ)", _f(dev))
    m2.metric("Slope (trend)", _f(slope, 3))
    m3.metric("Acceleration", _f(accel, 3))
    m4.metric("Kalman level", "n/a" if level is None else f"{level:.2f}")

    aux = viz.kalman_aux(meta, mode)
    if aux is not None:
        st.plotly_chart(aux, width="stretch", key="v5_aux")
    else:
        st.caption("Slope & acceleration shown as latest values (meta carries scalars, not "
                   "a full trend series).")


# --------------------------------------------------------------------------- #
# One version tab
# --------------------------------------------------------------------------- #
def version_tab(v: str, mode: str, start: str, force: bool, prod_ref: dict) -> None:
    label = VERSIONS[v][0]
    params = controls.render_params(v)                          # drives every widget
    d = build_cached(v, _items(params), start, force)
    comp, bounded = d["composite"], d["bounded"]

    st.markdown(f"##### {d['name']}")
    st.caption(d["description"])
    if v == "v4":
        st.caption("V4 sign convention: POSITIVE = crowded STEEPENER (front-long vs long-short), "
                   "not net-long duration.")

    resp = base.responsiveness(comp, bounded)
    latest = resp["latest"]
    fmt = ("{:.1f}".format(latest) if bounded else "{:+.2f}".format(latest)) \
        if latest is not None else "n/a"

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest composite", fmt)
    c2.metric("Regime", _regime(latest, bounded, params))
    c3.metric("First crowd flag", resp["first_flag"] or "—", delta=_lead_str(resp, prod_ref))

    st.plotly_chart(viz.composite_ts(comp, bounded, mode, title=d["name"]),
                    width="stretch", key=f"comp_{v}")

    # V5 Kalman: surface the filter's slope & acceleration alongside the composite.
    if v == "v5":
        try:
            _v5_extras(d["meta"], mode)
        except Exception as e:      # a slow/odd meta must never blank the rest of the tab
            st.exception(e)

    inp = load(start, force)
    _conditional(comp, bounded, inp, mode, key=v)

    # Shared predictive coefficient panel (guarded so a stub-empty read never blanks the tab).
    try:
        _coefficients(comp, bounded, inp, mode, key=v)
    except Exception as e:
        st.exception(e)

    with st.expander("Method internals (meta)"):
        st.json(_jsonable(d["meta"]))


# --------------------------------------------------------------------------- #
# Compare tab
# --------------------------------------------------------------------------- #
def compare_tab(mode: str, start: str, force: bool, prod_ref: dict) -> None:
    results = {v: build_cached(v, _items(P.defaults(v)), start, force) for v in ORDER}

    st.markdown("##### All methodologies at their default settings")
    st.caption("z-family composites (unbounded, +/-1.5 crowd bands) and the bounded COT-index "
               "shown separately because their axes differ.")

    # Split by each build's own bounded flag so V5 (deviation-σ, z-family) joins the overlay
    # automatically and any bounded version is drawn on its own axis.
    z_versions = [v for v in ORDER if not results[v]["bounded"]]
    b_versions = [v for v in ORDER if results[v]["bounded"]]

    zmap = {f"{v.upper()} · {VERSIONS[v][0]}": results[v]["composite"] for v in z_versions}
    st.plotly_chart(viz.overlay(zmap, bounded=False, mode=mode,
                                title="z-family composites ("
                                      + " / ".join(v.upper() for v in z_versions) + ")"),
                    width="stretch", key="ov_z")

    bmap = {f"{v.upper()} · {VERSIONS[v][0]}": results[v]["composite"] for v in b_versions}
    if bmap:
        st.plotly_chart(viz.overlay(bmap, bounded=True, mode=mode,
                                    title="Bounded COT-index composite ("
                                          + " / ".join(v.upper() for v in b_versions) + ")"),
                        width="stretch", key="ov_b")

    rows = []
    for v in ORDER:
        d = results[v]
        r = base.responsiveness(d["composite"], d["bounded"])
        w, pw = r["weeks_since_flag"], prod_ref.get("weeks_since_flag")
        lead = (w - pw) if (w is not None and pw is not None) else None
        rows.append({
            "version": f"{v.upper()} {VERSIONS[v][0]}",
            "bounded": d["bounded"],
            "latest": None if r["latest"] is None else round(r["latest"], 2),
            "first_flag": r["first_flag"],
            "weeks_since_flag": w,
            "lead_vs_prod_w": lead,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Lead vs prod (weeks) > 0 means the version flagged the current crowding "
               "build-up earlier than the production composite. V4 is a curve/steepener "
               "signal, not duration.")


# --------------------------------------------------------------------------- #
def main() -> None:
    with st.sidebar:
        mode = "dark" if st.toggle("Dark theme", value=True) else "light"
        st.markdown("### Data")
        start = st.text_input("History start", DEFAULT_START)
        force = st.button("↻ Refresh data", width="stretch")
        st.caption("One tab per composite methodology. Each tab's knobs come straight from "
                   "params.VERSION_PARAMS and drive that version's build(inp, params).")

    theme.inject_css(mode)
    if force:
        st.cache_data.clear()

    try:
        with st.spinner("Fetching CFTC / NY Fed / FRED …"):
            inp = load(start, force)
    except Exception as e:
        st.error(f"Could not load inputs: {type(e).__name__}: {e}")
        return

    pc = inp.prod_composite.dropna()
    as_of = str(pc.index.max().date()) if len(pc) else "n/a"
    theme.header(as_of, f"{len(inp.report_dates)} weekly CFTC reports")

    try:
        prod_ref = base.responsiveness(
            build_cached("v0", _items(P.defaults("v0")), start, force)["composite"], False)
    except Exception:
        prod_ref = {"weeks_since_flag": None}

    tabs = st.tabs(["Production (V0)", "Kaiser-PCA (V1)", "CFTC/Street (V2)",
                    "EWMA/52w (V3)", "Curve (V4)", "Kalman (V5)", "Compare"])
    for tab, v in zip(tabs[:5], BASE_ORDER):
        with tab:
            try:
                version_tab(v, mode, start, force, prod_ref)
            except Exception as e:  # one bad tab must never blank the app
                st.exception(e)

    with tabs[5]:                   # Kalman (V5) — same machinery + slope/acceleration
        try:
            if _V5_OK:
                version_tab("v5", mode, start, force, prod_ref)
            else:
                st.info("Kalman (V5) build is being finalized — its controls, composite and "
                        "slope/acceleration read will appear here once v5_kalman.build lands.")
        except Exception as e:
            st.exception(e)

    with tabs[6]:
        try:
            compare_tab(mode, start, force, prod_ref)
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
else:
    # Streamlit executes the script top-to-bottom; run under the runtime too. Guarded so a
    # bare `import` (no ScriptRunContext) can never raise.
    try:
        main()
    except Exception:
        pass
