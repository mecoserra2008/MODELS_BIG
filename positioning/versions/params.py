"""Editable-parameter contract shared by the version builds and the Streamlit app.

`VERSION_PARAMS[version]` is the ordered schema of tunable knobs for each methodology; the
app renders one widget per `Param`, the version `build(inp, params)` reads them. Defaults equal
the current hardcoded behaviour, so `build(inp)` / `build(inp, None)` reproduce today's numbers.

Two groups: 'Data inputs' (which categories / tenors / signal transform) and 'Method params'
(the normalization knobs). `resolve_signal` / `resolve_front_long` apply the data-input choices
so every version derives its underlying series the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from positioning.config import SCORING_CATEGORIES
from . import base

CATS = list(SCORING_CATEGORIES)                 # ('lev_money','asset_mgr')
ALL_CATS = ["lev_money", "asset_mgr", "dealer"]
INSTRUMENTS = ["TU", "FV", "TY", "TN", "US", "UB"]
GROUP_DATA = "Data inputs"
GROUP_METHOD = "Method params"


@dataclass
class Param:
    key: str
    label: str
    kind: str                          # 'int'|'float'|'choice'|'multiselect'|'bool'|'text'
    default: object
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list | None = None
    help: str = ""
    group: str = GROUP_METHOD


# --------------------------------------------------------------------------- #
# Per-version schemas (defaults == current hardcoded values)
# --------------------------------------------------------------------------- #
_DATA_CATS = Param("categories", "Trader categories", "multiselect", CATS, choices=ALL_CATS,
                   help="Which TFF categories form the buy-side signal.", group=GROUP_DATA)
_DATA_SIGNAL = Param("signal_mode", "Signal transform", "choice", "level",
                     choices=["level", "pct_oi"],
                     help="Raw net-DV01 level, or net as %-of-open-interest (market-size invariant).",
                     group=GROUP_DATA)
# Shared upstream normalization knob (transforms.apply_norm) — used by the richer-deviations versions.
_NORM = Param("norm_mode", "Upstream normalization", "choice", "raw",
              choices=["raw", "fracdiff", "volscale", "robust"], group=GROUP_METHOD,
              help="Transform applied to the source BEFORE the signal is built: raw; fracdiff "
                   "(min-d fractional differencing, stationary+memory); volscale (constant-risk); "
                   "robust (median/MAD z).")

VERSION_PARAMS: dict[str, list[Param]] = {
    "v0": [
        Param("train_end", "PCA train-end", "text", "2019-12-31", group=GROUP_METHOD,
              help="PCA/scaler fit window end (production leakage guard)."),
        Param("k_entry", "Crowd threshold (z)", "float", 1.5, 0.5, 3.0, 0.1, group=GROUP_METHOD),
    ],
    "v1": [
        _DATA_CATS,
        Param("include_curve", "Include curve feature", "bool", True, group=GROUP_DATA),
        Param("train_end", "Train-end (fit scaler+PCA)", "text", "2019-12-31", group=GROUP_METHOD),
        Param("kaiser_cutoff", "Kaiser eigenvalue cutoff", "float", 1.0, 0.5, 2.0, 0.1,
              group=GROUP_METHOD, help="Retain PCs with eigenvalue > cutoff."),
        Param("standardize_scope", "Standardize scope", "choice", "train",
              choices=["train", "full"], group=GROUP_METHOD),
    ],
    "v2": [
        _DATA_CATS, _DATA_SIGNAL,
        Param("cot_short", "COT-index short window (w)", "int", 52, 13, 208, 1, group=GROUP_METHOD),
        Param("cot_long", "COT-index long window (w)", "int", 156, 52, 312, 1, group=GROUP_METHOD),
        Param("z_window", "z window (w)", "int", 52, 13, 208, 1, group=GROUP_METHOD),
        Param("flow_window", "Flow change window (w)", "int", 13, 2, 52, 1, group=GROUP_METHOD),
        Param("crowd_thr", "Crowd threshold (0-100)", "float", 80.0, 50.0, 95.0, 1.0, group=GROUP_METHOD),
        Param("w_z", "street_z weight: z_52", "float", 0.5, 0.0, 1.0, 0.05, group=GROUP_METHOD),
        Param("w_oi", "street_z weight: %OI z", "float", 0.3, 0.0, 1.0, 0.05, group=GROUP_METHOD),
        Param("w_flow", "street_z weight: flow z", "float", 0.2, 0.0, 1.0, 0.05, group=GROUP_METHOD),
    ],
    "v3": [
        _DATA_CATS, _DATA_SIGNAL,
        Param("ewma_hl", "EWMA half-life (w)", "int", 26, 4, 104, 1, group=GROUP_METHOD),
        Param("z_window", "52w z window (w)", "int", 52, 13, 208, 1, group=GROUP_METHOD),
        Param("flow_window", "Flow change window (w)", "int", 13, 2, 52, 1, group=GROUP_METHOD),
        Param("w_ewma", "Blend weight: EWMA z", "float", 0.7, 0.0, 1.0, 0.05, group=GROUP_METHOD),
        Param("w_flow", "Blend weight: flow z", "float", 0.3, 0.0, 1.0, 0.05, group=GROUP_METHOD),
    ],
    "v4": [
        _DATA_CATS,
        Param("front_set", "Front-end contracts", "multiselect", ["TU", "FV"], choices=INSTRUMENTS,
              group=GROUP_DATA),
        Param("long_set", "Long-end contracts", "multiselect", ["US", "UB"], choices=INSTRUMENTS,
              group=GROUP_DATA),
        Param("ewma_hl", "Momentum EWMA half-life (w)", "int", 26, 4, 104, 1, group=GROUP_METHOD),
        Param("z_window", "z window (w)", "int", 52, 13, 208, 1, group=GROUP_METHOD),
        Param("cot_window", "COT-index window (w)", "int", 52, 13, 208, 1, group=GROUP_METHOD),
        Param("train_end", "Train-end (PC2 fit)", "text", "2019-12-31", group=GROUP_METHOD),
    ],
    "v5": [
        Param("source", "Signal to filter", "choice", "v0",
              choices=["v0", "v1", "v2", "v3", "v4", "signal"], group=GROUP_DATA,
              help="Which series the Kalman filter runs on: a version's composite, or the raw "
                   "buy-side net-DV01 signal."),
        _DATA_CATS, _DATA_SIGNAL,
        Param("output", "Headline output", "choice", "deviation",
              choices=["deviation", "slope", "acceleration"], group=GROUP_METHOD,
              help="deviation = (signal - Kalman level)/σ; slope = filter trend; "
                   "acceleration = change in trend."),
        Param("kalman_smooth", "Kalman smoothness", "choice", "auto",
              choices=["auto", "responsive", "smooth"], group=GROUP_METHOD,
              help="auto = MLE-fit variances; responsive/smooth = fixed-variance presets "
                   "(faster/slower trend)."),
        Param("dev_window", "Deviation σ window (w)", "int", 52, 13, 156, 1, group=GROUP_METHOD),
    ],
    "v6": [
        _DATA_CATS, _DATA_SIGNAL, _NORM,
        Param("cell_level", "Roof each cell then aggregate", "bool", False, group=GROUP_DATA,
              help="False = roof the aggregate buy-side net-DV01; True = roof each category×tenor "
                   "cell then DV01-weight aggregate the oscillators."),
        Param("output", "Headline output", "choice", "oscillator",
              choices=["oscillator", "phase", "fisher"], group=GROUP_METHOD,
              help="oscillator = roofing band-pass; phase = Hilbert Sine/LeadSine crossover + "
                   "amplitude; fisher = Fisher-sharpened extremes."),
        Param("hp_cutoff", "High-pass cutoff (w)", "int", 48, 20, 104, 1, group=GROUP_METHOD,
              help="Kills cycles longer than this (trend removal)."),
        Param("ss_period", "SuperSmoother period (w)", "int", 10, 4, 26, 1, group=GROUP_METHOD),
        Param("phase_period", "Hilbert nominal cycle (w)", "int", 20, 8, 52, 1, group=GROUP_METHOD),
        Param("fisher_window", "Fisher min-max window (w)", "int", 52, 13, 156, 1, group=GROUP_METHOD),
        Param("cf_crosscheck", "Christiano-Fitzgerald cross-check", "bool", True, group=GROUP_METHOD,
              help="Also compute the CF band-pass (random-walk-optimal, full-sample) for meta."),
    ],
    "v7": [
        _DATA_CATS, _DATA_SIGNAL, _NORM,
        Param("panel", "Cells for breadth/dispersion", "choice", "buyside",
              choices=["buyside", "all"], group=GROUP_DATA,
              help="buyside = lev+AM cells; all = incl. dealer."),
        Param("components", "Signals to combine", "multiselect",
              ["level", "flow", "accel", "breadth", "dispersion"],
              choices=["level", "flow", "accel", "breadth", "dispersion"], group=GROUP_METHOD),
        Param("combine", "Combine method", "choice", "equal",
              choices=["equal", "rank", "detoned_pca", "ica"], group=GROUP_METHOD,
              help="equal/rank = robust average; detoned_pca = strip PC1 then PC2/PC3; "
                   "ica = FastICA components (whitened)."),
        Param("orthogonalize", "Gram-Schmidt orthogonalize", "bool", True, group=GROUP_METHOD,
              help="Residualize flow⊥level, accel⊥flow so each is an independent bet (raises BR_eff)."),
        Param("z_window", "z window (w)", "int", 52, 13, 208, 1, group=GROUP_METHOD),
        Param("flow_window", "Flow Δ window (w)", "int", 4, 1, 26, 1, group=GROUP_METHOD),
        Param("accel_window", "Accel Δ² window (w)", "int", 4, 1, 26, 1, group=GROUP_METHOD),
        Param("breadth_thr", "Breadth |z| threshold", "float", 1.0, 0.5, 2.5, 0.1, group=GROUP_METHOD),
    ],
    "v8": [
        Param("source", "Signal to monitor", "choice", "signal",
              choices=["signal", "v0", "v3", "v6"], group=GROUP_DATA,
              help="Series the change-point/surprise layer runs on (raw buy-side signal or a version)."),
        _DATA_CATS, _DATA_SIGNAL, _NORM,
        Param("output", "Headline output", "choice", "blend",
              choices=["blend", "changepoint_prob", "nis_surprise", "regime_prob"],
              group=GROUP_METHOD,
              help="blend = rank-average of the three event signals; or a single one."),
        Param("bocpd_hazard", "BOCPD hazard (per week)", "float", 0.01, 0.002, 0.1, 0.002,
              group=GROUP_METHOD, help="Prior break rate ≈ 1/mean-weeks-between-breaks."),
        Param("bocpd_obs", "BOCPD observation model", "choice", "t",
              choices=["t", "gaussian"], group=GROUP_METHOD,
              help="t (dof=2, default) is robust to a single fat-tailed CFTC week and, with the "
                   "run-length-mass readout P(r_t≤4), fires cleanly on genuine breaks (COVID-2020 ≈0.94); "
                   "gaussian (dof≈20) is a slightly less peaky alternative."),
        Param("hmm_states", "HMM regimes", "int", 2, 2, 3, 1, group=GROUP_METHOD),
        Param("nis_window", "NIS σ window (w)", "int", 52, 13, 156, 1, group=GROUP_METHOD),
    ],
    "v9": [
        Param("components", "Component signals", "multiselect", ["v6", "v7", "v8"],
              choices=["v6", "v7", "v8"], group=GROUP_DATA,
              help="Candidate signals eligible for the final blend."),
        Param("gate", "Acceptance gate", "choice", "strict",
              choices=["strict", "report_only"], group=GROUP_METHOD,
              help="strict = only components whose incremental deviations pass the OOS rubric enter "
                   "the blend; report_only = blend all and show the scorecard."),
        Param("blend", "Blend method", "choice", "rank",
              choices=["rank", "robust_z", "equal"], group=GROUP_METHOD),
        Param("horizon", "IC/gate horizon (w)", "int", 8, 4, 13, 1, group=GROUP_METHOD),
    ],
    "v10": [
        Param("panel", "KPI panel", "choice", "feature_panel",
              choices=["feature_panel", "raw_level"], group=GROUP_DATA,
              help="feature_panel = the 7 rolling-3y-z KPIs V0 feeds to PCA (direct PCA->Kalman swap); "
                   "raw_level = raw per-category net-DV01 levels (longer sample, near-integrated)."),
        Param("k_factors", "Latent factors", "int", 1, 1, 2, 1, group=GROUP_METHOD,
              help="Common dynamic factors the KPIs load on (the 'informed average')."),
        Param("factor_order", "Factor AR order", "int", 1, 1, 2, 1, group=GROUP_METHOD),
        Param("fit_mode", "Fit mode", "choice", "auto", choices=["auto", "fixed"],
              group=GROUP_METHOD,
              help="auto = MLE (full-sample fit when the train slice is too small on the free cache) "
                   "with a fixed-variance fallback on non-convergence; fixed = robust fixed variances."),
        Param("train_end", "MLE train-end", "text", "2019-12-31", group=GROUP_METHOD,
              help="Leakage-guard train slice; auto-falls back to full-sample fit if <30 rows."),
        Param("output", "Headline output", "choice", "deviation",
              choices=["deviation", "factor", "slope", "acceleration"], group=GROUP_METHOD,
              help="deviation = standardized innovation (deviation from the informed average, NIS); "
                   "factor = filtered crowding state (dynamic PC1); slope/acceleration = factor "
                   "trend / change-in-trend."),
        Param("break_detector", "Break detector", "choice", "both",
              choices=["both", "cusum", "bocpd"], group=GROUP_METHOD,
              help="cusum = CUSUM of recursive residuals (Brown-Durbin-Evans); bocpd = reuse V8 "
                   "run-length break prob on the innovations; both = combine."),
        Param("break_response", "Break response", "choice", "both",
              choices=["both", "resample", "retrain"], group=GROUP_METHOD,
              help="resample = bootstrap the new-regime distribution for robust quantile bands "
                   "(headline); retrain = refit factor / adaptive-Q so the filter snaps; both."),
        Param("regime_window", "Regime window (w)", "int", 104, 26, 260, 1, group=GROUP_METHOD,
              help="Trailing window defining the current regime for resampled bands and retraining."),
        Param("bocpd_hazard", "Break hazard (per week)", "float", 0.01, 0.002, 0.1, 0.002,
              group=GROUP_METHOD),
    ],
}


# --------------------------------------------------------------------------- #
def defaults(version: str) -> dict:
    return {p.key: p.default for p in VERSION_PARAMS[version]}


def merge(version: str, overrides: dict | None) -> dict:
    d = defaults(version)
    if overrides:
        d.update({k: v for k, v in overrides.items() if k in d})
    return d


def resolve_signal(inp: "base.VersionInputs", params: dict) -> pd.Series:
    """Buy-side net-DV01 signal ($mm/bp) for the selected categories, optionally %-of-OI."""
    cats = params.get("categories", CATS) or CATS
    cols = [c for c in cats if c in inp.cat_dv01.columns]
    level = inp.cat_dv01[cols].sum(axis=1).rename("signal")
    if params.get("signal_mode", "level") == "pct_oi":
        return base.pct_of_oi(level, inp.oi).rename("signal")
    return level


def resolve_front_long(inp: "base.VersionInputs", params: dict) -> pd.Series:
    """front(front_set) - long(long_set) net DV01 ($mm/bp), summed over selected categories,
    from the per-(category,instrument) panel so the tenor sets are editable."""
    cats = params.get("categories", CATS) or CATS
    front = params.get("front_set", ["TU", "FV"])
    long = params.get("long_set", ["US", "UB"])
    cd = inp.contract_dv01
    def pick(instrs):
        keep = [c for c in cd.columns if c[0] in cats and c[1] in instrs]
        return cd[keep].sum(axis=1) if keep else pd.Series(0.0, index=cd.index)
    return (pick(front) - pick(long)).rename("front_minus_long")
