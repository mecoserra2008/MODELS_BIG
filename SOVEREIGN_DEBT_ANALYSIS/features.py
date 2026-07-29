"""
Design matrix for the cloglog hazard model.

Hand-built (no patsy) so the whole model bundle pickles cleanly for the app.
The duration baseline is a linear spline on weeks-since-last (fixed knots);
categoricals are explicit drop-first dummies whose level lists are stored in the
DesignSpec and reused verbatim at prediction time (unseen level -> reference).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Duration baseline knots (weeks since last issuance); weeks_since_last is
# capped at 260 in panel.py so we never extrapolate.
DURATION_KNOTS = [2, 4, 8, 13, 26, 52, 104]
MONTHS = list(range(1, 13))
_NUM = ["const", "wsl", "log_wsl", "log_trail", "rating_filled", "rating_missing", "quarter_end"]
_CATS = ["itype", "sector", "country"]           # source cols: issuer_type, sector, country_c
_SRC = {"itype": "issuer_type", "sector": "sector", "country": "country_c"}


@dataclass
class DesignSpec:
    columns: list                 # exact ordered output columns
    country_levels: list          # countries kept (others -> "OTHER")
    cat_levels: dict              # cat -> ordered level list (first = reference)


def _prep(df: pd.DataFrame, country_levels=None):
    out = df.copy()
    if country_levels is None:
        vc = out["country"].value_counts()
        country_levels = vc[vc >= 200].index.tolist()
    out["country_c"] = np.where(out["country"].isin(country_levels), out["country"], "OTHER")
    return out, country_levels


def _numeric_block(frame: pd.DataFrame) -> dict:
    wsl = frame["weeks_since_last"].to_numpy(dtype=float)
    cols = {
        "const": np.ones(len(frame)),
        "wsl": wsl,
        "log_wsl": np.log1p(wsl),
        "log_trail": frame["log_trail"].to_numpy(dtype=float) if "log_trail" in frame
                     else np.log1p(frame["issues_trailing_52w"].to_numpy(dtype=float)),
        "rating_filled": frame["rating_filled"].to_numpy(dtype=float),
        "rating_missing": frame["rating_missing"].to_numpy(dtype=float),
        "quarter_end": frame["quarter_end"].to_numpy(dtype=float),
    }
    for k in DURATION_KNOTS:
        cols[f"wsl_k{k}"] = np.clip(wsl - k, 0.0, None)
    return cols


def _assemble(frame: pd.DataFrame, cat_levels: dict) -> pd.DataFrame:
    cols = _numeric_block(frame)
    # month dummies (reference = January)
    mon = frame["month"].to_numpy()
    for m in MONTHS[1:]:
        cols[f"m_{m}"] = (mon == m).astype(float)
    # categorical drop-first dummies
    for cat in _CATS:
        vals = frame[_SRC[cat]].astype(object).to_numpy()
        for lev in cat_levels[cat][1:]:
            cols[f"{cat}__{lev}"] = (vals == lev).astype(float)
    return pd.DataFrame(cols, index=frame.index)


def build_design(df: pd.DataFrame):
    """Fit-time: return (y, X_df, DesignSpec)."""
    frame, country_levels = _prep(df)
    cat_levels = {}
    for cat in _CATS:
        cat_levels[cat] = sorted(frame[_SRC[cat]].dropna().astype(str).unique().tolist())
    X = _assemble(frame, cat_levels)
    spec = DesignSpec(columns=list(X.columns), country_levels=country_levels, cat_levels=cat_levels)
    y = df["Y"].to_numpy(dtype=float)
    return y, X, spec


def design_from_spec(df: pd.DataFrame, spec: DesignSpec) -> pd.DataFrame:
    """Predict-time: rebuild the identical design matrix for new rows."""
    frame, _ = _prep(df, country_levels=spec.country_levels)
    X = _assemble(frame, spec.cat_levels)
    return X.reindex(columns=spec.columns, fill_value=0.0)


def penalized_mask(columns: list) -> np.ndarray:
    """True for high-cardinality dummies that should be ridge-shrunk."""
    pref = ("m_", "itype__", "sector__", "country__")
    return np.array([c.startswith(pref) for c in columns])


if __name__ == "__main__":
    from panel import build_panel
    grid, _, _ = build_panel()
    y, X, spec = build_design(grid)
    print("design:", X.shape, " penalized dummies:", int(penalized_mask(spec.columns).sum()))
    print("event rate:", round(y.mean(), 4), " kept countries:", len(spec.country_levels))
    print("sample cols:", spec.columns[:6], "...", [c for c in spec.columns if c.startswith("wsl_k")][:3])
