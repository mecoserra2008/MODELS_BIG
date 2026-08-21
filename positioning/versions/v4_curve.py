"""V4 Curve — curve positioning re-done as two explicit dimensions.

Fixes the book-vs-momentum ambiguity found in the earlier curve read: the *raw* buy-side
front-minus-long DV01 book is roughly flat right now (~ -2.6 $mm/bp), yet a 3y rolling-z of
that same book reads "steepener". Those two statements are not in conflict once we separate:

  1. BOOK TILT  — the raw LEVEL of the curve book ($mm/bp). Currently ~flat.
  2. MOMENTUM   — a fast, causal EWMA-z of the book (half-life 26w). This is what reads
                  steepener-*direction*: the book has been drifting toward front-long faster
                  than its own recent history, even though the level itself is still ~flat.

SIGN CONVENTION (this version):
  POSITIVE = crowded STEEPENER (front-end TU+FV long vs long-end US+UB short).

  IMPORTANT — `composite` here does NOT carry base.py's default meaning (duration crowding,
  POSITIVE = crowded net long duration). For THIS version, `composite` encodes CURVE steepener
  crowding, and `curve` is set to the SAME series. Consumers comparing V4 against duration
  versions must read `composite` as a curve/steepener signal, not a duration signal.

Primary signal = MOMENTUM (`mom = ewma_z(book, 26)`), unbounded, so `bounded=False`.

Supporting diagnostics (in meta):
  * z52          — 52w rolling-z of the book (slower normalization, for context).
  * pca_pc2_curve — projection of the per-category front-long features onto the PCA slope
                    factor (the component with opposite loadings across categories). Scaler +
                    PCA are fit ONLY on the training slice (config.TRAIN_END) and projected
                    forward, so there is no look-ahead.
  * cot_front_long — classic 52w COT-index (0..100) of the raw book, saturating stochastic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from positioning.config import SCORING_CATEGORIES, TRAIN_END
from positioning.versions import params as P
from positioning.versions.base import (
    VersionInputs,
    VersionResult,
    cot_index,
    ewma_z,
    load_inputs,
    responsiveness,
    rolling_z,
)


def _last(s: pd.Series) -> float:
    """Latest finite value of a series, or NaN if none."""
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _pca_slope_factor(front_long: pd.DataFrame,
                      train_end: str = TRAIN_END) -> tuple[pd.Series, dict]:
    """Small PCA on the per-category front-long features to isolate the curve/slope factor.

    Standardize ONCE with a StandardScaler fit on the training slice (<= train_end) and apply
    to the full history; fit PCA on the (standardized) training slice; project the full history.

    The "slope" factor is the component whose loadings have OPPOSITE signs across the two
    categories (a category-differential = curve tilt) rather than a common level move. We prefer
    such a component that also passes the Kaiser criterion (eigenvalue > 1); otherwise we take
    the opposite-loading component regardless of eigenvalue; otherwise we fall back to PC2.

    Returns (projected slope score over full history, diagnostics dict).
    """
    cols = [c for c in SCORING_CATEGORIES if c in front_long.columns]
    X = front_long[cols].dropna()
    diag: dict = {"pca_cols": cols}

    if X.shape[1] < 2 or X.loc[:train_end].shape[0] < 10:
        # Not enough features / training rows to isolate a slope factor.
        diag.update(pca_eigenvalues=None, pca_chosen_index=None,
                    pca_loadings=None, pca_kaiser=None)
        return pd.Series(index=front_long.index, dtype=float), diag

    train = X.loc[:train_end]
    scaler = StandardScaler().fit(train.values)
    Xz_train = scaler.transform(train.values)
    Xz_all = scaler.transform(X.values)

    pca = PCA(n_components=X.shape[1]).fit(Xz_train)
    eig = pca.explained_variance_                 # eigenvalues of the (train) covariance
    loadings = pca.components_                     # [n_comp, n_feat]
    scores_all = pca.transform(Xz_all)             # [n_rows, n_comp]

    # Identify opposite-loading (slope / differential) components.
    opposite = [i for i in range(loadings.shape[0])
                if np.sign(loadings[i, 0]) != np.sign(loadings[i, 1])]
    kaiser = [i for i in opposite if eig[i] > 1.0]
    if kaiser:
        idx = int(kaiser[np.argmax(eig[kaiser])])
    elif opposite:
        idx = int(opposite[np.argmax(eig[opposite])])
    else:
        idx = 1 if loadings.shape[0] > 1 else 0

    # Orient deterministically: make the first-category loading positive so re-runs are stable.
    sign = 1.0 if loadings[idx, 0] >= 0 else -1.0
    slope = pd.Series(sign * scores_all[:, idx], index=X.index).reindex(front_long.index)

    diag.update(
        pca_eigenvalues=[round(float(e), 4) for e in eig],
        pca_chosen_index=idx,
        pca_loadings=[round(float(v), 4) for v in (sign * loadings[idx])],
        pca_kaiser=bool(eig[idx] > 1.0),
    )
    return slope, diag


def build(inp: VersionInputs, params: dict | None = None) -> VersionResult:
    p = P.merge("v4", params)
    hl = int(p["ewma_hl"])
    z_w = int(p["z_window"])
    cot_w = int(p["cot_window"])
    train_end = p.get("train_end", TRAIN_END)

    # --- 1. BOOK TILT (raw level, $mm/bp) -------------------------------------------------- #
    book = P.resolve_front_long(inp, p)            # + = front-long (steepener book); tenor sets editable
    book_tilt_latest = _last(book)

    # --- 2. MOMENTUM (fast-normalized) ----------------------------------------------------- #
    mom = ewma_z(book, hl).rename("v4_curve_momentum")   # + = steepener-direction momentum
    z52 = rolling_z(book, z_w).rename("v4_curve_z52")

    # --- 3. PCA slope factor (fit on train, projected forward) ----------------------------- #
    pca_slope, pca_diag = _pca_slope_factor(inp.front_long, train_end)

    # --- 4. COT-index of the curve book ---------------------------------------------------- #
    cot = cot_index(book, cot_w).rename("v4_curve_cot")

    note = (
        "Two-dimension curve read: the RAW book tilt is ~flat (book_tilt "
        f"~ {book_tilt_latest:+.2f} $mm/bp) while MOMENTUM (EWMA-z hl=26) reads "
        f"steepener-direction ({_last(mom):+.2f}). The raw book being ~flat and the momentum "
        "reading steepener are not in conflict: momentum measures drift vs recent history, not "
        "the absolute level. composite/curve = momentum; POSITIVE = crowded steepener."
    )

    meta = {
        "book_tilt": book_tilt_latest,       # latest raw level ($mm/bp), ~flat
        "mom_latest": _last(mom),            # latest EWMA-z (hl=26)
        "z52_latest": _last(z52),            # latest 52w rolling-z (context)
        "pca_pc2_curve": _last(pca_slope),   # latest slope-factor projection
        "cot_front_long": _last(cot),        # latest 52w COT-index (0..100)
        "note": note,
        # extra diagnostics
        "book_tilt_units": "$mm/bp (+ = front-long / steepener book)",
        "momentum_halflife_weeks": hl,
        "z52_window_weeks": z_w,
        "cot_window_weeks": cot_w,
        "pca_train_end": train_end,
        "front_set": p.get("front_set"),
        "long_set": p.get("long_set"),
        **pca_diag,
    }

    return VersionResult(
        name="V4 Curve (book-tilt vs momentum, PC2)",
        description=(
            "Curve positioning split into two explicit dimensions: (1) raw BOOK TILT level "
            "of the buy-side front(TU+FV)-minus-long(US+UB) DV01 book ($mm/bp, currently "
            "~flat) and (2) fast MOMENTUM (EWMA-z, half-life 26w) which reads "
            "steepener-direction; supported by a 52w rolling-z, a train-fit PCA slope factor, "
            "and a 52w COT-index. composite=curve=momentum, POSITIVE = crowded steepener."
        ),
        composite=mom,          # NOTE: curve/steepener crowding, NOT base's duration meaning
        bounded=False,
        curve=mom,
        meta=meta,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    _inp = load_inputs()
    _res = build(_inp)
    print(_res.name)
    print(f"book_tilt (raw level, $mm/bp): {_res.meta['book_tilt']:+.4f}")
    print(f"momentum (EWMA-z hl=26):       {_res.meta['mom_latest']:+.4f}")
    print(f"z52 (52w rolling-z):           {_res.meta['z52_latest']:+.4f}")
    print(f"pca_pc2_curve (slope factor):  {_res.meta['pca_pc2_curve']:+.4f}")
    print(f"cot_front_long (0..100):       {_res.meta['cot_front_long']:.2f}")
    print(f"pca_eigenvalues:               {_res.meta.get('pca_eigenvalues')}")
    print(f"pca_loadings (chosen):         {_res.meta.get('pca_loadings')}")
    print(f"responsiveness(curve, False):  {responsiveness(_res.curve, False)}")
