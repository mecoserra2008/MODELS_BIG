"""
Reconstruct the primary-issuance event history from the snapshot and build the
discrete-time survival panel used by the timing hazard model.

Outputs (build_panel):
  grid          : issuer x week rows over the estimation window, with
                  Y (issued that week), weeks_since_last, issues_trailing_52w,
                  seasonality, and issuer static covariates.  One row = one
                  "at-risk" issuer-week (the unit of the discrete-time hazard).
  events        : bond-level table of realized in-window issuances (feeds the
                  conditional characteristic models).
  issuer_static : one row per active issuer with its covariates.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
from load_merge import build_security_table


def _to_week_idx(dates: pd.Series, base: pd.Period) -> pd.Series:
    """Integer week index (0-based from `base`) for a datetime series, W-MON."""
    per = dates.dt.to_period("W-MON")
    return (per.astype("int64") - base.ordinal).astype("float")


def build_panel():
    sec = build_security_table()
    as_of = pd.Timestamp(C.AS_OF)
    win_start = pd.Timestamp(C.WINDOW_START)
    warm_start = pd.Timestamp(C.WARMUP_START)

    # ---- realized issuance events (full history for features) ----
    ev = sec[sec["issue_date"].notna() & (sec["issue_date"] <= as_of) &
             (sec["issue_date"] >= warm_start)].copy()
    base = pd.Timestamp(C.WARMUP_START).to_period("W-MON")
    ev["wk"] = _to_week_idx(ev["issue_date"], base).astype(int)

    win_wk = int((win_start.to_period("W-MON").ordinal - base.ordinal))
    asof_wk = int((as_of.to_period("W-MON").ordinal - base.ordinal))
    roll_wk = win_wk - 52                       # extend back 1y so trailing-52 is warm at window start

    # ---- active issuers: >=1 event within the estimation window ----
    in_win = ev[(ev["wk"] >= win_wk) & (ev["wk"] <= asof_wk)]
    active = in_win["issuer"].unique()
    ev = ev[ev["issuer"].isin(active)].copy()
    ev.sort_values(["issuer", "wk"], inplace=True)

    # ---- issuer static covariates ----
    def _mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) else "(unknown)"

    grp = ev.groupby("issuer")
    issuer_static = pd.DataFrame({
        "issuer_type": grp["issuer_type"].agg(_mode),
        "sector": grp["sector"].agg(_mode),
        "country": grp["country"].agg(_mode),
        "issuer_rating_notch": grp["issuer_rating_notch"].median(),
        "first_wk": grp["wk"].min(),
        "n_events_all": grp.size(),
    })
    issuer_static["pais_setor"] = issuer_static["country"] + " | " + issuer_static["sector"]
    med_notch = issuer_static["issuer_rating_notch"].median()
    issuer_static["rating_filled"] = issuer_static["issuer_rating_notch"].fillna(med_notch)
    issuer_static["rating_missing"] = issuer_static["issuer_rating_notch"].isna().astype(int)
    issuer_static = issuer_static.reset_index()

    # ---- per-issuer event-week arrays (with repeats for multi-issuance weeks) ----
    ev_weeks = {iss: g["wk"].to_numpy() for iss, g in ev.groupby("issuer")}
    ev_count = {iss: (g["wk"].value_counts().sort_index()) for iss, g in ev.groupby("issuer")}

    # ---- build grid rows per issuer (from roll_wk .. asof, then trimmed) ----
    weeks_full = np.arange(roll_wk, asof_wk + 1)
    frames = []
    for iss in active:
        earr = ev_weeks[iss]                       # sorted event week idxs (repeats ok)
        first = earr.min()
        wk = weeks_full[weeks_full > first]        # at risk only after first observed issuance
        if wk.size == 0:
            continue
        # weeks since previous event (strictly before)
        prev_pos = np.searchsorted(earr, wk, side="left") - 1
        last_evt = earr[prev_pos]
        wsl = wk - last_evt
        # trailing-52w issuance count in [wk-52, wk-1]
        lo = np.searchsorted(earr, wk - 52, side="left")
        hi = np.searchsorted(earr, wk, side="left")
        trail = hi - lo
        # outcome: did an issuance occur this exact week
        y = np.isin(wk, earr).astype(np.int8)
        frames.append(pd.DataFrame({"issuer": iss, "wk": wk, "Y": y,
                                    "weeks_since_last": wsl, "issues_trailing_52w": trail}))

    grid = pd.concat(frames, ignore_index=True)
    grid = grid[grid["wk"] >= win_wk].copy()       # trim the rolling warm-up weeks

    # ---- seasonality from the calendar (week idx -> representative Monday) ----
    base_start = base.to_timestamp(how="start")
    ts = pd.DatetimeIndex(base_start + pd.to_timedelta(grid["wk"].to_numpy() * 7, unit="D"))
    grid["date"] = ts
    grid["week_of_year"] = ts.isocalendar().week.to_numpy()
    grid["month"] = ts.month
    grid["quarter_end"] = ts.month.isin([3, 6, 9, 12]).astype(int)
    grid["year"] = ts.year

    # ---- caps + transforms + join static covariates ----
    grid["weeks_since_last"] = grid["weeks_since_last"].clip(upper=260)
    grid["log_wsl"] = np.log1p(grid["weeks_since_last"])
    grid["log_trail"] = np.log1p(grid["issues_trailing_52w"])
    grid = grid.merge(issuer_static.drop(columns=["first_wk", "n_events_all"]),
                      on="issuer", how="left")

    events = in_win.copy()
    events["date"] = events["issue_date"]
    return grid, events, issuer_static


if __name__ == "__main__":
    grid, events, iss = build_panel()
    print("panel rows:", f"{len(grid):,}", " issuers:", grid['issuer'].nunique())
    print("weeks:", grid['date'].min().date(), "->", grid['date'].max().date())
    print("overall weekly issuance rate  Y=1:", round(grid['Y'].mean(), 4))
    print("in-window events (bond-level):", f"{len(events):,}")
    print("pais_setor groups:", iss['pais_setor'].nunique())
    print("\nweekly hazard by trailing-52 frequency bucket:")
    b = pd.cut(grid['issues_trailing_52w'], [-1, 0, 4, 12, 52, 10000],
              labels=['0', '1-4', '5-12', '13-52', '52+'])
    print(grid.groupby(b, observed=True)['Y'].agg(['mean', 'size']).round(4).to_string())
    print("\nweekly hazard vs weeks_since_last (bucketed):")
    w = pd.cut(grid['weeks_since_last'], [0, 1, 2, 4, 8, 26, 260])
    print(grid.groupby(w, observed=True)['Y'].agg(['mean', 'size']).round(4).to_string())
