"""
End-to-end driver: build the panel, fit the timing hazard and characteristic
models, run out-of-sample validation, write figures, and pickle the model
bundle used by the Streamlit app.

    python run_issuance_model.py
"""
from __future__ import annotations
import pickle
import time

import config as C
from panel import build_panel
from timing_model import fit_timing_model
from char_models import CharModels
from predict import IssuanceModel
import diagnostics


def main():
    t0 = time.time()
    print("[1/5] building issuance panel ...")
    grid, events, iss = build_panel()
    print(f"      {len(grid):,} issuer-weeks, {grid['issuer'].nunique():,} issuers, "
          f"{len(events):,} in-window issuances")

    print("[2/5] fitting cloglog timing hazard ...")
    tm = fit_timing_model(grid)

    print("[3/5] fitting conditional characteristic models ...")
    cm = CharModels.fit(events)

    model = IssuanceModel.build(tm, cm, iss, events)
    cm.events = None                       # drop raw frame from the pickled bundle
    with open(C.MODEL_BUNDLE, "wb") as f:
        pickle.dump(model, f)
    print(f"      bundle saved -> {C.MODEL_BUNDLE}")

    print("[4/5] out-of-sample validation + figures ...")
    metrics = diagnostics.run_all(grid, model)
    print(f"      AUC={metrics['auc']:.3f}  Brier={metrics['brier']:.4f}  "
          f"(test base rate {metrics['base_rate_test']:.4f})")

    print("[5/5] example queries:")
    for name in ["Federal Home Loan Bank System", "Government of Argentina"]:
        if name not in model.issuer_static.index:
            continue
        p1 = model.probability(name, ("within", 1))["probability"]
        p8 = model.probability(name, ("within", 8))["probability"]
        pc = model.probability(name, ("within", 8),
                               {"callable": True, "coupon_interval": (4, 5)})["probability"]
        print(f"      {name[:34]:34s}  next-wk={p1:.3f}  8-wk={p8:.3f}  callable&4-5%(8w)={pc:.4f}")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
