"""Central configuration for the Fixed Income positioning score.

This is the de-facto registry: which instruments, trader categories, and contract
specs feed the score. Both the free (Track A) and Bloomberg/BQuant (Track B) adapters
target the same instrument keys defined here, so the scoring core is source-agnostic.

Nothing here fetches data; it is pure declarative config.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data" / "raw"
RESULTS_DIR = PKG_DIR / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# FRED key lives in the alternative_models/.env in this repo (see run_10s5s.py).
REPO_ROOT = PKG_DIR.parent
FRED_ENV_PATH = REPO_ROOT / "alternative_models" / ".env"


# --------------------------------------------------------------------------- #
# US Treasury futures complex
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContractSpec:
    """Specification of one Treasury futures contract.

    key            short internal id (also the CME root): TU/FV/TY/TN/US/UB
    label          human label
    cftc_code      CFTC contract_market_code (stable across the 2022 name change)
    tenor_years    representative CTD tenor, used for the DV01 approximation
    face_value     contract face/notional in USD
    fred_yield     FRED series whose yield drives the DV01 modified-duration proxy
    """

    key: str
    label: str
    cftc_code: str
    tenor_years: float
    face_value: float
    fred_yield: str


# Ordered short -> long. cftc_code confirmed live against publicreporting.cftc.gov.
CONTRACTS: tuple[ContractSpec, ...] = (
    ContractSpec("TU", "2Y T-Note",       "042601",  1.9, 200_000, "DGS2"),
    ContractSpec("FV", "5Y T-Note",       "044601",  4.3, 100_000, "DGS5"),
    ContractSpec("TY", "10Y T-Note",      "043602",  6.8, 100_000, "DGS10"),
    ContractSpec("TN", "Ultra 10Y T-Note","043607",  9.2, 100_000, "DGS10"),
    ContractSpec("US", "T-Bond",          "020601", 15.5, 100_000, "DGS30"),
    ContractSpec("UB", "Ultra T-Bond",    "020604", 19.5, 100_000, "DGS30"),
)
CONTRACT_BY_CODE = {c.cftc_code: c for c in CONTRACTS}
CONTRACT_BY_KEY = {c.key: c for c in CONTRACTS}
CFTC_CODES = [c.cftc_code for c in CONTRACTS]

# Front vs long end split for the curve-positioning factor (by contract key).
FRONT_END = ("TU", "FV")
LONG_END = ("US", "UB")

# --------------------------------------------------------------------------- #
# Trader categories (CFTC Traders in Financial Futures, TFF)
# --------------------------------------------------------------------------- #
# key -> (long_field, short_field) in the CFTC Socrata TFF dataset (gpe5-46if).
TFF_CATEGORIES: dict[str, tuple[str, str]] = {
    "lev_money": ("lev_money_positions_long", "lev_money_positions_short"),
    "asset_mgr": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
    "dealer":    ("dealer_positions_long_all", "dealer_positions_short_all"),
}
# Behavioural priors used by the scoring core:
#  - lev_money: fast money, most predictive at extremes -> level AND change matter
#  - asset_mgr: real money, structurally long -> change/flow matters more than level
#  - dealer:    offsetting counterparty -> mirror / sanity check
CATEGORY_LABELS = {
    "lev_money": "Leveraged Funds",
    "asset_mgr": "Asset Managers",
    "dealer": "Dealer/Intermediary",
}
# Categories that actually enter the composite (dealer kept for display/mirror only).
SCORING_CATEGORIES = ("lev_money", "asset_mgr")

CONCENTRATION_FIELDS = (
    "conc_net_le_4_tdr_long_all",
    "conc_net_le_4_tdr_short_all",
)

# --------------------------------------------------------------------------- #
# NY Fed primary-dealer cash Treasury net positions (FR2004), coupon by tenor.
# keyid -> representative tenor (years) for DV01 weighting of the cash leg.
# --------------------------------------------------------------------------- #
NYFED_CASH_SERIES: dict[str, float] = {
    "PDPOSGSC-L2":     1.0,   # <= 2y
    "PDPOSGSC-G2L3":   2.5,   # 2-3y
    "PDPOSGSC-G3L6":   4.5,   # 3-6y
    "PDPOSGSC-G6L7":   6.5,   # 6-7y
    "PDPOSGSC-G7L11":  9.0,   # 7-11y
    "PDPOSGSC-G11L21": 16.0,  # 11-21y
    "PDPOSGSC-G21":    25.0,  # > 21y
}

# --------------------------------------------------------------------------- #
# Methodology parameters
# --------------------------------------------------------------------------- #
# Publication lags (business days) so the score at t only uses data known by t.
COT_PUBLICATION_LAG_BDAYS = 3    # Tue snapshot released the following Friday
NYFED_PUBLICATION_LAG_BDAYS = 6  # ~1 week

# Normalisation lookbacks (weeks).
Z_LOOKBACK_SHORT = 52    # 1y
Z_LOOKBACK_LONG = 156    # 3y

# Train/test split for PCA + predictive layer (fit only on train, project forward).
TRAIN_END = "2019-12-31"

# Predictive overlay: forward horizon (weeks) and the yield proxy for the label.
FORWARD_HORIZON_WEEKS = 4
LABEL_YIELD = "DGS10"   # selloff = yield up over the horizon

# Contrarian signal band thresholds on the composite z-index (with hysteresis).
SIGNAL_K_ENTRY = 1.5
SIGNAL_K_EXIT = 0.5

DEFAULT_START = "2006-06-13"  # first CFTC TFF report date
