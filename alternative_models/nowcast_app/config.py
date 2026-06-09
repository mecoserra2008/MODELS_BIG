"""
Target variable and regressor registry for the nowcasting app.

Each target: (fred_col, ecb_col, transform, freq, pub_lag_bdays)
Each regressor: (source_col, obs_freq, default_K, default_pub_lag_bdays)

pub_lag_bdays: how many business days after a reference period ends before
the data is typically published. Used to enforce the ragged-edge cutoff so
no future information leaks into the feature matrix.
"""

# ---------------------------------------------------------------------------
# Target variable registry
# ---------------------------------------------------------------------------
# (fred_col_or_None, ecb_col_or_None, transform, target_freq, pub_lag_bdays)
# transform: "yoy" | "mom" | "diff" | "lvl"
TARGETS: dict[str, tuple] = {
    # --- United States ---
    "US_CPI_YOY":  ("CPI",    None,              "yoy",  "M", 12),
    "US_CPI_MOM":  ("CPI",    None,              "mom",  "M", 12),
    "US_PPI_MOM":  ("PPI",    None,              "mom",  "M", 12),
    "US_PPI_YOY":  ("PPI",    None,              "yoy",  "M", 12),
    "US_NFP":      ("NFP",    None,              "diff", "M",  5),
    "US_UNRATE":   ("UNRATE", None,              "lvl",  "M", 10),
    "US_RETAIL":   ("RETAIL", None,              "mom",  "M", 14),
    "US_UMCSENT":  ("UMCSENT",None,              "lvl",  "M",  3),
    # --- Euro Area ---
    "EU_HICP_YOY": (None,     "EZ_HICP",         "yoy",  "M", 16),
    "EU_HICP_MOM": (None,     "EZ_HICP",         "mom",  "M", 16),
    "EU_RETAIL":   (None,     "EZ_RETAIL_MOM",   "lvl",  "M", 25),
    "EU_UNEMP":    (None,     "EZ_UNEMP",        "lvl",  "M", 30),
    # --- United Kingdom ---
    "UK_UNEMP":    ("UK_UNEMP", None,            "lvl",  "M", 45),
    "UK_WAGES":    ("UK_WAGES", None,            "mom",  "M", 45),
}

TARGET_GROUPS = {
    "United States": ["US_CPI_YOY","US_CPI_MOM","US_PPI_MOM","US_PPI_YOY",
                      "US_NFP","US_UNRATE","US_RETAIL","US_UMCSENT"],
    "Euro Area":     ["EU_HICP_YOY","EU_HICP_MOM","EU_RETAIL","EU_UNEMP"],
    "United Kingdom":["UK_UNEMP","UK_WAGES"],
}

# ---------------------------------------------------------------------------
# Regressor registry
# ---------------------------------------------------------------------------
# (source_col_in_merged_data, obs_freq, default_K, default_pub_lag_bdays)
# obs_freq: "D" daily | "W" weekly | "M" monthly
REGRESSORS: dict[str, tuple] = {
    # High-frequency financial (daily, 1-day pub lag)
    "VIX":           ("VIX",           "D", 22, 1),
    "SP500":         ("SP500",         "D", 22, 1),
    "OIL_WTI":       ("OIL_WTI",       "D", 22, 1),
    "EURUSD":        ("EURUSD",        "D", 22, 1),
    "CADUSD":        ("CADUSD",        "D", 22, 1),
    "USDBRL":        ("USDBRL",        "D", 22, 1),
    "CREDIT_SPREAD": ("CREDIT_SPREAD", "D", 22, 1),
    "YIELD_SPREAD":  ("YIELD_SPREAD",  "D", 22, 1),
    "FED_RATE":      ("FED_RATE",      "D", 22, 1),
    "GAS_PRICE":     ("GAS_PRICE",     "W",  4, 3),
    # Weekly indicators
    "CLAIMS":        ("CLAIMS",        "W",  4, 3),
    "CONT_CLAIMS":   ("CONT_CLAIMS",   "W",  4, 3),
    # Monthly macro
    "UNRATE":        ("UNRATE",        "M",  6, 10),
    "CPI":           ("CPI",           "M",  6, 12),
    "NFP":           ("NFP",           "M",  6,  5),
    "AHE":           ("AHE",           "M",  6, 10),
    "TEMP_HELP":     ("TEMP_HELP",     "M",  6, 10),
    "AWH_MFG":       ("AWH_MFG",       "M",  6, 10),
    "UMCSENT":       ("UMCSENT",       "M",  6, 10),
    "ISM_EMP":       ("ISM_EMP",       "M",  6,  3),
    "PPI":           ("PPI",           "M",  6, 12),
    "RETAIL":        ("RETAIL",        "M",  6, 14),
    "PRODUCTIVITY":  ("PRODUCTIVITY",  "M",  6, 30),
    "PARTICIPATION": ("PARTICIPATION", "M",  6, 10),
    "MICH_INF_EXP":  ("MICH_INF_EXP", "M",  6,  3),
    # Euro area
    "EZ_HICP":       ("EZ_HICP",       "M",  6, 16),
    "EZ_UNEMP":      ("EZ_UNEMP",      "M",  6, 30),
    "ECB_RATE":      ("ECB_RATE",      "M",  6,  0),
    # UK
    "UK_BOE_RATE":   ("UK_BOE_RATE",   "M",  6,  0),
    "UK_UNEMP":      ("UK_UNEMP",      "M",  6, 45),
    "UK_WAGES":      ("UK_WAGES",      "M",  6, 45),
    # Nelson-Siegel factors (daily)
    "US_Level":      ("US_Level",      "D", 22,  1),
    "US_Slope":      ("US_Slope",      "D", 22,  1),
    "US_Curvature":  ("US_Curvature",  "D", 22,  1),
    "EU_Level":      ("EU_Level",      "D", 22,  1),
    "EU_Slope":      ("EU_Slope",      "D", 22,  1),
    "UK_Level":      ("UK_Level",      "D", 22,  1),
    "UK_Slope":      ("UK_Slope",      "D", 22,  1),
}

REGRESSOR_GROUPS = {
    "Financial (Daily)": ["VIX","SP500","OIL_WTI","EURUSD","CADUSD","USDBRL",
                          "CREDIT_SPREAD","YIELD_SPREAD","FED_RATE"],
    "Weekly Indicators": ["CLAIMS","CONT_CLAIMS","GAS_PRICE"],
    "US Monthly":        ["UNRATE","CPI","NFP","AHE","TEMP_HELP","AWH_MFG",
                          "UMCSENT","ISM_EMP","PPI","RETAIL","PRODUCTIVITY",
                          "PARTICIPATION","MICH_INF_EXP"],
    "Euro Area":         ["EZ_HICP","EZ_UNEMP","ECB_RATE"],
    "United Kingdom":    ["UK_BOE_RATE","UK_UNEMP","UK_WAGES"],
    "NS Yield Factors":  ["US_Level","US_Slope","US_Curvature",
                          "EU_Level","EU_Slope","UK_Level","UK_Slope"],
}

# Default suggested regressors per target
TARGET_DEFAULT_REGRESSORS: dict[str, list[str]] = {
    "US_CPI_YOY":  ["OIL_WTI","GAS_PRICE","EURUSD","YIELD_SPREAD","FED_RATE",
                    "MICH_INF_EXP","PPI","US_Level"],
    "US_CPI_MOM":  ["OIL_WTI","GAS_PRICE","EURUSD","YIELD_SPREAD","FED_RATE",
                    "MICH_INF_EXP","PPI","US_Level"],
    "US_PPI_MOM":  ["OIL_WTI","EURUSD","YIELD_SPREAD","CREDIT_SPREAD","ISM_EMP","US_Slope"],
    "US_PPI_YOY":  ["OIL_WTI","EURUSD","YIELD_SPREAD","CREDIT_SPREAD","ISM_EMP","US_Slope"],
    "US_NFP":      ["CLAIMS","CONT_CLAIMS","ISM_EMP","TEMP_HELP","AWH_MFG",
                    "VIX","SP500","US_Slope"],
    "US_UNRATE":   ["NFP","CLAIMS","CONT_CLAIMS","PARTICIPATION","FED_RATE","US_Slope"],
    "US_RETAIL":   ["SP500","UMCSENT","GAS_PRICE","OIL_WTI","MICH_INF_EXP","US_Level"],
    "US_UMCSENT":  ["SP500","GAS_PRICE","OIL_WTI","VIX","UNRATE","CPI"],
    "EU_HICP_YOY": ["OIL_WTI","EURUSD","EU_Level","EU_Slope","ECB_RATE","EZ_UNEMP"],
    "EU_HICP_MOM": ["OIL_WTI","EURUSD","EU_Level","ECB_RATE"],
    "EU_RETAIL":   ["EU_Level","EURUSD","EZ_HICP","EZ_UNEMP","ECB_RATE","SP500"],
    "EU_UNEMP":    ["EZ_HICP","ECB_RATE","EU_Level","EU_Slope","SP500"],
    "UK_UNEMP":    ["UK_Level","UK_Slope","UK_BOE_RATE","CLAIMS","VIX","OIL_WTI"],
    "UK_WAGES":    ["UK_UNEMP","UK_BOE_RATE","UK_Level","OIL_WTI","GAS_PRICE"],
}

# Default Almon/lag parameters per observation frequency
DEFAULT_LAG_CONFIG: dict[str, dict] = {
    "D": {"weight_type": "Almon PDL", "K": 22, "degree": 2, "constraint": "far",
          "theta1": 1.0, "theta2": 5.0},
    "W": {"weight_type": "Almon PDL", "K":  4, "degree": 2, "constraint": "far",
          "theta1": 1.0, "theta2": 5.0},
    "M": {"weight_type": "Direct lags", "K": 3, "degree": 2, "constraint": "far",
          "theta1": 1.0, "theta2": 5.0},
}

TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
