"""
Configuration for the debt-issuance probability model.

Central place for: file locations, the estimation window / as-of date, the
Portuguese->English category translations, the rating notch scale, and the
coupon / tenor bucket definitions.

Everything downstream (load_merge, panel, features, models, app) imports from
here so the vocabulary is defined exactly once.
"""
from __future__ import annotations
import os
import glob
import unicodedata

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _find(suffix: str) -> str:
    """Locate one of the 7 snapshot CSVs by filename suffix."""
    hits = [f for f in glob.glob(os.path.join(BASE_DIR, "*.csv")) if f.endswith(suffix)]
    if not hits:
        raise FileNotFoundError(f"No CSV ending in {suffix!r} under {BASE_DIR}")
    return hits[0]


# The seven files, addressed by their trailing suffix (robust to the space in
# the filename).  File (7) is the no-suffix master (it alone carries ISIN).
def csv_paths() -> dict[str, str]:
    return {
        "issue":   _find("(1).csv"),   # issue/maturity/tenor/placement
        "amount":  _find("(2).csv"),   # amounts + currency
        "struct":  _find("(3).csv"),   # optionality / seniority
        "coupon":  _find("(4).csv"),   # coupon mechanics
        "issuer":  _find("(5).csv"),   # Emissor / country / sector / ratings
        "rating":  _find("(6).csv"),   # issue-level ratings + outlooks
        "master":  _master_path(),     # ISIN / exchange / issuer type
    }


def _master_path() -> str:
    hits = [
        f for f in glob.glob(os.path.join(BASE_DIR, "*.csv"))
        if f.endswith("2026-07-28.csv") and "(" not in os.path.basename(f)
    ]
    if not hits:
        raise FileNotFoundError("Master (no-suffix) CSV not found")
    return hits[0]


# --------------------------------------------------------------------------
# Time frame
# --------------------------------------------------------------------------
AS_OF = "2026-07-28"          # snapshot date == prediction "now"
WINDOW_START = "2019-01-01"   # primary estimation window start
WARMUP_START = "2012-01-01"   # earlier events used only to warm up duration/frequency features

WEEK_FREQ = "W-MON"           # ISO-ish weekly buckets anchored on Monday

# --------------------------------------------------------------------------
# Portuguese -> English / normalized category maps
# --------------------------------------------------------------------------
COUPON_TYPE = {                       # column: "Tipo de cupom"
    "Fixo": "fixed",
    "Variável": "floating",
    "Zero": "zero",
}

DURATION_TYPE = {                     # column: "Tipo de duração (prazo)"
    "Curto prazo": "short",
    "Médio prazo": "medium",
    "Longo prazo": "long",
    "Perpétuo": "perpetual",
}

# "Tipo de resgate (call/put)" -> callable / puttable indicators
REDEMPTION_CALLABLE = {"Callable", "Callable e putable"}
REDEMPTION_PUTTABLE = {"Putable", "Callable e putable"}

PLACEMENT = {"Público": "public", "Privado": "private"}
ALLOCATION = {"Global": "global", "País único": "single_country"}
SINKABLE = {"Sinkable": 1, "Non-sinkable": 0}
CONVERTIBLE = {"Não conversível": 0, "Conversíveis": 1, "Permutável": 1}
EARLY_REDEEM = {"Resgatável": 1, "Não-resgatável": 0}

# Seniority collapsed to an ordinal (higher == more senior / safer)
SENIORITY_ORD = {
    "Sênior": 5, "Dívida sênior preferencial": 5, "Sênior preferencial": 5,
    "Sênior não preferencial": 4,
    "Sênior subordinado": 3,
    "Subordinado": 2,
    "Júnior preferencial": 2, "Júnior subordinado": 1, "Júnior": 1,
    "Não relevante": 3, "Não divulgado": 3,   # unknown -> middle
}

# --------------------------------------------------------------------------
# Rating notch scale (S&P / Fitch letter -> integer, higher == better credit)
# --------------------------------------------------------------------------
_RATING_ORDER = [
    "D", "C", "CC", "CCC-", "CCC", "CCC+", "B-", "B", "B+",
    "BB-", "BB", "BB+", "BBB-", "BBB", "BBB+",
    "A-", "A", "A+", "AA-", "AA", "AA+", "AAA",
]
RATING_NOTCH = {r: i + 1 for i, r in enumerate(_RATING_ORDER)}  # D=1 ... AAA=22


def normalize_minus(s: str) -> str:
    """Normalize the Unicode minus (U+2212) some files use to an ASCII hyphen."""
    if s is None:
        return s
    return unicodedata.normalize("NFKC", str(s)).replace("−", "-").strip()


def rating_to_notch(s):
    """Map an S&P/Fitch rating string to a 1..22 notch, else NaN.

    Handles the Unicode-minus quirk, 'NR'/blank as missing, and strips any
    outlook/watch suffix (e.g. 'BBB- *+').
    """
    if s is None:
        return None
    s = normalize_minus(s)
    if s in ("", "NR", "WR", "NaN", "nan"):
        return None
    token = s.split()[0].split("/")[0]
    return RATING_NOTCH.get(token)


# --------------------------------------------------------------------------
# Buckets used by the query interface
# --------------------------------------------------------------------------
COUPON_BUCKETS = [(-0.01, 2), (2, 4), (4, 5), (5, 7), (7, 100)]
COUPON_BUCKET_LABELS = ["0-2%", "2-4%", "4-5%", "5-7%", "7%+"]

# Tenor buckets in calendar days (issue -> maturity)
TENOR_BUCKETS_DAYS = [(0, 366 * 3), (366 * 3, 366 * 7), (366 * 7, 366 * 12), (366 * 12, 10_000 * 366)]
TENOR_BUCKET_LABELS = ["<3y", "3-7y", "7-12y", "12y+"]

# Local-currency heuristic: currency of the issuer's country (used for the
# "FX vs local" characteristic).  Only the common ones matter for queries.
LOCAL_CCY_BY_COUNTRY = {
    "Estados Unidos": "USD", "Alemanha": "EUR", "França": "EUR", "Itália": "EUR",
    "Espanha": "EUR", "Países Baixos": "EUR", "Áustria": "EUR", "Bélgica": "EUR",
    "Reino Unido": "GBP", "Canadá": "CAD", "Índia": "INR", "Israel": "ILS",
    "China": "CNY", "Suíça": "CHF", "Japão": "JPY", "Argentina": "ARS",
    "Brasil": "BRL", "México": "MXN", "Austrália": "AUD",
}

# The model artifact bundle written by run_issuance_model.py
MODEL_BUNDLE = os.path.join(RESULTS_DIR, "model_bundle.pkl")
