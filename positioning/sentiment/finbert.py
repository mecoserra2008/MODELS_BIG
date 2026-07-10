"""FinBERT headline sentiment, oriented to the *bond* convention.

FinBERT (ProsusAI/finbert, trained on the Financial PhraseBank) reads generic financial
tone: it calls "Treasury yields surge on hot inflation" *positive* because "surge" reads
bullish — but that is **bearish for bonds**. Raw FinBERT is therefore misleading on a
fixed-income desk.

So the final per-headline score in [-1, +1] is **bond-oriented**:
* FinBERT supplies the tone *magnitude / confidence* (|P(pos) - P(neg)|),
* a compact rate-direction layer supplies the *sign* in bond terms (dovish/cuts/rally/
  safe-haven → +bullish bonds; hawkish/hikes/hot-inflation/supply/selloff → -bearish),
* when a headline has no rate-direction cue, we fall back to FinBERT's own signed tone.

Positive = bullish fixed income (yields lower). The model is loaded once and cached.
If transformers/torch or the weights are unavailable, we fall back to the repo's finance
lexicon (ecb_analysis/lexicon), already inverted to the bond convention.
"""
from __future__ import annotations

import functools
import os
import re
import sys
from pathlib import Path

import numpy as np

MODEL_NAME = "ProsusAI/finbert"
# finbert label order can vary; we map by name at runtime.
_POS, _NEG = "positive", "negative"

_ECB_DIR = Path(__file__).resolve().parents[2] / "alternative_models" / "ecb_analysis"

# Rate-direction cues (bond convention). Bullish bonds = yields down.
_BULLISH_BONDS = (
    "rate cut", "rate cuts", "cut rates", "dovish", "rally", "rallies", "easing",
    "ease", "cooling", "cools", "cooler", "soft", "softer", "disinflation",
    "safe haven", "safe-haven", "haven bid", "flight to quality", "lower yields",
    "yields fall", "yields drop", "yields decline", "yields slide", "recession",
    "slowdown", "weak jobs", "dovish hold", "bid", "bonds gain",
)
_BEARISH_BONDS = (
    "rate hike", "rate hikes", "hike", "hikes", "hawkish", "surge", "surges",
    "spike", "jump", "climb", "hot inflation", "sticky inflation", "higher yields",
    "yields rise", "yields climb", "yields surge", "yields jump", "selloff",
    "sell-off", "tantrum", "supply", "oversupply", "upsize", "strong jobs",
    "robust", "sticky", "reprice", "term premium",
)


def _direction(headline: str) -> int:
    """+1 bond-bullish, -1 bond-bearish, 0 no clear rate cue."""
    t = headline.lower()
    b = sum(1 for k in _BULLISH_BONDS if k in t)
    s = sum(1 for k in _BEARISH_BONDS if re.search(r"\b" + re.escape(k), t))
    if b > s:
        return 1
    if s > b:
        return -1
    return 0


@functools.lru_cache(maxsize=1)
def _pipeline():
    """Load FinBERT once. Returns a callable(list[str]) -> list[dict], or None.

    Opt-in only: unless POSITIONING_USE_FINBERT=1, we never import transformers. This
    keeps the default path lexicon-based (matching Streamlit Cloud) and avoids Streamlit's
    file-watcher enumerating transformers.models.* (which triggers a torchvision version
    mismatch spam locally). Set the env var on a machine with torch/transformers to enable.
    """
    if os.environ.get("POSITIONING_USE_FINBERT") != "1":
        return None
    try:
        import torch  # noqa: F401
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                   TextClassificationPipeline)
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, top_k=None,
                                          truncation=True, max_length=128)
        return pipe
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _lexicon_scorer():
    """Fallback: the repo's finance lexicon scorer, normalized to [-1, 1]."""
    try:
        if str(_ECB_DIR) not in sys.path:
            sys.path.insert(0, str(_ECB_DIR))
        import lexicon  # type: ignore

        def score(text: str) -> float:
            # score_segment returns [-7.5, +7.5] but per-token normalization compresses
            # short headlines; scale so headline-level signals reach usable magnitude.
            # hawkish(+) => bearish bonds, so invert to the bond convention.
            return float(np.clip(-lexicon.score_segment(text) / 2.0, -1.0, 1.0))
        return score
    except Exception:
        return None


def backend() -> str:
    """Which scorer is active: 'finbert' or 'lexicon' (or 'none')."""
    if _pipeline() is not None:
        return "finbert"
    if _lexicon_scorer() is not None:
        return "lexicon"
    return "none"


def raw_finbert(headlines: list[str]) -> np.ndarray:
    """Generic FinBERT signed tone P(pos) - P(neg) in [-1, 1] (NOT bond-oriented)."""
    pipe = _pipeline()
    if pipe is None or not headlines:
        return np.zeros(len(headlines), dtype=float)
    out = pipe(list(headlines))
    scores = []
    for row in out:
        d = {r["label"].lower(): r["score"] for r in row}
        scores.append(d.get(_POS, 0.0) - d.get(_NEG, 0.0))
    return np.array(scores, dtype=float)


def score_headlines(headlines: list[str]) -> np.ndarray:
    """Bond-oriented sentiment in [-1, 1] per headline (positive = bullish FI).

    FinBERT confidence x rate-direction sign; falls back to FinBERT tone when no rate cue,
    or to the (already bond-inverted) lexicon when FinBERT is unavailable.
    """
    if not headlines:
        return np.array([])
    if _pipeline() is not None:
        fb = raw_finbert(headlines)
        out = []
        for h, f in zip(headlines, fb):
            d = _direction(h)
            out.append(d * abs(f) if d != 0 else float(f))
        return np.array(out, dtype=float)
    lex = _lexicon_scorer()
    if lex is not None:
        return np.array([lex(h) for h in headlines], dtype=float)
    return np.zeros(len(headlines), dtype=float)
