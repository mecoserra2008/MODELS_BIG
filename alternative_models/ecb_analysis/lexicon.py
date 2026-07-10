"""
ECB hawkish/dovish lexicon — scores text segments on a [-7.5, +7.5] scale.
Positive = hawkish, negative = dovish. Normalised to [0, 100] for barometer display.
Seed words from the MD file word-frequency table, extended with CB-NLP literature.
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Sequence

# ---------------------------------------------------------------------------
# Lexicon definition: term -> weight in [-7.5, +7.5]
# ---------------------------------------------------------------------------
LEXICON: dict[str, float] = {
    # --- Strong hawkish (+4 to +7.5) ---
    "inflation": 5.0,
    "persistent": 6.0,
    "above target": 6.5,
    "above 2": 5.5,
    "entrenched": 7.0,
    "second-round": 6.5,
    "second round": 6.5,
    "wage pressure": 6.0,
    "wages": 4.5,
    "wage growth": 5.0,
    "services inflation": 5.5,
    "services": 4.0,
    "vigilant": 6.0,
    "determined": 5.5,
    "forceful": 5.5,
    "act decisively": 6.0,
    "restrictive": 4.5,
    "sufficiently restrictive": 5.5,
    "further tightening": 6.5,
    "additional hike": 6.5,
    "tighten": 5.0,
    "tightening": 5.0,
    "raise rates": 6.0,
    "rate hike": 6.0,
    "upside risk": 5.0,
    "upside risks": 5.0,
    "energy prices": 3.5,
    "energy shock": 3.0,
    "core inflation": 5.0,
    "above consensus": 4.5,
    "above expectations": 4.5,
    "strong demand": 4.0,
    "labour market tight": 4.5,
    "tight labour": 4.5,
    "tightness": 4.0,
    "credibility": 3.5,
    "mandate": 3.0,
    "price stability": 3.5,
    "not yet done": 5.5,
    "not done": 5.0,
    "still work to do": 5.5,
    "projections revised upward": 5.5,
    "inflation expectations": 4.0,
    "de-anchoring": 6.5,
    "unanchored": 7.0,
    # --- Mild hawkish (+1 to +3.5) ---
    "target": 2.0,
    "energy": 2.0,
    "risks": 1.5,
    "upside": 2.5,
    "projections": 1.5,
    "resilient": 2.5,
    "robust": 2.0,
    "strong": 2.0,
    "elevated": 3.0,
    "above": 2.0,
    "concern": 2.0,
    "careful": 1.5,
    "monitor": 1.5,
    "watching closely": 2.0,
    "further meetings": 3.0,
    "warranted": 2.5,
    # --- Neutral (0) — present for completeness, weight = 0 ---
    "data dependent": 0.0,
    "data-dependent": 0.0,
    "meeting by meeting": 0.0,
    "meeting-by-meeting": 0.0,
    "incoming data": 0.0,
    "assessment": 0.0,
    # --- Mild dovish (-1 to -3.5) ---
    "uncertainty": -3.0,
    "downside": -2.5,
    "downside risks": -3.0,
    "growth": -1.5,
    "slowdown": -2.5,
    "weakness": -2.5,
    "transmission": -2.0,
    "pass-through": -2.0,
    "lag": -1.5,
    "lags": -1.5,
    "gradual": -2.5,
    "cautious": -2.5,
    "careful approach": -2.0,
    "balanced": -1.5,
    "symmetric": -1.5,
    "monitor closely": -1.5,
    "supply": -1.5,
    "temporary": -3.0,
    "transitory": -3.5,
    "base effect": -2.5,
    "energy fading": -3.0,
    "easing": -2.5,
    "below expectations": -3.0,
    "below consensus": -3.0,
    "weak demand": -3.0,
    "slack": -2.5,
    "output gap": -2.0,
    # --- Strong dovish (-4 to -7.5) ---
    "pause": -5.0,
    "hold": -4.5,
    "on hold": -5.0,
    "stop": -5.5,
    "rate cut": -7.0,
    "cut rates": -7.0,
    "lower rates": -6.5,
    "easing cycle": -6.5,
    "accommodation": -5.0,
    "accommodative": -5.5,
    "one and done": -6.0,
    "one-and-done": -6.0,
    "sufficient": -4.5,
    "sufficiently": -4.0,
    "peak": -5.0,
    "terminal rate": -4.5,
    "near peak": -5.5,
    "recession": -6.0,
    "contraction": -5.5,
    "disinflation": -5.0,
    "inflation falling": -5.5,
    "inflation declining": -5.5,
    "towards target": -4.0,
    "approaching target": -4.5,
    "anchored": -3.5,
    "well anchored": -4.5,
    "expectations anchored": -4.5,
}

# Topic cluster assignment for each term (used by scenario.py)
TOPIC_MAP: dict[str, str] = {
    "inflation": "Inflation", "persistent": "Inflation", "above target": "Inflation",
    "core inflation": "Inflation", "entrenched": "Inflation", "disinflation": "Inflation",
    "energy": "Energy", "energy prices": "Energy", "energy shock": "Energy",
    "energy fading": "Energy",
    "wages": "Wages", "wage growth": "Wages", "wage pressure": "Wages",
    "second-round": "Wages", "second round": "Wages",
    "services": "Services", "services inflation": "Services",
    "growth": "Growth", "recession": "Growth", "slowdown": "Growth",
    "downside": "Growth", "downside risks": "Growth", "slack": "Growth",
    "data-dependent": "Guidance", "data dependent": "Guidance",
    "meeting-by-meeting": "Guidance", "meeting by meeting": "Guidance",
    "pause": "Guidance", "hold": "Guidance", "one-and-done": "Guidance",
    "further meetings": "Guidance", "not yet done": "Guidance",
    "uncertainty": "Uncertainty", "transmission": "Uncertainty",
    "temporary": "Uncertainty", "transitory": "Uncertainty",
}


def _normalise(text: str) -> str:
    return text.lower().strip()


def score_segment(text: str) -> float:
    """
    Score a text segment on [-7.5, +7.5].
    Matches multi-word phrases first, then single words.
    Returns a per-word-normalised score.
    """
    text_lower = _normalise(text)
    words = re.findall(r"[a-z'-]+", text_lower)
    if not words:
        return 0.0

    weighted_sum = 0.0
    matched_positions: set[int] = set()

    # --- multi-word phrase scan (longest-match wins) ---
    phrases = sorted(
        [(k, v) for k, v in LEXICON.items() if " " in k or "-" in k],
        key=lambda x: -len(x[0])
    )
    for phrase, weight in phrases:
        idx = 0
        while True:
            pos = text_lower.find(phrase, idx)
            if pos == -1:
                break
            n_chars = len(phrase)
            # mark character positions as matched
            word_start = len(re.findall(r"[a-z'-]+", text_lower[:pos]))
            n_words_in_phrase = len(phrase.split())
            span = set(range(word_start, word_start + n_words_in_phrase))
            if not span & matched_positions:
                weighted_sum += weight
                matched_positions |= span
            idx = pos + n_chars

    # --- single-word scan ---
    for i, word in enumerate(words):
        if i in matched_positions:
            continue
        w = LEXICON.get(word, 0.0)
        weighted_sum += w

    # normalise by segment length
    raw = weighted_sum / max(len(words), 1)
    # clip to [-7.5, +7.5]
    return float(max(-7.5, min(7.5, raw)))


def to_barometer(score: float) -> float:
    """Map [-7.5, +7.5] score to [0, 100] barometer value."""
    return 50.0 + (score / 7.5) * 50.0


def top_terms(text: str, n: int = 5) -> list[tuple[str, float]]:
    """Return the n terms with the largest absolute contribution to the score."""
    text_lower = _normalise(text)
    hits: list[tuple[str, float]] = []

    phrases = sorted(
        [(k, v) for k, v in LEXICON.items() if " " in k or "-" in k],
        key=lambda x: -len(x[0])
    )
    for phrase, weight in phrases:
        if weight == 0.0:
            continue
        if phrase in text_lower:
            hits.append((phrase, weight))

    words = re.findall(r"[a-z'-]+", text_lower)
    counted = Counter(words)
    for word, cnt in counted.items():
        w = LEXICON.get(word, 0.0)
        if w != 0.0:
            hits.append((word, w * cnt))

    hits.sort(key=lambda x: abs(x[1]), reverse=True)
    seen: set[str] = set()
    unique: list[tuple[str, float]] = []
    for term, w in hits:
        if term not in seen:
            seen.add(term)
            unique.append((term, w))
    return unique[:n]


def label(score: float) -> str:
    bar = to_barometer(score)
    if bar >= 70:
        return "Hawkish"
    elif bar >= 55:
        return "Mildly Hawkish"
    elif bar >= 45:
        return "Neutral"
    elif bar >= 30:
        return "Mildly Dovish"
    else:
        return "Dovish"
