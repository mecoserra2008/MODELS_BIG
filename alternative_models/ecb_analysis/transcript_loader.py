"""
Fetches and parses ECB press conference transcripts from ecb.europa.eu.
Returns a list of segment dicts: {meeting_date, segment_index, speaker, text, word_count}.

Historical meetings used (from the MD file):
  - 2024-09-12, 2025-03-06, 2025-06-05, 2025-07-17, 2026-03-06

Live mode: also tails /tmp/ecb_live_feed.txt written by audio_pipe.py.
"""

from __future__ import annotations
import os
import re
import time
import hashlib
import json
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path(__file__).parent / "data"
CACHE_DIR.mkdir(exist_ok=True)

# ECB press conference transcript URLs (introductory statement + Q&A)
# Format: https://www.ecb.europa.eu/press/pressconf/YYYY/html/ecb.is{YYMMDD}~<hash>.en.html
KNOWN_MEETINGS: dict[str, str] = {
    "2024-09-12": "https://www.ecb.europa.eu/press/pressconf/2024/html/ecb.is240912~54d4cdfa5f.en.html",
    "2025-03-06": "https://www.ecb.europa.eu/press/pressconf/2025/html/ecb.is250306~0cf72b9748.en.html",
    "2025-06-05": "https://www.ecb.europa.eu/press/pressconf/2025/html/ecb.is250605~2e1c9c4f01.en.html",
    "2025-07-17": "https://www.ecb.europa.eu/press/pressconf/2025/html/ecb.is250717~4e45a793fb.en.html",
    "2026-03-06": "https://www.ecb.europa.eu/press/pressconf/2026/html/ecb.is260306~b3a4f7e291.en.html",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Fallback synthetic corpus when fetching fails — built from the MD file word-frequency
# table and the 5 scenario descriptions, labelled by expected FGBL reaction.
FALLBACK_CORPUS: list[dict] = [
    # --- Hawkish segments (score ~+4 to +7) ---
    {"meeting_date": "2024-09-12", "segment_index": 0, "speaker": "Lagarde",
     "text": "Inflation remains persistent and above our 2% target. "
             "Wages are growing strongly, feeding second-round effects into services inflation. "
             "We must act decisively to restore price stability and preserve credibility.",
     "bund_reaction": "lower"},
    {"meeting_date": "2024-09-12", "segment_index": 1, "speaker": "Lagarde",
     "text": "The upside risks to inflation are significant. Energy prices and wage growth "
             "are more elevated than our projections anticipated. "
             "Further meetings may warrant additional tightening.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-03-06", "segment_index": 0, "speaker": "Lagarde",
     "text": "Core inflation is still entrenched at levels inconsistent with our mandate. "
             "Services inflation is proving particularly persistent, driven by tight labour markets "
             "and strong wage growth. We remain vigilant and determined.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-03-06", "segment_index": 1, "speaker": "Lagarde",
     "text": "Inflation expectations risk becoming de-anchored if we do not act. "
             "We are not done yet. The restrictive stance must be maintained for as long "
             "as necessary to return inflation to target.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-06-05", "segment_index": 0, "speaker": "Lagarde",
     "text": "Today we decided to raise rates by 25 basis points. Inflation is too high "
             "and is set to remain so for too long. Our inflation projections have been "
             "revised upward significantly. Wages remain above consensus.",
     "bund_reaction": "lower"},
    # --- Neutral / balanced segments ---
    {"meeting_date": "2025-06-05", "segment_index": 1, "speaker": "Lagarde",
     "text": "We will continue to follow a data-dependent approach. Our decisions "
             "will be taken meeting by meeting based on the incoming data. "
             "We are not pre-committing to any particular rate path.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2025-07-17", "segment_index": 0, "speaker": "Lagarde",
     "text": "The Governing Council assessed that the current level of interest rates "
             "is contributing to bringing inflation back to target. Transmission of our "
             "policy is working through the economy. We monitor the data carefully.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2025-07-17", "segment_index": 1, "speaker": "Lagarde",
     "text": "There is uncertainty around the energy shock and its persistence. "
             "We balanced upside inflation risks against downside growth risks. "
             "The Governing Council remains flexible and open to reassessment.",
     "bund_reaction": "neutral"},
    # --- Dovish segments ---
    {"meeting_date": "2026-03-06", "segment_index": 0, "speaker": "Lagarde",
     "text": "Growth risks are tilted to the downside. We see clear signs of economic "
             "slowdown and weakness in the manufacturing sector. Transmission of higher "
             "rates is increasingly visible in credit and investment data.",
     "bund_reaction": "higher"},
    {"meeting_date": "2026-03-06", "segment_index": 1, "speaker": "Lagarde",
     "text": "Inflation is gradually moving towards our 2% target. The energy shock "
             "appears to be fading. Wage growth is stabilising. "
             "We see approaching target as our central scenario.",
     "bund_reaction": "higher"},
    {"meeting_date": "2026-03-06", "segment_index": 2, "speaker": "Lagarde",
     "text": "Uncertainty remains high. Data-dependent approach means we could pause "
             "if disinflation continues at pace. A one-and-done hike is the base case "
             "should the energy shock prove temporary.",
     "bund_reaction": "sharply higher"},
    # --- Additional Q&A style segments ---
    {"meeting_date": "2024-09-12", "segment_index": 2, "speaker": "Lagarde",
     "text": "Question about July signals. Our meeting-by-meeting approach is firm. "
             "We do not pre-commit. But upside risks are clearly dominant. "
             "The restrictive stance is warranted for longer.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-03-06", "segment_index": 2, "speaker": "Lagarde",
     "text": "On wages: the labour market remains tight. Unemployment is very low. "
             "Services inflation reflects these wage pressures. "
             "Second-round effects are a key upside risk.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-06-05", "segment_index": 2, "speaker": "Lagarde",
     "text": "On growth: we acknowledge some weakness in the industrial sector. "
             "However, demand remains resilient overall. The output gap is not deeply negative. "
             "Growth risks do not override our inflation mandate.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2025-07-17", "segment_index": 2, "speaker": "Lagarde",
     "text": "Inflation expectations are well anchored at 2%. The credibility of our "
             "framework is crucial. We will not hesitate to act if expectations shift upward. "
             "But the current trajectory supports a cautious gradual approach.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2026-03-06", "segment_index": 3, "speaker": "Lagarde",
     "text": "On the terminal rate: we see the current rate as sufficiently restrictive. "
             "If disinflation continues we may be near peak rates. "
             "Accommodation may become appropriate if growth deteriorates further.",
     "bund_reaction": "sharply higher"},
    {"meeting_date": "2026-03-06", "segment_index": 4, "speaker": "Lagarde",
     "text": "Energy prices are the key swing factor. If energy shock proves transitory "
             "and core inflation falls, we see no need for further hikes. "
             "Growth downside risks and slack in the economy support a pause.",
     "bund_reaction": "higher"},
    # --- Additional extended corpus (50+ total for t-SNE) ---
    {"meeting_date": "2024-09-12", "segment_index": 4, "speaker": "Lagarde",
     "text": "Services sector prices are proving stubborn. The pass-through from energy "
             "to services is incomplete. We need to see a sustained decline in services "
             "inflation before we can confidently dial back the restrictive stance.",
     "bund_reaction": "lower"},
    {"meeting_date": "2024-09-12", "segment_index": 5, "speaker": "Lagarde",
     "text": "The labour market remains tight across the euro area. Unemployment is at "
             "record lows. This supports wage growth and keeps services inflation elevated. "
             "Tightening financial conditions have not yet fully fed through.",
     "bund_reaction": "lower"},
    {"meeting_date": "2024-09-12", "segment_index": 6, "speaker": "Lagarde",
     "text": "We are not yet sufficiently confident that inflation is on a sustained "
             "downward path to 2%. Additional measures may be warranted. "
             "The risks to inflation are still tilted to the upside.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-03-06", "segment_index": 3, "speaker": "Lagarde",
     "text": "Core inflation, which strips out energy and food, remains sticky. "
             "It is running well above our 2% target, driven by persistent services inflation. "
             "We must be vigilant and not declare victory prematurely.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-03-06", "segment_index": 4, "speaker": "Lagarde",
     "text": "The decision today was unanimous. All members of the Governing Council "
             "agreed that current inflation dynamics require a further rate hike. "
             "Determined action now avoids more painful adjustment later.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-06-05", "segment_index": 4, "speaker": "Lagarde",
     "text": "On the projection revision: our staff have significantly upgraded the "
             "inflation path. We now see headline inflation above target through 2026. "
             "Wage growth is the key driver of the upward revision.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-06-05", "segment_index": 5, "speaker": "Lagarde",
     "text": "The energy component is contributing substantially to headline inflation. "
             "Beyond direct effects, energy is feeding into production costs and pushing "
             "up prices across the supply chain. This is not a temporary phenomenon.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-07-17", "segment_index": 4, "speaker": "Lagarde",
     "text": "We see some moderation in activity. Surveys point to weakening industrial "
             "production. Services are still holding up but showing early signs of softening. "
             "We are watching the data closely with symmetric attention to both sides.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2025-07-17", "segment_index": 5, "speaker": "Lagarde",
     "text": "Financial conditions have tightened meaningfully. The transmission of our "
             "policy rate increases is working its way through the banking system. "
             "Credit growth is slowing, which is the intended effect of our stance.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2025-07-17", "segment_index": 6, "speaker": "Lagarde",
     "text": "We acknowledge significant uncertainty in the global outlook. Trade "
             "fragmentation, geopolitical tensions, and energy supply dynamics "
             "cloud the picture. We will respond symmetrically to risks on either side.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2026-03-06", "segment_index": 5, "speaker": "Lagarde",
     "text": "Disinflation is proceeding at pace. Headline inflation has fallen "
             "substantially from its peak. Core inflation is declining, though services "
             "remain somewhat above target. The trajectory is encouraging.",
     "bund_reaction": "higher"},
    {"meeting_date": "2026-03-06", "segment_index": 6, "speaker": "Lagarde",
     "text": "We see recession risks as non-trivial in the current environment. "
             "The global outlook has deteriorated. Domestic demand is weakening. "
             "The case for maintaining maximum restriction is becoming harder to sustain.",
     "bund_reaction": "sharply higher"},
    {"meeting_date": "2026-03-06", "segment_index": 7, "speaker": "Lagarde",
     "text": "If the energy shock proves to be temporary and supply-driven, "
             "the inflationary impulse will fade without requiring prolonged restriction. "
             "A pause in rate hikes may be appropriate to assess the lag effects of transmission.",
     "bund_reaction": "higher"},
    {"meeting_date": "2026-03-06", "segment_index": 8, "speaker": "Lagarde",
     "text": "Inflation expectations remain well anchored at 2%. "
             "Market-based measures of long-run expectations have not de-anchored. "
             "This gives us some room to assess without urgency.",
     "bund_reaction": "higher"},
    {"meeting_date": "2024-09-12", "segment_index": 7, "speaker": "Lagarde",
     "text": "Household purchasing power is being squeezed by high energy prices. "
             "However, employment income and fiscal transfers are partially offsetting this. "
             "Demand has proved more resilient than expected despite tighter conditions.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2024-09-12", "segment_index": 8, "speaker": "Lagarde",
     "text": "We will raise rates as much as needed to ensure timely return to target. "
             "Our commitment to price stability is unconditional. "
             "A tighter stance for longer is preferable to premature easing.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-03-06", "segment_index": 5, "speaker": "Lagarde",
     "text": "The euro area economy contracted slightly in the fourth quarter. "
             "However, this was driven by temporary factors. Underlying momentum "
             "in services and employment points to resilience in domestic demand.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2025-06-05", "segment_index": 6, "speaker": "Lagarde",
     "text": "Some members of the Governing Council raised the possibility of pausing "
             "after this hike. However, the majority view held that continued tightening "
             "is necessary given the inflation outlook and upside risks.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2026-03-06", "segment_index": 9, "speaker": "Lagarde",
     "text": "Peak rates. We believe the current rate level may be close to sufficiently "
             "restrictive. Further hikes are not the baseline. "
             "We will hold rates at this level as long as needed, then assess.",
     "bund_reaction": "sharply higher"},
    {"meeting_date": "2026-03-06", "segment_index": 10, "speaker": "Lagarde",
     "text": "Rate cuts are not being discussed at this point. "
             "We are in a hold mode after this decision. "
             "Accommodation is a future consideration, not an imminent one.",
     "bund_reaction": "neutral_higher"},
    {"meeting_date": "2025-07-17", "segment_index": 7, "speaker": "Lagarde",
     "text": "The cumulative tightening of 400 basis points is substantial. "
             "We must now assess how much further restriction is needed. "
             "Data dependence means we keep all options open for future meetings.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2024-09-12", "segment_index": 9, "speaker": "Lagarde",
     "text": "Energy companies passed through cost increases rapidly. "
             "Now we need prices to reverse as energy costs fall. "
             "But second-round wage effects may keep core inflation elevated beyond the energy move.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-03-06", "segment_index": 6, "speaker": "Lagarde",
     "text": "The banking sector remains resilient. We do not see credit market stress "
             "threatening financial stability at this point. "
             "Our monetary tightening is working through the economy as intended.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2025-06-05", "segment_index": 7, "speaker": "Lagarde",
     "text": "Forward guidance is intentionally limited at this point. "
             "We will not pre-commit to a rate path. "
             "Flexibility is essential given high uncertainty in the energy and wage outlook.",
     "bund_reaction": "neutral"},
    {"meeting_date": "2026-03-06", "segment_index": 11, "speaker": "Lagarde",
     "text": "Growth is slowing. Consumption is weak. Investment has fallen. "
             "The output gap is closing and may turn negative. "
             "These are signs that our restrictive policy is having real economic effects.",
     "bund_reaction": "higher"},
    {"meeting_date": "2025-06-05", "segment_index": 3, "speaker": "Lagarde",
     "text": "On credibility: we are determined to return inflation to 2% in a timely manner. "
             "The price stability mandate is non-negotiable. "
             "Further tightening cannot be ruled out if the data warrants it.",
     "bund_reaction": "lower"},
    {"meeting_date": "2024-09-12", "segment_index": 3, "speaker": "Lagarde",
     "text": "On energy: energy prices have rebounded and are feeding into headline "
             "and now increasingly into core inflation through second-round effects. "
             "This is a concern. We must ensure pass-through does not become entrenched.",
     "bund_reaction": "lower"},
    {"meeting_date": "2025-07-17", "segment_index": 3, "speaker": "Lagarde",
     "text": "Projections show inflation returning to target by mid-2026 under our baseline. "
             "Uncertainty bands are wide. Both upside and downside risks are material. "
             "We will act symmetrically in either direction if needed.",
     "bund_reaction": "neutral"},
]


def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    return CACHE_DIR / f"ecb_{h}.html"


def _resolve_ecb_url(meeting_date: str) -> str | None:
    """
    Discover the actual ECB press conference URL for a given date by searching
    the ECB pressconf index page. The hash in the URL changes and cannot be hardcoded.
    """
    year = meeting_date[:4]
    date_compact = meeting_date.replace("-", "")[2:]  # e.g. "240912"
    index_url = f"https://www.ecb.europa.eu/press/pressconf/{year}/html/index.en.html"
    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if f"is{date_compact}" in href and ".en.html" in href:
                if href.startswith("http"):
                    return href
                return f"https://www.ecb.europa.eu{href}"
    except Exception:
        pass
    return None


def _fetch_html(url: str, timeout: int = 15) -> str | None:
    cache = _cache_path(url)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        cache.write_text(resp.text, encoding="utf-8")
        return resp.text
    except Exception as exc:
        print(f"[transcript_loader] Could not fetch {url}: {exc}")
        return None


def _parse_ecb_html(html: str, meeting_date: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    # ECB transcripts use <p> tags; speaker names appear in <strong> or <b>
    segments = []
    current_speaker = "Lagarde"
    seg_idx = 0

    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text or len(text) < 30:
            continue
        # Detect speaker line: "LAGARDE:" or "PRESIDENT LAGARDE"
        speaker_match = re.match(r"^([A-Z][A-Z\s]{2,30})[:.]", text)
        if speaker_match:
            name = speaker_match.group(1).strip().title()
            if any(kw in name.upper() for kw in ["LAGARDE", "PRESIDENT", "MODERATOR"]):
                current_speaker = "Lagarde"
            else:
                current_speaker = name
            text = text[speaker_match.end():].strip()
        if len(text) < 30:
            continue
        segments.append({
            "meeting_date": meeting_date,
            "segment_index": seg_idx,
            "speaker": current_speaker,
            "text": text,
            "word_count": len(text.split()),
        })
        seg_idx += 1

    return segments


def load_historical(max_meetings: int = 5) -> list[dict]:
    """
    Fetch historical ECB transcripts. Falls back to the synthetic corpus if
    fetching fails (network issues, URL changes, etc.).
    """
    all_segments: list[dict] = []
    meetings = list(KNOWN_MEETINGS.items())[:max_meetings]

    for meeting_date, known_url in meetings:
        # Try to resolve the real URL from the ECB index, fall back to known URL
        resolved = _resolve_ecb_url(meeting_date) or known_url
        html = _fetch_html(resolved)
        if html:
            segs = _parse_ecb_html(html, meeting_date)
            if segs:
                # Assign FGBL reaction label heuristically from lexicon score
                from lexicon import score_segment, to_barometer
                for s in segs:
                    bar = to_barometer(score_segment(s["text"]))
                    if bar >= 65:
                        s["bund_reaction"] = "lower"
                    elif bar >= 55:
                        s["bund_reaction"] = "neutral_lower"
                    elif bar >= 45:
                        s["bund_reaction"] = "neutral"
                    elif bar >= 35:
                        s["bund_reaction"] = "neutral_higher"
                    else:
                        s["bund_reaction"] = "higher"
                all_segments.extend(segs)
                print(f"[transcript_loader] Loaded {len(segs)} segments from {meeting_date}")
                continue
        # fallback: use synthetic segments for this meeting
        fallback = [s for s in FALLBACK_CORPUS if s["meeting_date"] == meeting_date]
        all_segments.extend(fallback)
        print(f"[transcript_loader] Using fallback corpus for {meeting_date} ({len(fallback)} segments)")

    if not all_segments:
        print("[transcript_loader] Using full synthetic fallback corpus.")
        return [dict(s) for s in FALLBACK_CORPUS]

    return all_segments


def tail_live_feed(feed_path: str = "/tmp/ecb_live_feed.txt") -> Iterator[dict]:
    """
    Yields new segments as audio_pipe.py appends them to the live feed file.
    Each line in the file is a JSON object with 'timestamp' and 'text' fields.
    """
    path = Path(feed_path)
    seg_idx = 0
    last_size = 0

    while True:
        if not path.exists():
            time.sleep(2)
            continue
        current_size = path.stat().st_size
        if current_size > last_size:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(last_size)
                new_content = f.read()
            last_size = current_size
            for line in new_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    text = line
                if text:
                    yield {
                        "meeting_date": "2026-06-11",
                        "segment_index": seg_idx,
                        "speaker": "Lagarde",
                        "text": text,
                        "word_count": len(text.split()),
                    }
                    seg_idx += 1
        time.sleep(3)
