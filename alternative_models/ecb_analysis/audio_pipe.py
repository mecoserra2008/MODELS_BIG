"""
Live audio transcription pipeline (codespace-native, no microphone needed).

ffmpeg fetches the ECB livestream URL directly in the codespace and writes
20-second .wav chunks. Whisper-base transcribes each chunk and appends the
text as a JSON line to /tmp/ecb_live_feed.txt.

Usage (called from Streamlit or CLI):
    from audio_pipe import start_pipeline, stop_pipeline
    start_pipeline("https://www.youtube.com/live/...")
    # ...
    stop_pipeline()
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

FEED_FILE = Path("/tmp/ecb_live_feed.txt")
CHUNK_DIR = Path("/tmp/ecb_chunks")
CHUNK_DURATION = 20   # seconds per chunk
WHISPER_MODEL = "tiny.en"   # 72 MB, RTF≈0.46 on CPU — keeps up with 20s chunks

_ffmpeg_proc: subprocess.Popen | None = None
_transcribe_thread: threading.Thread | None = None
_stop_event = threading.Event()
_is_running = False


def _chunk_filename(idx: int) -> Path:
    return CHUNK_DIR / f"chunk_{idx:04d}.wav"


def _run_ffmpeg(stream_url: str, stop_event: threading.Event):
    """
    Continuously pull audio from stream_url, segment into CHUNK_DURATION .wav files.
    ffmpeg exits cleanly when stop_event is set (process is killed).
    """
    global _ffmpeg_proc
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", stream_url,
        "-vn",                        # no video
        "-ac", "1",                   # mono
        "-ar", "16000",               # 16 kHz for Whisper
        "-f", "segment",
        "-segment_time", str(CHUNK_DURATION),
        "-segment_format", "wav",
        "-reset_timestamps", "1",
        str(CHUNK_DIR / "chunk_%04d.wav"),
    ]

    try:
        _ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        while not stop_event.is_set():
            if _ffmpeg_proc.poll() is not None:
                break
            time.sleep(1)
    except FileNotFoundError:
        print("[audio_pipe] ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg")
    finally:
        if _ffmpeg_proc and _ffmpeg_proc.poll() is None:
            _ffmpeg_proc.terminate()


def _transcribe_chunks(stop_event: threading.Event, on_text: callable | None = None):
    """
    Watch for completed .wav chunks, transcribe with Whisper, write to feed file.
    A chunk is 'complete' when the next chunk file appears (ffmpeg has moved on),
    OR when stop_event is set (flushes the final chunk).
    """
    import shutil
    print(f"[audio_pipe] ffmpeg: {shutil.which('ffmpeg') or 'NOT FOUND — install with: sudo apt install ffmpeg'}")

    try:
        import whisper
        cache_dir = getattr(whisper, "_download", None)
        cache_path = (
            cache_dir._get_default_download_root()
            if cache_dir and hasattr(cache_dir, "_get_default_download_root")
            else "~/.cache/whisper"
        )
        print(f"[audio_pipe] Whisper model cache: {cache_path}")
        model = whisper.load_model(WHISPER_MODEL)
        print(f"[audio_pipe] Whisper-{WHISPER_MODEL} loaded OK.")
    except ImportError:
        print("[audio_pipe] ERROR: openai-whisper not installed. Run: pip install openai-whisper")
        return
    except Exception as exc:
        print(f"[audio_pipe] ERROR loading Whisper model: {exc}")
        return

    def _transcribe_one(chunk_path: Path, idx: int):
        try:
            result = model.transcribe(
                str(chunk_path),
                language="en",
                fp16=False,
                verbose=False,
            )
            text = result.get("text", "").strip()
            if text and len(text) > 10:
                payload = json.dumps({
                    "timestamp":   datetime.now(timezone.utc).isoformat(),
                    "chunk_index": idx,
                    "text":        text,
                })
                with open(FEED_FILE, "a", encoding="utf-8") as f:
                    f.write(payload + "\n")
                if on_text:
                    on_text(text)
                print(f"[audio_pipe] Chunk {idx}: {text[:80]}…")
            try:
                chunk_path.unlink()
            except Exception:
                pass
        except Exception as exc:
            print(f"[audio_pipe] Transcription error on chunk {idx}: {exc}")

    processed: set[int] = set()
    chunk_idx = 0
    FEED_FILE.parent.mkdir(parents=True, exist_ok=True)

    while True:
        current    = _chunk_filename(chunk_idx)
        next_chunk = _chunk_filename(chunk_idx + 1)
        stopping   = stop_event.is_set()

        # Process current chunk when next one exists (normal) OR on shutdown (flush last chunk)
        if current.exists() and (next_chunk.exists() or stopping):
            if chunk_idx not in processed:
                _transcribe_one(current, chunk_idx)
                processed.add(chunk_idx)
                chunk_idx += 1
                continue   # immediately check the next chunk without sleeping

        if stopping:
            break
        time.sleep(2)


def start_pipeline(stream_url: str, on_text: callable | None = None):
    """
    Start ffmpeg + Whisper pipeline for the given ECB stream URL.
    Non-blocking: runs ffmpeg and transcription in background threads.
    """
    global _transcribe_thread, _is_running

    if _is_running:
        print("[audio_pipe] Pipeline already running.")
        return

    _stop_event.clear()
    FEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Clear previous feed
    if FEED_FILE.exists():
        FEED_FILE.unlink()

    ffmpeg_thread = threading.Thread(
        target=_run_ffmpeg, args=(stream_url, _stop_event), daemon=True
    )
    ffmpeg_thread.start()

    _transcribe_thread = threading.Thread(
        target=_transcribe_chunks, args=(_stop_event, on_text), daemon=True
    )
    _transcribe_thread.start()

    _is_running = True
    print(f"[audio_pipe] Pipeline started for: {stream_url}")


def stop_pipeline():
    global _is_running
    _stop_event.set()
    if _ffmpeg_proc and _ffmpeg_proc.poll() is None:
        _ffmpeg_proc.terminate()
    _is_running = False
    print("[audio_pipe] Pipeline stopped.")


def is_running() -> bool:
    return _is_running


def simulate_feed(segments: list[str], interval: float = 3.0):
    """
    Inject synthetic text into the live feed for testing without an actual stream.
    Writes one segment every `interval` seconds.
    """
    FEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(segments):
        payload = json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunk_index": i,
            "text": text,
        })
        with open(FEED_FILE, "a", encoding="utf-8") as f:
            f.write(payload + "\n")
        print(f"[simulate_feed] Injected: {text[:60]}...")
        time.sleep(interval)
