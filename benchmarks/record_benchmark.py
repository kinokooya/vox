#!/usr/bin/env python3
"""Interactive recording helper for STT benchmark audio files.

Usage:
    python benchmarks/record_benchmark.py           # Record missing entries from manifest
    python benchmarks/record_benchmark.py --add      # Add new entries interactively

Workflow (default mode):
    1. Shows unrecorded entries from manifest.json
    2. Press Enter to start recording each entry
    3. Press Enter again to stop recording
    4. WAV file is saved to audio/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

BENCHMARKS_DIR = Path(__file__).parent
AUDIO_DIR = BENCHMARKS_DIR / "audio"
MANIFEST_PATH = BENCHMARKS_DIR / "manifest.json"

SAMPLE_RATE = 16000
CHANNELS = 1


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "description": "Vox STT benchmark manifest",
        "entries": [],
    }


def save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  manifest updated: {MANIFEST_PATH}")


def record_audio() -> np.ndarray:
    """Record audio until Enter is pressed. Returns float32 numpy array."""
    frames: list[np.ndarray] = []

    def callback(indata: np.ndarray, frame_count: int, time_info: object, status: object) -> None:
        if status:
            print(f"  [warning] {status}", file=sys.stderr)
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )

    with stream:
        input("  >>> Press Enter to STOP recording...")

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()
    return audio


def record_entry(entry: dict) -> bool:
    """Record audio for a single manifest entry. Returns True if successful."""
    audio_path = AUDIO_DIR / Path(entry["audio"]).name

    print(f"\n  [{entry['id']}] {entry['expected']}")
    input("  >>> Press Enter to START recording...")
    print("  Recording... (press Enter to stop)")

    audio = record_audio()

    if len(audio) == 0:
        print("  No audio captured, skipping.")
        return False

    duration = len(audio) / SAMPLE_RATE
    print(f"  Recorded {duration:.1f}s ({len(audio)} samples)")

    sf.write(str(audio_path), audio, SAMPLE_RATE)
    print(f"  Saved: {audio_path}")
    return True


def mode_record_missing() -> None:
    """Record audio for entries in manifest that don't have audio files yet."""
    manifest = load_manifest()
    entries = manifest["entries"]

    missing = [e for e in entries if not (BENCHMARKS_DIR / e["audio"]).exists()]

    if not missing:
        print("All entries already have audio files!")
        print(f"  Total: {len(entries)} entries")
        return

    print(f"Found {len(missing)} entries without audio (of {len(entries)} total).")
    print("Type 's' to skip an entry, 'q' to quit.\n")

    recorded = 0
    for i, entry in enumerate(missing):
        remaining = len(missing) - i
        print(f"--- [{remaining} remaining] ---")

        prompt = f"  [{entry['id']}] \"{entry['expected']}\"\n"
        prompt += "  Record? (Enter=yes, s=skip, q=quit): "
        choice = input(prompt).strip().lower()
        if choice == "q":
            break
        if choice == "s":
            print("  Skipped.")
            continue

        if record_entry(entry):
            recorded += 1

    print(f"\nDone. Recorded {recorded} entries.")


def mode_add_new() -> None:
    """Add new entries interactively (original mode)."""
    manifest = load_manifest()

    print("=== Add New Benchmark Entries ===")
    print("  Type 'q' to quit.\n")

    while True:
        entry_id = input("Entry ID (e.g. X01, or 'q' to quit): ").strip()
        if entry_id.lower() == "q":
            break
        if not entry_id:
            continue

        existing_ids = {e["id"] for e in manifest["entries"]}
        if entry_id in existing_ids:
            print(f"  ID '{entry_id}' already exists, skipping.")
            continue

        expected = input("Expected text: ").strip()
        if not expected:
            print("  Empty text, skipping.\n")
            continue

        tags_input = input("Tags (comma-separated): ").strip()
        tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []

        filename = f"{entry_id}.wav"
        entry = {
            "id": entry_id,
            "audio": f"audio/{filename}",
            "expected": expected,
            "tags": tags,
        }

        input(f"  >>> Press Enter to START recording (id={entry_id})...")
        print("  Recording... (press Enter to stop)")

        audio = record_audio()

        if len(audio) == 0:
            print("  No audio captured, skipping.\n")
            continue

        duration = len(audio) / SAMPLE_RATE
        print(f"  Recorded {duration:.1f}s ({len(audio)} samples)")

        audio_path = AUDIO_DIR / filename
        sf.write(str(audio_path), audio, SAMPLE_RATE)
        print(f"  Saved: {audio_path}")

        manifest["entries"].append(entry)
        save_manifest(manifest)
        print()

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record benchmark audio files")
    parser.add_argument(
        "--add", action="store_true", help="Add new entries interactively"
    )
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Vox STT Benchmark Recorder ===")
    print(f"  Audio dir: {AUDIO_DIR}")
    print(f"  Format: {SAMPLE_RATE}Hz, mono, float32\n")

    if args.add:
        mode_add_new()
    else:
        mode_record_missing()


if __name__ == "__main__":
    main()
