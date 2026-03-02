#!/usr/bin/env python3
"""Run STT benchmark: transcribe audio files and compute CER.

Usage:
    python benchmarks/run_benchmark.py                  # Run with current config
    python benchmarks/run_benchmark.py --sweep           # Compare parameter combinations
    python benchmarks/run_benchmark.py --tag product_name  # Filter by tag
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import cer

# Add project root to path so we can import vox modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vox.config import AppConfig, load_config  # noqa: E402

BENCHMARKS_DIR = Path(__file__).parent


def _register_cuda_dll_dirs() -> None:
    """Add NVIDIA CUDA DLL directories installed via pip to the DLL search path."""
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site_packages.exists():
        return
    for bin_dir in site_packages.glob("*/bin"):
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
MANIFEST_PATH = BENCHMARKS_DIR / "manifest.json"
RESULTS_DIR = BENCHMARKS_DIR / "results"


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


# Full-width digit → half-width mapping
_FW_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def normalize_text(text: str) -> str:
    """Normalize text for fair CER comparison.

    - Strip leading/trailing whitespace
    - Remove trailing punctuation (。、．.!！?？)
    - Normalize full-width digits to half-width
    - Collapse multiple spaces to one
    """
    text = text.strip()
    text = text.rstrip("。、．.!！?？")
    text = text.translate(_FW_DIGITS)
    text = re.sub(r"\s+", " ", text)
    return text


def apply_word_replacements(text: str, replacements: dict[str, str]) -> str:
    """Apply word replacements (same logic as VoxApp, including regex support)."""
    for old, new in replacements.items():
        if old.startswith("/") and old.endswith("/") and len(old) > 2:
            pattern = old[1:-1]
            text = re.sub(pattern, new, text)
        else:
            text = text.replace(old, new)
    return text


def transcribe_entry(
    engine: object,
    entry: dict,
    sample_rate: int,
    word_replacements: dict[str, str],
) -> dict:
    """Transcribe a single audio file and compute CER."""
    audio_path = BENCHMARKS_DIR / entry["audio"]
    if not audio_path.exists():
        return {
            "id": entry["id"],
            "expected": entry["expected"],
            "actual": "",
            "cer": 1.0,
            "exact_match": False,
            "error": f"Audio file not found: {audio_path}",
            "time_sec": 0.0,
        }

    # Load audio
    audio, file_sr = sf.read(str(audio_path), dtype="float32")
    if file_sr != sample_rate:
        # Simple resampling via numpy (good enough for benchmarks)
        import scipy.signal

        audio = scipy.signal.resample(
            audio, int(len(audio) * sample_rate / file_sr)
        )
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono

    # Transcribe
    t0 = time.monotonic()
    raw_text = engine.transcribe(audio, sample_rate)
    elapsed = time.monotonic() - t0

    # Apply word replacements
    actual = apply_word_replacements(raw_text, word_replacements)

    # Compute CER (character-level for Japanese) with normalization
    expected = entry["expected"]
    norm_expected = normalize_text(expected)
    norm_actual = normalize_text(actual)
    if norm_expected and norm_actual:
        error_rate = cer(norm_expected, norm_actual)
    else:
        error_rate = 0.0 if norm_expected == norm_actual else 1.0

    return {
        "id": entry["id"],
        "expected": expected,
        "actual": actual,
        "raw": raw_text,
        "cer": round(error_rate, 4),
        "exact_match": norm_expected == norm_actual,
        "time_sec": round(elapsed, 2),
    }


def create_engine(config: AppConfig):
    """Create and load an STT engine from config."""
    from vox.stt import create_stt_engine

    engine = create_stt_engine(config.stt)
    engine.load_model()
    return engine


def run_benchmark(
    config: AppConfig,
    entries: list[dict],
    label: str = "default",
) -> dict:
    """Run benchmark on all entries and return results."""
    print(f"\n--- Running: {label} ---")
    engine = create_engine(config)

    results = []
    for entry in entries:
        result = transcribe_entry(
            engine,
            entry,
            config.audio.sample_rate,
            config.stt.word_replacements,
        )
        status = "OK" if result["exact_match"] else "MISS"
        print(f"  [{status}] {entry['id']}: CER={result['cer']:.3f} ({result['time_sec']:.1f}s)")
        if not result["exact_match"]:
            print(f"         expected: {result['expected']}")
            print(f"         actual:   {result['actual']}")
        results.append(result)

    # Aggregate
    cer_values = [r["cer"] for r in results]
    exact_matches = sum(1 for r in results if r["exact_match"])
    total_time = sum(r["time_sec"] for r in results)

    summary = {
        "label": label,
        "avg_cer": round(np.mean(cer_values), 4) if cer_values else 0.0,
        "exact_match_rate": round(exact_matches / len(results), 2) if results else 0.0,
        "exact_matches": exact_matches,
        "total": len(results),
        "total_time_sec": round(total_time, 1),
        "results": results,
    }
    return summary


def print_comparison_table(summaries: list[dict]) -> None:
    """Print a comparison table of multiple benchmark runs."""
    print("\n=== Benchmark Results ===")
    print(f"{'Config':<25} | {'Avg CER':>8} | {'Exact Match':>12} | {'Time':>8}")
    print("-" * 65)
    for s in summaries:
        match_str = f"{s['exact_matches']}/{s['total']} ({s['exact_match_rate']:.0%})"
        time_str = f"{s['total_time_sec']:>6.1f}s"
        print(f"{s['label']:<25} | {s['avg_cer']:>8.4f} | {match_str:>12} | {time_str}")
    print()


def save_results(summaries: list[dict], tag: str = "") -> Path:
    """Save results to JSON file in results/ directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = RESULTS_DIR / f"benchmark_{timestamp}{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {path}")
    return path


def get_sweep_configs(base_config: AppConfig) -> list[tuple[str, AppConfig]]:
    """Generate parameter sweep configurations."""
    configs: list[tuple[str, AppConfig]] = []

    # Baseline
    configs.append(("baseline", base_config))

    # beam_size variations
    for beam in [8, 10]:
        fw = base_config.stt.faster_whisper.model_copy(update={"beam_size": beam})
        stt = base_config.stt.model_copy(update={"faster_whisper": fw})
        cfg = base_config.model_copy(update={"stt": stt})
        configs.append((f"beam_{beam}", cfg))

    # compute_type=float32
    fw = base_config.stt.faster_whisper.model_copy(update={"compute_type": "float32"})
    stt = base_config.stt.model_copy(update={"faster_whisper": fw})
    cfg = base_config.model_copy(update={"stt": stt})
    configs.append(("float32", cfg))

    # model=large-v3 (full model)
    fw = base_config.stt.faster_whisper.model_copy(update={"model": "large-v3"})
    stt = base_config.stt.model_copy(update={"faster_whisper": fw})
    cfg = base_config.model_copy(update={"stt": stt})
    configs.append(("large-v3", cfg))

    return configs


def main() -> None:
    _register_cuda_dll_dirs()

    parser = argparse.ArgumentParser(description="Run Vox STT benchmark")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--tag", type=str, default="", help="Filter entries by tag")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml (default: project root)"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else PROJECT_ROOT / "config.yaml"
    config = load_config(config_path)

    # Load manifest
    manifest = load_manifest()
    entries = manifest["entries"]

    # Filter by tag
    if args.tag:
        entries = [e for e in entries if args.tag in e.get("tags", [])]
        print(f"Filtered to {len(entries)} entries with tag '{args.tag}'")

    if not entries:
        print("No entries found in manifest. Record some audio first:")
        print("  python benchmarks/record_benchmark.py")
        return

    # Check for missing audio files
    missing = [e for e in entries if not (BENCHMARKS_DIR / e["audio"]).exists()]
    if missing:
        print(f"Warning: {len(missing)} audio file(s) missing:")
        for e in missing:
            print(f"  - {e['audio']} (id={e['id']}): \"{e['expected']}\"")
        print("Record them with: python benchmarks/record_benchmark.py\n")
        # Filter to available entries only
        entries = [e for e in entries if (BENCHMARKS_DIR / e["audio"]).exists()]
        if not entries:
            print("No audio files available. Exiting.")
            return

    if args.sweep:
        # Parameter sweep mode
        sweep_configs = get_sweep_configs(config)
        summaries = []
        for label, cfg in sweep_configs:
            summary = run_benchmark(cfg, entries, label=label)
            summaries.append(summary)
        print_comparison_table(summaries)
        save_results(summaries, tag="sweep")
    else:
        # Single run
        summary = run_benchmark(config, entries, label="current_config")
        print_comparison_table([summary])
        save_results([summary])


if __name__ == "__main__":
    main()
