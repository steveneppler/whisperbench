#!/usr/bin/env python3
"""
Faster-Whisper benchmark — matches the transcription settings used by the
scanner call-transcription script so results are comparable across systems.

Usage:
    pip install faster-whisper psutil
    python whisper_benchmark.py                       # downloads a sample clip
    python whisper_benchmark.py --audio ./clip.m4a    # use your own audio
    python whisper_benchmark.py --device cuda --compute-type float16
    python whisper_benchmark.py --model large-v3 --runs 5

Reports: model load time, per-run wall time, real-time factor (RTF),
peak RSS, and system info.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Same defaults as the scanner script
DEFAULT_MODEL = "large-v3"
DEFAULT_BEAM_SIZE = 10
DEFAULT_PROMPT = (
    "Transcribe these police, sheriff, fire, and emergency services radio "
    "transmissions in the Grand Junction, Colorado area."
)
VAD_PARAMS = dict(
    min_speech_duration_ms=500,
    threshold=0.5,
    max_speech_duration_s=30,
)

# Short model names -> community MLX conversions (used by --backend mlx)
MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def resolve_mlx_model(name):
    if "/" in name:
        return name
    try:
        return MLX_MODEL_MAP[name]
    except KeyError:
        raise ValueError(
            f"No MLX mapping for model '{name}'. Use one of: "
            f"{', '.join(sorted(MLX_MODEL_MAP))}, or pass a full Hugging Face "
            f"repo path (e.g. mlx-community/whisper-large-v3-mlx)."
        ) from None

# Public domain speech sample (~33s) used when no --audio is given
SAMPLE_URL = "https://github.com/SYSTRAN/faster-whisper/raw/master/tests/data/jfk.flac"
SAMPLE_PATH = Path("bench_sample.flac")


def system_info():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }

    # CPU model / RAM
    try:
        if platform.system() == "Darwin":
            info["processor"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info["ram_gb"] = round(
                int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
                / 1024**3, 1
            )
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["processor"] = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except Exception:
        pass

    # GPU (best effort)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        info["gpu"] = out
    except Exception:
        pass

    try:
        import faster_whisper
        info["faster_whisper"] = faster_whisper.__version__
    except Exception:
        pass

    return info


def peak_rss_gb():
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KB
        divisor = 1024**3 if platform.system() == "Darwin" else 1024**2
        return round(rss / divisor, 2)
    except Exception:
        return None


def fetch_sample():
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH
    print(f"Downloading sample audio -> {SAMPLE_PATH}")
    urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return SAMPLE_PATH


def setup_faster_whisper(args, audio):
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    kwargs = dict(device=args.device, compute_type=args.compute_type)
    if args.cpu_threads:
        kwargs["cpu_threads"] = args.cpu_threads
    model = WhisperModel(args.model, **kwargs)
    load_time = time.perf_counter() - t0

    def transcribe():
        segments, meta = model.transcribe(
            str(audio),
            vad_filter=not args.no_vad,
            vad_parameters=None if args.no_vad else dict(VAD_PARAMS),
            beam_size=args.beam_size,
            language="en",
            initial_prompt=DEFAULT_PROMPT,
        )
        text = " ".join(s.text for s in segments).strip()  # generator, forces work here
        return text, meta.duration

    desc = f"{args.model} / {args.device} / {args.compute_type}"
    return transcribe, load_time, desc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path, help="Path to audio file (default: download sample)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--compute-type", default="int8",
                   help="int8, int8_float16, float16, float32, etc.")
    p.add_argument("--cpu-threads", type=int, default=0, help="0 = library default")
    p.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--no-vad", action="store_true")
    p.add_argument("--json", type=Path, help="Write results to a JSON file")
    args = p.parse_args()

    audio = args.audio or fetch_sample()
    if not Path(audio).exists():
        sys.exit(f"Audio file not found: {audio}")

    info = system_info()
    print("\n=== System ===")
    for k, v in info.items():
        print(f"{k:20} {v}")

    print("\n=== Loading model ===")
    transcribe, load_time, desc = setup_faster_whisper(args, audio)
    print(f"{desc} loaded in {load_time:.2f}s")

    print("\n=== Warmup ===")
    t0 = time.perf_counter()
    text, duration = transcribe()
    print(f"warmup: {time.perf_counter() - t0:.2f}s  (audio duration {duration:.1f}s)")

    print("\n=== Runs ===")
    times = []
    for i in range(1, args.runs + 1):
        t0 = time.perf_counter()
        text, duration = transcribe()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"run {i}: {dt:6.2f}s   RTF {dt / duration:5.3f}   "
              f"speed {duration / dt:5.2f}x realtime")

    mean = statistics.mean(times)
    results = {
        "system": info,
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "cpu_threads": args.cpu_threads or "default",
        "beam_size": args.beam_size,
        "vad": not args.no_vad,
        "audio": str(audio),
        "audio_duration_s": round(duration, 2),
        "load_time_s": round(load_time, 2),
        "runs_s": [round(t, 3) for t in times],
        "mean_s": round(mean, 3),
        "median_s": round(statistics.median(times), 3),
        "stdev_s": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "rtf": round(mean / duration, 4),
        "speed_x_realtime": round(duration / mean, 2),
        "peak_rss_gb": peak_rss_gb(),
        "transcript": text,
    }

    print("\n=== Summary ===")
    print(f"mean      {results['mean_s']}s  (median {results['median_s']}s, "
          f"stdev {results['stdev_s']}s)")
    print(f"RTF       {results['rtf']}  ->  {results['speed_x_realtime']}x realtime")
    print(f"peak RSS  {results['peak_rss_gb']} GB")
    print(f"\ntranscript: {text[:200]}{'...' if len(text) > 200 else ''}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
