# whisperbench

A small benchmark script for [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
that mirrors the transcription settings used by our scanner call-transcription
script, so results can be compared apples-to-apples across different machines
(Apple Silicon, x86 CPU, GPU boxes, etc.).
On Apple Silicon it can also benchmark [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Metal GPU) for comparison.

It reports model load time, per-run wall time, real-time factor (RTF), peak
RSS, and system info (CPU/GPU/RAM), and can optionally dump everything to
JSON for diffing between systems.

## Setup

Requires Python 3.9+. Since installing packages into your system Python
usually isn't possible (or advisable) without admin/root access, set up a
virtual environment instead:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install faster-whisper psutil
```

On Apple Silicon, additionally `pip install mlx-whisper` if you want `--backend mlx`.

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install faster-whisper psutil
```

`psutil` isn't strictly required by the script itself but is commonly needed
by `faster-whisper`/`ctranslate2` on some platforms, so it's installed
alongside it.

Once the venv is active, `python`/`pip` point at the venv's copies, so the
`python whisper_benchmark.py ...` commands below work as-is. When you're
done, run `deactivate` to leave the venv. You'll need to re-run the
`activate` step (not the install step) each time you open a new shell.

## Usage

```bash
# Downloads a ~33s public-domain sample clip (JFK) and benchmarks it
python whisper_benchmark.py

# Benchmark against a real scanner clip instead of the sample.
# Use the *same* file across machines so results are comparable.
python whisper_benchmark.py --audio ./some_call.m4a

# GPU box
python whisper_benchmark.py --device cuda --compute-type float16

# Larger model, more runs
python whisper_benchmark.py --model large-v3 --runs 5

# Write results to JSON for later comparison
python whisper_benchmark.py --json results.json
```

### Apple Silicon (mlx-whisper)

On Apple Silicon Macs you can also benchmark [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper),
which runs on the GPU via Metal (note: the GPU, not the Neural Engine):

```bash
pip install mlx-whisper
python whisper_benchmark.py --backend mlx
```

mlx-whisper's audio loading requires `ffmpeg` (e.g. `brew install ffmpeg`), which is not bundled.

Short model names are mapped to the `mlx-community` conversions on Hugging
Face (e.g. `large-v3` → `mlx-community/whisper-large-v3-mlx`); pass a full
repo path to use a different conversion.

**Comparability caveat:** mlx-whisper always uses greedy decoding (no beam
search) and has no VAD filter, so `--beam-size`, `--no-vad`, `--device`,
`--compute-type`, and `--cpu-threads` are ignored (with a warning). Language
and the initial prompt match the faster-whisper settings, but mlx numbers are
not settings-identical to the faster-whisper baseline — the JSON records
`"backend"`, `"beam_size": null`, `"vad": false` so runs can't be confused.

### Key flags

| Flag | Description |
| --- | --- |
| `--backend {faster-whisper,mlx}` | Inference engine. Default `faster-whisper`. `mlx` runs mlx-whisper on the Apple GPU (Metal); Apple Silicon only. |
| `--audio PATH` | Path to audio file. Defaults to downloading a public-domain ~33s JFK speech sample. Use the same file across machines for a representative comparison. |
| `--device {cpu,cuda,auto}` | Inference device. Default `cpu`. |
| `--compute-type` | CTranslate2 compute type, e.g. `int8`, `int8_float16`, `float16`, `float32`. Default `int8`. |
| `--cpu-threads N` | Pin CPU thread count. CTranslate2's default thread count can differ across systems, so set this explicitly for a fair comparison. `0` (default) uses the library default. |
| `--model NAME` | Whisper model size/name, e.g. `large-v3`, `medium`, `small`. Default `large-v3` (matches the scanner script). |
| `--beam-size N` | Beam search width. Default `10` (matches the scanner script). |
| `--runs N` | Number of timed runs after warmup. Default `3`. |
| `--no-vad` | Disable voice activity detection filtering. |
| `--json PATH` | Write full results (including system info) to a JSON file. |

## Output

The script prints:

- System info (OS, CPU, RAM, GPU if available, `faster-whisper` version)
- Model load time
- A warmup run (excluded from timing stats, since first-run overhead
  distorts steady-state numbers)
- Per-run wall time, RTF, and x-realtime speed
- Summary: mean/median/stdev, RTF, x-realtime speed, peak RSS, and a
  transcript preview

When `--json` is passed, the same data is written as JSON so you can diff
results between systems (e.g. Apple Silicon `cpu`/`int8` vs. a GPU box with
`cuda`/`float16`).
