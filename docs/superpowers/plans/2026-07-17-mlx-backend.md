# mlx-whisper Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--backend mlx` option to `whisper_benchmark.py` that benchmarks mlx-whisper (Apple Silicon Metal GPU) with the same harness and metrics as the existing faster-whisper CPU baseline.

**Architecture:** The script keeps one shared harness (warmup, timed runs, stats, JSON) and gains per-backend setup functions that each return a `transcribe()` closure, a load time, and a description string. `transcribe()` returns `(text, audio_duration_s)` so the harness no longer depends on faster-whisper's `meta.duration`. Model-name mapping to MLX community repos is a pure function with unit tests.

**Tech Stack:** Python 3.9+, faster-whisper, mlx-whisper (Apple Silicon only), pytest for the one pure-function test.

**Spec:** `docs/superpowers/specs/2026-07-17-mlx-backend-design.md`

## Global Constraints

- Default behavior must be unchanged: `--backend` defaults to `faster-whisper` and all existing invocations produce the same output as before.
- On the mlx path, JSON must record `backend`, `beam_size: null`, `vad: false`, and the console must print the greedy/no-VAD caveat line.
- `--device`, `--compute-type`, `--cpu-threads`, `--beam-size`, `--no-vad` on the mlx path: warn and continue, never error.
- Missing `mlx_whisper` or non-Apple-Silicon machine: clear message via `sys.exit`, no traceback.
- Testing uses `--model tiny --runs 1` only (never download large-v3, ~3 GB).
- The repo venv lives at `.venv/`; run everything via `.venv/bin/python` / `.venv/bin/pip` (no activation needed).

---

### Task 1: Environment setup + `resolve_mlx_model` with tests

**Files:**
- Create: `.venv/` (git-ignored tooling, not committed)
- Create: `test_benchmark.py`
- Modify: `whisper_benchmark.py` (add `MLX_MODEL_MAP` and `resolve_mlx_model` after the `VAD_PARAMS` block, around line 39)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `resolve_mlx_model(name: str) -> str` — maps short model names to MLX community Hugging Face repos, passes through any name containing `/`, raises `ValueError` for unknown short names. Task 3 calls this.

- [ ] **Step 1: Create the venv and install dependencies**

```bash
cd /Users/seppler/dev/whisperbench
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install faster-whisper psutil mlx-whisper pytest
```

Expected: installs succeed (mlx-whisper only resolves on Apple Silicon macOS — this machine qualifies). If `.gitignore` does not exist or lacks `.venv/`, add a `.gitignore` with a `.venv/` line and include it in this task's commit.

- [ ] **Step 2: Write the failing test**

Create `test_benchmark.py`:

```python
import pytest

from whisper_benchmark import resolve_mlx_model


def test_short_names_map_to_mlx_community_repos():
    assert resolve_mlx_model("tiny") == "mlx-community/whisper-tiny-mlx"
    assert resolve_mlx_model("base") == "mlx-community/whisper-base-mlx"
    assert resolve_mlx_model("small") == "mlx-community/whisper-small-mlx"
    assert resolve_mlx_model("medium") == "mlx-community/whisper-medium-mlx"
    assert resolve_mlx_model("large-v3") == "mlx-community/whisper-large-v3-mlx"
    assert resolve_mlx_model("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"


def test_full_repo_path_passes_through():
    assert resolve_mlx_model("someuser/custom-whisper") == "someuser/custom-whisper"


def test_unknown_short_name_raises_with_name_in_message():
    with pytest.raises(ValueError, match="large-v2"):
        resolve_mlx_model("large-v2")
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest test_benchmark.py -v`
Expected: FAIL at import with `ImportError: cannot import name 'resolve_mlx_model'`

- [ ] **Step 4: Implement the mapping**

In `whisper_benchmark.py`, directly after the `VAD_PARAMS = dict(...)` block (line 39), add:

```python
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest test_benchmark.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add test_benchmark.py whisper_benchmark.py .gitignore
git commit -m "Add MLX model-name mapping with tests"
```

(Omit `.gitignore` from the `git add` if it already covered `.venv/` and was not modified.)

---

### Task 2: Refactor faster-whisper path into a backend setup function

**Files:**
- Modify: `whisper_benchmark.py:140-164` (model loading block and `transcribe` closure inside `main()`)
- Test: `test_benchmark.py` (existing tests must still pass; no new tests — this is a behavior-preserving refactor verified by running the script)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `setup_faster_whisper(args, audio) -> (transcribe, load_time, desc)` where `transcribe() -> (text: str, duration: float)`, `load_time: float` (seconds), `desc: str` (e.g. `"large-v3 / cpu / int8"`). Task 3's `setup_mlx` must return the same shape (with `load_time` possibly `None`).

- [ ] **Step 1: Add the setup function**

In `whisper_benchmark.py`, add above `main()`:

```python
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
```

- [ ] **Step 2: Rewire `main()` to use it**

In `main()`, replace everything from `from faster_whisper import WhisperModel` (line 129) through the end of the old `transcribe()` closure (line 159) — keeping the `audio = args.audio or fetch_sample()` block and the system-info printing where they are — so the middle of `main()` reads:

```python
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
```

Then in the `results = {...}` dict, replace the two `meta.duration` uses:

```python
        "audio_duration_s": round(duration, 2),
```
```python
        "rtf": round(mean / duration, 4),
        "speed_x_realtime": round(duration / mean, 2),
```

(The `audio = args.audio or fetch_sample()` and file-existence check move to just after `args = p.parse_args()`, before `info = system_info()`, since `setup_faster_whisper` needs `audio`. The `from faster_whisper import WhisperModel` line at the top of `main()` is deleted — the import now lives inside `setup_faster_whisper`.)

- [ ] **Step 3: Verify tests still pass and the script still works**

Run: `.venv/bin/python -m pytest test_benchmark.py -v`
Expected: 3 passed

Run: `.venv/bin/python whisper_benchmark.py --model tiny --runs 1`
Expected: downloads `bench_sample.flac` (first run) and the tiny model, then prints the System / Loading model / Warmup / Runs / Summary sections exactly as before, with a plausible JFK transcript preview. Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add whisper_benchmark.py
git commit -m "Refactor model setup into backend function"
```

---

### Task 3: Add the mlx backend

**Files:**
- Modify: `whisper_benchmark.py` (new `--backend` flag, new `setup_mlx` function, backend dispatch in `main()`, backend-aware `results` dict, docstring usage line)

**Interfaces:**
- Consumes: `resolve_mlx_model` (Task 1); returns the same `(transcribe, load_time, desc)` shape as `setup_faster_whisper` (Task 2), except `load_time` may be `None`.
- Produces: `setup_mlx(args, audio) -> (transcribe, load_time, desc)`; `--backend {faster-whisper,mlx}` CLI flag; `backend` key in JSON results.

- [ ] **Step 1: Add the `--backend` flag and docstring line**

In `main()`'s argparse block, add as the first argument:

```python
    p.add_argument("--backend", default="faster-whisper",
                   choices=["faster-whisper", "mlx"],
                   help="Inference engine. 'mlx' = mlx-whisper on Apple Silicon "
                        "(Metal GPU); greedy decoding, no VAD.")
```

In the module docstring's Usage section, add:

```
    python whisper_benchmark.py --backend mlx            # Apple Silicon GPU (Metal)
```

- [ ] **Step 2: Add `setup_mlx`**

Add below `setup_faster_whisper`:

```python
def setup_mlx(args, audio):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.exit("--backend mlx requires an Apple Silicon Mac (mlx runs on Metal).")
    try:
        import mlx_whisper
    except ImportError:
        sys.exit("mlx-whisper is not installed. Run: pip install mlx-whisper")

    ignored = [
        ("--device", args.device != "cpu"),
        ("--compute-type", args.compute_type != "int8"),
        ("--cpu-threads", args.cpu_threads != 0),
        ("--beam-size", args.beam_size != DEFAULT_BEAM_SIZE),
        ("--no-vad", args.no_vad),
    ]
    for flag, is_set in ignored:
        if is_set:
            print(f"warning: {flag} is ignored with --backend mlx")

    try:
        repo = resolve_mlx_model(args.model)
    except ValueError as e:
        sys.exit(str(e))

    from mlx_whisper.audio import SAMPLE_RATE, load_audio
    duration = len(load_audio(str(audio))) / SAMPLE_RATE

    t0 = time.perf_counter()
    try:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        ModelHolder.get_model(repo, mx.float16)
        load_time = time.perf_counter() - t0
    except Exception:
        # Preloading uses mlx-whisper internals; if they change, fall back to
        # counting the load inside the warmup run.
        load_time = None
        print("note: could not preload mlx model; load time will be included "
              "in the warmup run")

    def transcribe():
        result = mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=repo,
            language="en",
            initial_prompt=DEFAULT_PROMPT,
        )
        return result["text"].strip(), duration

    print("note: mlx backend uses greedy decoding (no beam search) and no VAD "
          "-- not settings-identical to the faster-whisper baseline")
    desc = f"{repo} / mlx (metal gpu) / float16"
    return transcribe, load_time, desc
```

- [ ] **Step 3: Dispatch in `main()` and make the load-time print `None`-safe**

Replace the two lines from Task 2:

```python
    transcribe, load_time, desc = setup_faster_whisper(args, audio)
    print(f"{desc} loaded in {load_time:.2f}s")
```

with:

```python
    if args.backend == "mlx":
        transcribe, load_time, desc = setup_mlx(args, audio)
    else:
        transcribe, load_time, desc = setup_faster_whisper(args, audio)
    if load_time is not None:
        print(f"{desc} loaded in {load_time:.2f}s")
    else:
        print(desc)
```

- [ ] **Step 4: Make the results dict backend-aware**

Replace the top of the `results = {...}` dict (the keys from `"system"` through `"vad"`) with:

```python
    is_mlx = args.backend == "mlx"
    results = {
        "system": info,
        "backend": args.backend,
        "model": args.model,
        "device": "mlx (metal gpu)" if is_mlx else args.device,
        "compute_type": "float16" if is_mlx else args.compute_type,
        "cpu_threads": "n/a" if is_mlx else (args.cpu_threads or "default"),
        "beam_size": None if is_mlx else args.beam_size,
        "vad": False if is_mlx else not args.no_vad,
```

and make the load-time entry `None`-safe:

```python
        "load_time_s": round(load_time, 2) if load_time is not None else None,
```

- [ ] **Step 5: Verify both backends end-to-end**

Run: `.venv/bin/python -m pytest test_benchmark.py -v`
Expected: 3 passed

Run: `.venv/bin/python whisper_benchmark.py --backend mlx --model tiny --runs 1 --json /private/tmp/claude-501/-Users-seppler-dev-whisperbench/65112ebf-0970-450c-8b6d-d36ca1436666/scratchpad/mlx.json`
Expected: downloads the MLX tiny model (first run), prints the greedy/no-VAD caveat line, load time, warmup, one run with RTF, summary, transcript preview. Exit code 0.

Run: `.venv/bin/python -c "import json; d = json.load(open('/private/tmp/claude-501/-Users-seppler-dev-whisperbench/65112ebf-0970-450c-8b6d-d36ca1436666/scratchpad/mlx.json')); assert d['backend'] == 'mlx' and d['beam_size'] is None and d['vad'] is False; print('json ok')"`
Expected: `json ok`

Run: `.venv/bin/python whisper_benchmark.py --backend mlx --model tiny --runs 1 --beam-size 5 2>&1 | head -30`
Expected: output includes `warning: --beam-size is ignored with --backend mlx`

Run: `.venv/bin/python whisper_benchmark.py --model tiny --runs 1 --json /private/tmp/claude-501/-Users-seppler-dev-whisperbench/65112ebf-0970-450c-8b6d-d36ca1436666/scratchpad/fw.json`
Expected: faster-whisper path still works; JSON contains `"backend": "faster-whisper"`, `"beam_size": 10`, `"vad": true`.

- [ ] **Step 6: Commit**

```bash
git add whisper_benchmark.py
git commit -m "Add mlx-whisper backend (--backend mlx)"
```

---

### Task 4: README section for the mlx backend

**Files:**
- Modify: `README.md` (add an "Apple Silicon (mlx-whisper)" section after "Usage", add `--backend` to the Key flags table, mention mlx in the intro paragraph)

**Interfaces:**
- Consumes: CLI flags from Task 3.
- Produces: documentation only.

- [ ] **Step 1: Add the section and flag row**

After the Usage code block (before "### Key flags"), insert:

````markdown
### Apple Silicon (mlx-whisper)

On Apple Silicon Macs you can also benchmark [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper),
which runs on the GPU via Metal (note: the GPU, not the Neural Engine):

```bash
pip install mlx-whisper
python whisper_benchmark.py --backend mlx
```

Short model names are mapped to the `mlx-community` conversions on Hugging
Face (e.g. `large-v3` → `mlx-community/whisper-large-v3-mlx`); pass a full
repo path to use a different conversion.

**Comparability caveat:** mlx-whisper always uses greedy decoding (no beam
search) and has no VAD filter, so `--beam-size`, `--no-vad`, `--device`,
`--compute-type`, and `--cpu-threads` are ignored (with a warning). Language
and the initial prompt match the faster-whisper settings, but mlx numbers are
not settings-identical to the faster-whisper baseline — the JSON records
`"backend"`, `"beam_size": null`, `"vad": false` so runs can't be confused.
````

In the Key flags table, add as the first row:

```markdown
| `--backend {faster-whisper,mlx}` | Inference engine. Default `faster-whisper`. `mlx` runs mlx-whisper on the Apple GPU (Metal); Apple Silicon only. |
```

In the intro paragraph, after the first sentence, add: "On Apple Silicon it can also benchmark [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Metal GPU) for comparison."

Also update the Setup install lines to mention the optional extra: after the `pip install faster-whisper psutil` line in both OS blocks, no change — instead add one sentence below the macOS/Linux block: "On Apple Silicon, additionally `pip install mlx-whisper` if you want `--backend mlx`."

- [ ] **Step 2: Verify the markdown renders sanely**

Run: `.venv/bin/python -c "print(open('README.md').read())" | head -80`
Expected: new section present, table row intact (pipes balanced).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document --backend mlx in README"
```
