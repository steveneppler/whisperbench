# Design: mlx-whisper backend for whisper_benchmark.py

**Date:** 2026-07-17
**Status:** Approved

## Goal

Add an Apple Silicon–accelerated benchmark path to `whisper_benchmark.py` using
mlx-whisper (Metal GPU), so Mac results can be compared against the existing
faster-whisper CPU baseline. One backend per invocation; comparison happens via
the `--json` output.

Note: mlx-whisper uses the Apple GPU via Metal, not the Neural Engine. A true
ANE benchmark (WhisperKit) is out of scope for this change.

## Interface

- New flag: `--backend {faster-whisper,mlx}`, default `faster-whisper`.
  Existing invocations behave exactly as before.
- All shared options work identically for both backends: `--model`, `--audio`,
  `--runs`, `--json`, warmup run, timed runs, summary output.

## mlx backend behavior

- Imports `mlx_whisper` only when selected (faster-whisper likewise only
  imported on its path).
- Model name mapping: short names map to community MLX conversions:
  - `tiny` → `mlx-community/whisper-tiny-mlx`
  - `base` → `mlx-community/whisper-base-mlx`
  - `small` → `mlx-community/whisper-small-mlx`
  - `medium` → `mlx-community/whisper-medium-mlx`
  - `large-v3` → `mlx-community/whisper-large-v3-mlx`
  - `large-v3-turbo` → `mlx-community/whisper-large-v3-turbo`
  - Any name containing `/` is passed through as a literal Hugging Face repo.
  - Any other name is an error listing the supported short names.
- Load time is measured separately by preloading the model before warmup
  (via mlx-whisper's model loader), matching the faster-whisper flow where
  load time and transcription time are reported independently.
- Transcription uses the same `language="en"` and `initial_prompt` as the
  faster-whisper path.
- Audio duration is computed from the decoded audio length
  (samples / 16000), since mlx-whisper returns no duration field.

## Comparability caveats (explicit, by design)

mlx-whisper has no VAD filter and no beam search (greedy decoding only).
Therefore on the mlx path:

- JSON records `backend`, `beam_size: null`, `vad: false`.
- The console output prints a one-line caveat stating the mlx run uses greedy
  decoding and no VAD, so numbers are not settings-identical to the
  faster-whisper baseline.
- The transcript is included in output (as today) for eyeball quality checks.

## Flag handling on the mlx path

- `--device`, `--compute-type`, `--cpu-threads`: ignored with a printed
  warning (not an error).
- `--beam-size`, `--no-vad`: ignored with a printed warning (greedy/no-VAD is
  always the case for mlx).

## Error handling

- `mlx_whisper` not installed → clear message: `pip install mlx-whisper`,
  exit non-zero, no traceback.
- Non-Apple-Silicon machine (import or runtime failure) → clear platform
  message, exit non-zero.

## README

Add an "Apple Silicon (mlx-whisper)" section: install command, example
invocation, and the comparability caveat paragraph.

## Testing

Functional check on an Apple Silicon Mac with `--model tiny --runs 1` for both
backends: metrics print, JSON file is well-formed, mlx caveat line appears.
Large-v3 downloads (~3 GB) are avoided in testing.
