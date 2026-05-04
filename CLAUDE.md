# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRIBE v2 is a deep multimodal brain encoding model (by Meta) that predicts fMRI brain responses to video, audio, and text. This fork adds an optimized single-PC inference pipeline, a WebGL brain viewer, and a Flask web app.

## Commands

```bash
# Install (Windows)
python -m venv .venv
.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -e ".[plotting]"
.venv\Scripts\pip install decord flask gradio

# Install (WSL/Linux)
uv venv .venv-wsl && source .venv-wsl/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -e ".[plotting]" decord flask gradio

# Run Flask web app
.venv\Scripts\python server.py          # Open http://localhost:5000

# Run Gradio app (legacy)
.venv\Scripts\python app.py

# Benchmark extractors
.venv\Scripts\python test_extractors.py video.mp4 --video-only
.venv\Scripts\python test_extractors.py video.mp4 --audio-only
.venv\Scripts\python test_extractors.py --text-only
.venv\Scripts\python test_extractors.py video.mp4 --model-only

# Run tests
.venv\Scripts\python -m pytest
```

## Architecture

### Two Inference Pipelines

**Standard pipeline** (`TribeModel.predict()` via `demo_utils.py`):
- Uses `neuralset` library extractors with per-batch GPU/CPU sync points
- Features cached to disk via `exca` infrastructure
- Video chunked at 60s, audio chunked at 60s for whisperx
- Patches applied to `neuralset` site-packages for Windows compatibility and performance (NVDEC, decord batch extraction, pinned memory, GPU torchvision transforms)

**Fast pipeline** (`FastTribePipeline` in `tribev2/fast/`):
- Pure-PyTorch extractors with zero per-batch sync points
- Brain model offloaded to CPU during extraction, moved back for inference
- Text/audio/video extractors run sequentially, each freed after use
- `torch.compile` auto-enabled on Linux, disabled on Windows (triton unavailable)
- Audio auto-chunked at 30s internally to prevent VRAM exhaustion
- Video batch_size forced to 1 (GPU-saturated, batching is slower)
- Features cached as `.npy` files; `group_mean` layer aggregation in extractors, `cat` in brain model

### Brain Model

`FmriEncoderModel` in `model.py`: Takes multimodal features `(B, L, D, T)` per modality, applies layer aggregation (`cat`), projects through per-modality linear layers, combines via `cat`/`stack`/`sum`, runs through a Transformer encoder, outputs `(B, n_vertices, n_output_timesteps)` predictions on the fsaverage5 cortical surface (20,484 vertices).

The checkpoint expects `feature_dims = {'text': (2, 3072), 'audio': (2, 1024), 'video': (2, 1408)}` — 2 layers per modality after `group_mean` aggregation from `layers_to_use=[0.5, 0.75, 1.0]`.

### Feature Extractors

| Extractor | Model | Input | Output | VRAM |
|-----------|-------|-------|--------|------|
| Text | Llama-3.2-3B (`unsloth/Llama-3.2-3B`) | Word events with context | `(2, 3072)` per word | ~6.5 GB |
| Audio | Wav2Vec-BERT 2.0 (`facebook/w2v-bert-2.0`) | Audio waveform | `(2, 1024, T)` temporal | ~2.7 GB |
| Video | VJEPA2-ViT-G (`facebook/vjepa2-vitg-fpc64-256`) | 64 frames per 4s clip | `(2, 1408, T)` temporal | ~5.6 GB |

### WebGL Viewer (`tribev2/viewer.py`)

Generates self-contained HTML with three.js brain renderer + Plotly timeline. Uses fsaverage5 mesh data packed as binary (header + coords + faces + sulcal_depth + predictions). The Flask app at `server.py` serves the viewer at `/viewer/<run_id>`.

Key template tokens: `__DATA_B64__`, `__N_VERTS__`, `__N_FACES__`, `__N_TIMESTEPS__`, `__TIMES__`, `__ZONES__`, `__VIDEO_B64__`, `__HAS_VIDEO__`, `__HEIGHT__`.

Meta's GLB files in `models/` are the same vertex count as fsaverage5 (10,242 per hemisphere) — no baked AO. High-res variants (`*-high*.glb`) are 163,842 verts with `*-upsample.bin` mapping files.

## Windows Compatibility

Multiple site-packages are patched at runtime for Windows:
- `neuralset/extractors/video.py`: NVDEC/decord frame extraction, CUDA stream prefetch, GPU transforms
- `neuralset/events/study.py`: Study re-registration guard (for Streamlit/Flask reloads)
- `exca/cachedict/inflight.py`: `os.kill` liveness check fix
- `tribev2/demo_utils.py`: PosixPath monkey-patch for YAML config loading, HF repo ID backslash fix

These patches are in the Windows `.venv` site-packages, not in the repo. They need to be reapplied after reinstalling packages.

## Key Constraints

- `num_workers=0` required on Windows (DataLoader multiprocessing deadlocks in notebooks/Flask)
- Video chunking should be skipped for the fast pipeline (no `ChunkEvents` for Video, only Audio for whisperx)
- The brain model's `layer_aggregation="cat"` expects extractors to output `group_mean`-aggregated features (2 layers), not raw selected layers (3)
- `torch.compile` needs triton which is only available on Linux; `ENABLE_COMPILE` auto-detects via `os.name`
- FP8 via torchao requires torch >= 2.11 and only works on Linux; SM89 (RTX 4080) supports TensorWise FP8 but not AxisWise
- Default text model is `unsloth/Llama-3.2-3B` (ungated mirror of Meta's gated `meta-llama/Llama-3.2-3B`)
