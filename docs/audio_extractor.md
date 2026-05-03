# Audio Extractor (Wav2Vec-BERT 2.0)

## Overview
Extracts temporal audio embeddings from raw waveforms. Reads audio (from .wav or video file), resamples to model's expected sample rate, runs the encoder forward, and produces time-aligned feature arrays.

## Model
- **Architecture**: `facebook/w2v-bert-2.0` (Wav2Vec2BertModel)
- **Loaded dtype**: bf16 on Ampere+ (SM>=80), fp16 on older CUDA, fp32 on CPU
- **VRAM for weights**: ~2.3 GB (bf16)

## Input
| Field | Source | Description |
|-------|--------|-------------|
| `event.filepath` | Audio/Video event | Path to .wav/.mp3 or .mp4 file |
| `event.type` | `"Audio"` or `"Video"` | Determines load method |
| `event.start` | Event metadata | Start time in source (seconds) |
| `event.duration` | Event metadata | Duration to extract (seconds) |

## Audio Loading
1. **Audio files**: `soundfile.read()` → numpy `(n_samples, n_channels)` float32
2. **Video files**: `moviepy.VideoFileClip` → extract audio track → `to_soundarray()`
3. **Mono conversion**: mean across channels if stereo
4. **Windowing**: slice `wav[start_sample:end_sample]` based on event start/duration
5. **Resampling**: `julius.resample_frac()` (fallback: `torchaudio.functional.resample`) to model's expected rate

## VRAM During Inference
| Tensor | Shape | Notes |
|--------|-------|-------|
| Model weights | ~2.3 GB | Persistent until `free()` |
| Feature tensor | `(1, T_audio, D_feat)` | From HF feature extractor, async H2D |
| Hidden states | `(L, 1, T_audio, D)` | All layers from model output |
| Interpolated output | `(L_out, D, T_event)` | After layer agg + time interpolation |

Where: T_audio=audio frames at model rate, D=1024, L=model layers, T_event=event.duration * frequency (2Hz), L_out=selected layers

## CPU During Inference
- **HuggingFace feature extractor**: runs on CPU (expects numpy input)
- **Raw waveform**: loaded and resampled on CPU as torch tensor
- **Host buffer**: for final D2H copy

## Data Flow
```
Audio file (disk)
    |
    v
soundfile/moviepy (CPU) → waveform tensor (CPU)
    |
    v
Mono + resample (CPU, julius/torchaudio)
    |
    v
Normalize: wav / max(abs(wav)) (CPU)
    |
    v
HF feature extractor (CPU) → feature tensor
    |
    v  [non_blocking H2D]
Model forward (GPU, autocast bf16)
    |
    v
Hidden states (L, 1, T, D) on GPU
    |
    v  [on GPU]
squeeze + transpose → (L, D, T)
    |
    v  [on GPU]
F.interpolate to T_event frames (fp32 cast for interpolation)
    |
    v  [on GPU]
Layer aggregation → (L_out, D, T_event)
    |
    v  [single D2H copy, non_blocking=False]
Host buffer (float32) → .npy cache
```

## Sync Points
1. `host.copy_(layer_agg.detach(), non_blocking=False)` — single blocking D2H per event

No per-batch `.item()` calls. All aggregation and interpolation stays on device.

## Output
- **Shape**: `(L_out, D, T_event)` — temporal feature with time axis
- **Dtype**: float32
- **T_event**: `max(1, round(event.duration * frequency))` — e.g. 120 for 60s at 2Hz
- **Cached**: individual `.npy` file per audio event

## Cleanup
`free()` sets `_model = None`, `_feature_extractor = None`, calls `gc.collect()` + `torch.cuda.empty_cache()`

## Downstream
The brain model receives audio features as temporal arrays. The segment/dataloader slices by time window overlap, then the features for each TR are aggregated (sum) across the overlap.
