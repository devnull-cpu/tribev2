# Video Extractor (VJEPA2-ViT-G)

## Overview
Extracts temporal video embeddings from video frames. Reads frames via decord (batch extraction), preprocesses with HF processor or manual ImageNet normalization, runs the VJEPA2 vision transformer, and produces time-aligned feature arrays.

## Model
- **Architecture**: `facebook/vjepa2-vitg-fpc64-256` (ViT-Giant, 64 frames per clip, 256px)
- **Loaded dtype**: bf16 on Ampere+ (SM>=80), fp16 on older CUDA, fp32 on CPU
- **VRAM for weights**: ~4-5 GB (bf16) — largest extractor model
- **Frames per clip**: 64 (num_frames)
- **Clip duration**: 4 seconds (default) — the backward-looking window size in seconds. Each prediction timestep feeds the model 64 frames sampled from the preceding 4s of video. Adjacent windows overlap heavily.

## Input
| Field | Source | Description |
|-------|--------|-------------|
| `event.filepath` | Video event | Path to .mp4 file |
| `event.start` | Event metadata | Start time in source (seconds) |
| `event.duration` | Event metadata | Duration to extract (seconds) |

## Frame Extraction (Decord)
1. Compute all needed timestamps: `T_event` windows × 64 subtimes per window
2. Convert timestamps to frame indices: `int(time * fps)`
3. Deduplicate: `unique_indices = sorted(set(frame_indices))`
4. **Single batch call**: `vr.get_batch(unique_indices).asnumpy()` — one decord operation
5. Build lookup: `idx_to_frame[frame_idx] = numpy_array`
6. Reconstruct full sequence with deduplication reuse
7. Free decord reader and batch immediately after

For a 60s video at 50fps with 2Hz output and 64-frame clips:
- T_event = 120 windows
- Total frame requests = 120 × 64 = 7,680
- Unique frames ≈ 961 (many overlap between adjacent clips)

## VRAM During Inference
| Tensor | Shape | Notes |
|--------|-------|-------|
| Model weights | ~4-5 GB | Persistent until `free()` |
| `pixel_values` | `(1, 64, 3, 256, 256)` | One clip's preprocessed frames |
| Hidden states | `(L, 1, T_tokens, D)` | Per-window model output |
| `out_device` | `(T_event, L, D)` | Accumulator — grows across windows |
| `layer_agg` | `(L_out, D, T_event)` | After layer aggregation |

Where: L=model layers, D=hidden dim, T_event=event.duration * frequency, T_tokens=spatial tokens per frame

## CPU During Inference
| Data | Size | Notes |
|------|------|-------|
| `all_frames` | `T_event × 64` numpy arrays `(H,W,3)` | Full frame list from decord |
| `frames_np` | `(64, H, W, 3)` per window | Stacked for current window |
| Host buffer | `(L_out, D, T_event)` float32 | Final D2H destination |

## Data Flow
```
Video file (disk)
    |
    v
Decord VideoReader — single batch extraction of all unique frames
    |
    v
all_frames: list of numpy (H, W, 3) uint8 on CPU
    |
    v  [per window, T_event iterations]
    |
    +-- frames_np = np.stack(64 frames) → (64, H, W, 3) CPU
    |       |
    |       v
    |   _encode_frames() — GPU pipeline:
    |     pinned_buf.copy_(frames)       [CPU, page-locked]
    |     pinned_buf.to(device)          [async H2D on transfer stream]
    |     permute + /255 + resize + crop + normalize  [all GPU]
    |     wait_stream                    [sync transfer → compute]
    |       |
    |       v
    |   pixel_values on GPU → (1, 64, 3, 256, 256)
    |       |
    |       v
    |   Model forward (GPU, autocast bf16)
    |       |
    |       v
    |   Hidden states → token aggregation (mean) → (L, D)
    |       |
    |       v
    |   out_device[k] = hidden  [stays on GPU]
    |
    v  [after all windows]
Layer aggregation on GPU → (L_out, D, T_event)
    |
    v  [single D2H copy, non_blocking=False]
Host buffer (float32) → .npy cache
```

## Sync Points
1. `host.copy_(layer_agg.detach(), non_blocking=False)` — single blocking D2H for entire event

No per-window D2H copies. All window results accumulate in `out_device` on GPU.

## Preprocessing (_encode_frames) — GPU Pipeline
All preprocessing runs on GPU via torchvision transforms with pinned memory staging:

1. **Pinned buffer**: reusable page-locked CPU buffer (`torch.empty(..., pin_memory=True)`)
2. **Async H2D**: copy to GPU on a dedicated CUDA transfer stream (`non_blocking=True`)
3. **On GPU** (transfer stream):
   - Permute NHWC → NCHW
   - Rescale: `/ 255.0`
   - `torchvision.transforms.Resize(292, antialias=True)`
   - `torchvision.transforms.CenterCrop(256)`
   - `torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
4. **Stream sync**: compute stream waits for transfer stream before model forward
5. Add batch dim: `(1, T, 3, 256, 256)`

No HuggingFace processor — eliminated entirely. Zero CPU preprocessing.

## Output
- **Shape**: `(L_out, D, T_event)` — temporal feature with time axis
- **Dtype**: float32
- **T_event**: `max(1, round(event.duration * frequency))` — e.g. 120 for 60s at 2Hz
- **Cached**: individual `.npy` file per video event

## Memory Lifecycle
```
1. Decord loads → all_frames on CPU
2. del batch, vr, idx_to_frame     ← free decord resources
3. Per-window loop: stack 64 frames, encode, forward, accumulate on GPU
4. del all_frames                   ← free CPU frame list
5. Layer aggregation on GPU
6. Single D2H copy
7. free() → del model, processor, gc.collect(), empty_cache()
```

## Cleanup
`free()` sets `_model = None`, `_processor = None`, calls `gc.collect()` + `torch.cuda.empty_cache()`

## Downstream
The brain model receives video features as temporal arrays. The segment/dataloader slices by time window overlap. Video features are the dominant modality for visual content — the model's visual cortex predictions come primarily from these.

## Bottleneck Analysis
| Stage | Status | Implementation |
|-------|--------|---------------|
| Frame extraction | Done | Decord batch (CPU) or NVDEC slice (GPU, standard pipeline) |
| Preprocessing | Done | GPU torchvision transforms, no HF processor |
| H2D transfer | Done | Pinned memory + dedicated CUDA transfer stream |
| Model forward | Done | autocast bf16, inference_mode |
| Accumulation | Done | On-device `out_device` tensor, no per-window D2H |
| D2H transfer | Done | Single blocking copy per event |
| Remaining | -- | Batch multiple clips per forward (requires model changes) |
