# Text Extractor (Llama-3.2-3B)

## Overview
Per-word contextualised language model embeddings. Tokenizes each word's context, runs the LM forward, extracts hidden states at selected layers, aggregates tokens to produce one feature vector per word.

## Model
- **Architecture**: `unsloth/Llama-3.2-3B` (causal LM)
- **Loaded dtype**: bf16 on Ampere+ (SM>=80), fp16 on older CUDA, fp32 on CPU
- **VRAM for weights**: ~6 GB (bf16)

## Input
| Field | Source | Description |
|-------|--------|-------------|
| `event.text` | Word event | The target word (e.g. "brings") |
| `event.context` | Word event | Left context window (e.g. "What brings") |
| `event.type` | Must be `"Word"` | Filters which events to process |

## VRAM During Inference
| Tensor | Shape | Notes |
|--------|-------|-------|
| Model weights | ~6 GB | Persistent until `free()` |
| `input_ids` + `attention_mask` | `(B, T)` | Tokenized batch, async H2D |
| Hidden states | `(L, B, T, D)` | All layers stacked — largest allocation |
| Per-word slices | `(L, n_target, D)` | Extracted per word in batch |
| Batch output | `(B, L_out, D)` | Stacked before D2H |

Where: B=batch_size (4), T=token length, L=model layers (28), D=hidden dim (3072), L_out=selected layers after aggregation

## CPU During Inference
- **Tokenizer**: HuggingFace tokenizer stays on CPU
- **Padding count**: `.item()` call per row to count pad tokens (minor sync point)
- **Host buffer**: Pinned memory for D2H copy
- **Cache I/O**: numpy `.npy` files to disk

## Data Flow
```
Words + Context (CPU strings)
    |
    v
Tokenizer (CPU) → input_ids, attention_mask
    |
    v  [non_blocking H2D]
Model forward (GPU, autocast bf16)
    |
    v
Hidden states (L, B, T, D) on GPU
    |
    v  [on GPU]
Token selection (find target word tokens via prefix re-tokenization)
    |
    v  [on GPU]
Token aggregation (mean/sum over target tokens) → (L, D) per word
    |
    v  [on GPU]
Layer aggregation (group_mean/mean/sum) → (L_out, D) per word
    |
    v  [single D2H copy per batch, non_blocking=False]
Host buffer (float32) → .npy cache
```

## Sync Points
1. `.item()` per row — extracts pad count scalar from GPU (minor)
2. `host.copy_(stacked.detach(), non_blocking=False)` — single blocking D2H per batch

## Output
- **Shape**: `(L_out, D)` per word event — 2D, no time axis (static feature)
- **Dtype**: float32
- **Cached**: individual `.npy` file per word event

## Cleanup
`free()` sets `_model = None`, `_tokenizer = None`, calls `gc.collect()` + `torch.cuda.empty_cache()`

## Downstream
The brain model receives text features as static per-word arrays. The segment/dataloader slices them by timestamp overlap with each prediction window.
