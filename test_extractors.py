"""Benchmark individual feature extractors with GPU profiling."""

import argparse
import contextlib
import os
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")

if os.name == "nt":
    _orig = subprocess.Popen.__init__
    def _quiet(self, *a, **kw):
        kw.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        _orig(self, *a, **kw)
    subprocess.Popen.__init__ = _quiet

import numpy as np
import pandas as pd
import torch

from tribev2.demo_utils import build_text_events_from_text, get_audio_and_text_events
from tribev2.fast.extractors import (
    AudioExtractor,
    ExtractorConfig,
    TextExtractor,
    VideoExtractor,
)
from tribev2.fast.segments import events_from_dataframe

CACHE_DIR = "./cache/test_extractors"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYERS = [0.5, 0.75, 1.0]


def _dtype():
    if DEVICE.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _vram_mb():
    if DEVICE.type != "cuda":
        return 0, 0
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    return allocated, reserved


@contextlib.contextmanager
def gpu_timer(label):
    """Time a block using CUDA events for accurate GPU measurement."""
    if DEVICE.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        alloc_before, _ = _vram_mb()
        start_event.record()
        yield
        end_event.record()
        torch.cuda.synchronize()
        gpu_ms = start_event.elapsed_time(end_event)
        alloc_after, _ = _vram_mb()
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"  {label}:")
        print(f"    GPU time:  {gpu_ms:.1f} ms")
        print(f"    VRAM:      {alloc_before:.0f} -> {alloc_after:.0f} MB (peak {peak:.0f} MB)")
    else:
        t0 = time.perf_counter()
        yield
        print(f"  {label}: {(time.perf_counter() - t0) * 1000:.1f} ms (CPU)")


def make_text_config():
    return ExtractorConfig(
        model_name="unsloth/Llama-3.2-3B",
        layers=LAYERS,
        frequency=2.0,
        batch_size=4,
        dtype=_dtype(),
    )


def make_audio_config():
    return ExtractorConfig(
        model_name="facebook/w2v-bert-2.0",
        layers=[0.75, 1.0],
        frequency=2.0,
        batch_size=1,
        dtype=_dtype(),
    )


def make_video_config(quantize=None):
    return ExtractorConfig(
        model_name="facebook/vjepa2-vitg-fpc64-256",
        layers=[0.75, 1.0],
        frequency=2.0,
        batch_size=1,
        dtype=_dtype(),
        quantize=quantize,
    )


def bench_text(events_df, warmup=True):
    print("\n--- Text Extractor (Llama-3.2-3B) ---")
    fast_events = events_from_dataframe(events_df)
    word_events = [e for e in fast_events if e.type == "Word"]
    if not word_events:
        print("  No Word events, skipping")
        return
    print(f"  Events: {len(word_events)} words")

    config = make_text_config()
    extractor = TextExtractor(config, cache_dir=CACHE_DIR, device=DEVICE)

    if warmup:
        print("  Warm-up run...")
        import shutil; shutil.rmtree(extractor.cache.root, ignore_errors=True); extractor.cache.root.mkdir(parents=True, exist_ok=True)
        extractor.prepare(word_events[:min(5, len(word_events))])
        import shutil; shutil.rmtree(extractor.cache.root, ignore_errors=True); extractor.cache.root.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats() if DEVICE.type == "cuda" else None

    with gpu_timer("prepare (model load + inference)"):
        import shutil; shutil.rmtree(extractor.cache.root, ignore_errors=True); extractor.cache.root.mkdir(parents=True, exist_ok=True)
        extractor.prepare(word_events)

    cached = sum(1 for e in word_events if extractor.has_cached(e))
    sample = extractor.load_cached(word_events[0]) if cached else None
    print(f"  Cached: {cached}/{len(word_events)}")
    if sample is not None:
        print(f"  Output shape: {sample.shape} ({sample.dtype})")

    # Cleanup
    del extractor
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None


def bench_audio(events_df, warmup=True):
    print("\n--- Audio Extractor (Wav2Vec-BERT 2.0) ---")
    fast_events = events_from_dataframe(events_df)
    audio_events = [e for e in fast_events if e.type in ("Audio", "Video")]
    if not audio_events:
        print("  No Audio/Video events, skipping")
        return
    total_dur = sum(e.duration for e in audio_events)
    print(f"  Events: {len(audio_events)} ({total_dur:.1f}s total)")

    config = make_audio_config()
    extractor = AudioExtractor(config, cache_dir=CACHE_DIR, device=DEVICE)

    if warmup:
        print("  Warm-up run...")
        import shutil; shutil.rmtree(extractor.cache.root, ignore_errors=True); extractor.cache.root.mkdir(parents=True, exist_ok=True)
        extractor.prepare(audio_events[:1])
        import shutil; shutil.rmtree(extractor.cache.root, ignore_errors=True); extractor.cache.root.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats() if DEVICE.type == "cuda" else None

    with gpu_timer("prepare (model load + inference)"):
        import shutil; shutil.rmtree(extractor.cache.root, ignore_errors=True); extractor.cache.root.mkdir(parents=True, exist_ok=True)
        extractor.prepare(audio_events)

    cached = sum(1 for e in audio_events if extractor.has_cached(e))
    sample = extractor.load_cached(audio_events[0]) if cached else None
    print(f"  Cached: {cached}/{len(audio_events)}")
    if sample is not None:
        print(f"  Output shape: {sample.shape} ({sample.dtype})")

    del extractor
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None


def bench_video(events_df, warmup=True, quantize=None):
    q_label = f" [{quantize}]" if quantize else ""
    print(f"\n--- Video Extractor (VJEPA2-ViT-G){q_label} ---")
    fast_events = events_from_dataframe(events_df)
    video_events = [e for e in fast_events if e.type == "Video"]
    if not video_events:
        print("  No Video events, skipping")
        return
    total_dur = sum(e.duration for e in video_events)
    print(f"  Events: {len(video_events)} ({total_dur:.1f}s total)")

    config = make_video_config(quantize=quantize)
    extractor = VideoExtractor(config, cache_dir=CACHE_DIR, device=DEVICE)

    if warmup:
        print("  Warm-up (dummy forward only)...")
        extractor._load_model()
        dummy = np.random.randint(0, 255, (extractor.num_frames, 256, 256, 3), dtype=np.uint8)
        pv = extractor._encode_frames(dummy)
        fwd_key = "pixel_values_videos" if "vjepa2" in config.model_name else "pixel_values"
        with torch.inference_mode():
            extractor._model(**{fwd_key: pv}, output_hidden_states=True)
        del pv, dummy
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
        torch.cuda.reset_peak_memory_stats() if DEVICE.type == "cuda" else None
        print("  Warm-up done.")

    with gpu_timer("prepare (model load + inference)"):
        import shutil; shutil.rmtree(extractor.cache.root, ignore_errors=True); extractor.cache.root.mkdir(parents=True, exist_ok=True)
        extractor.prepare(video_events)

    cached = sum(1 for e in video_events if extractor.has_cached(e))
    sample = extractor.load_cached(video_events[0]) if cached else None
    print(f"  Cached: {cached}/{len(video_events)}")
    if sample is not None:
        print(f"  Output shape: {sample.shape} ({sample.dtype})")

    del extractor
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None


def prepare_text_events():
    text = """To be or not to be, that is the question.
Whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles
and by opposing end them. To die, to sleep,
no more; and by a sleep to say we end
the heartache and the thousand natural shocks
that flesh is heir to. Tis a consummation
devoutly to be wished. To die, to sleep.
To sleep, perchance to dream. Ay, there's the rub,
for in that sleep of death what dreams may come
when we have shuffled off this mortal coil
must give us pause. There's the respect
that makes calamity of so long life.
For who would bear the whips and scorns of time,
the oppressor's wrong, the proud man's contumely,
the pangs of despised love, the law's delay,
the insolence of office and the spurns
that patient merit of the unworthy takes,
when he himself might his quietus make
with a bare bodkin. Who would fardels bear,
to grunt and sweat under a weary life,
but that the dread of something after death,
the undiscovered country from whose bourn
no traveller returns, puzzles the will
and makes us rather bear those ills we have
than fly to others that we know not of.
Thus conscience does make cowards of us all,
and thus the native hue of resolution
is sicklied o'er with the pale cast of thought,
and enterprises of great pith and moment
with this regard their currents turn awry,
and lose the name of action."""
    return build_text_events_from_text(text)


def bench_model(events_df, warmup=True):
    print("\n--- Brain Model (FmriEncoder) via FastTribePipeline ---")
    from tribev2.fast import FastTribePipeline

    print("  Loading pipeline...")
    pipeline = FastTribePipeline.from_pretrained(
        "facebook/tribev2",
        cache_dir="./cache/fast_bench",
        text_model="unsloth/Llama-3.2-3B",
        num_workers=0,
    )

    print("  First run (extract features + model forward)...")
    t0 = time.perf_counter()
    preds, segments = pipeline.predict(events=events_df)
    print(f"  Full pipeline: {time.perf_counter() - t0:.1f}s — {preds.shape}")

    print("  Second run (features cached, model-only)...")
    torch.cuda.reset_peak_memory_stats() if DEVICE.type == "cuda" else None
    with gpu_timer("model forward only (cached features)"):
        preds, segments = pipeline.predict(events=events_df)

    print(f"  Output: {preds.shape} ({preds.dtype})")
    del pipeline
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None


def prepare_video_events(video_path):
    from neuralset.events.transforms import (
        ExtractAudioFromVideo, AddText, AddSentenceToWords,
        AddContextToWords, RemoveMissing, ChunkEvents,
    )
    from neuralset.events.utils import standardize_events
    from tribev2.eventstransforms import ExtractWordsFromAudio
    events = pd.DataFrame([{
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }])
    transforms = [
        ExtractAudioFromVideo(),
        ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
        ExtractWordsFromAudio(),
        AddText(),
        AddSentenceToWords(max_unmatched_ratio=0.05),
        AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
        RemoveMissing(),
    ]
    events = standardize_events(events)
    for t in transforms:
        events = t(events)
    return standardize_events(events)


def main():
    parser = argparse.ArgumentParser(description="Benchmark TRIBE v2 feature extractors")
    parser.add_argument("video", nargs="?", help="Path to video file")
    parser.add_argument("--text-only", action="store_true", help="Only run text extractor")
    parser.add_argument("--audio-only", action="store_true", help="Only run audio extractor")
    parser.add_argument("--video-only", action="store_true", help="Only run video extractor")
    parser.add_argument("--model-only", action="store_true", help="Only run brain model (needs cached features)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warm-up run")
    parser.add_argument("--quantize", choices=["int8", "int4", "fp8"], default=None, help="Quantize models")
    args = parser.parse_args()

    run_all = not (args.text_only or args.audio_only or args.video_only or args.model_only)
    warmup = not args.no_warmup

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute: sm_{props.major}{props.minor}")
        print(f"Dtype: {_dtype()}")
    print()

    video_path = args.video
    if video_path is None and os.path.exists("cache/sample_video.mp4"):
        video_path = "cache/sample_video.mp4"

    if video_path and (run_all or args.model_only):
        print(f"Preparing full events from: {video_path}")
        t0 = time.perf_counter()
        df = prepare_video_events(video_path)
        print(f"Event prep: {time.perf_counter() - t0:.1f}s — {len(df)} events")
        print(f"Types: {df.type.value_counts().to_dict()}")
    elif video_path:
        from neuralset.events.utils import standardize_events
        rows = []
        ext = os.path.splitext(video_path)[1].lower()
        file_type = "Audio" if ext in (".wav", ".mp3", ".flac", ".ogg") else "Video"
        if args.video_only or args.audio_only:
            rows.append({"type": file_type, "filepath": str(video_path), "start": 0,
                         "duration": None, "timeline": "default", "subject": "default"})
        if args.text_only:
            df = prepare_text_events()
            print(f"Events: {len(df)} ({df.type.value_counts().to_dict()})")
        if rows:
            df = standardize_events(pd.DataFrame(rows))
            print(f"Events (direct): {len(df)} ({df.type.value_counts().to_dict()})")
    else:
        print("No video — running text-only test")
        df = prepare_text_events()
        print(f"Events: {len(df)} ({df.type.value_counts().to_dict()})")

    if run_all or args.text_only:
        bench_text(df, warmup=warmup)

    if (run_all or args.audio_only) and ("Audio" in df.type.values or "Video" in df.type.values):
        bench_audio(df, warmup=warmup)

    if (run_all or args.video_only) and "Video" in df.type.values:
        bench_video(df, warmup=warmup, quantize=args.quantize)

    if run_all or args.model_only:
        bench_model(df, warmup=warmup)

    print("\n" + "=" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
