"""TRIBE v2 Explorer — Gradio dashboard for brain activity prediction."""

import base64
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=r"LabelEncoder.*")
warnings.filterwarnings("ignore", message=r".*Index.*column.*")
warnings.filterwarnings("ignore", message=r".*autocast.*deprecated.*")
logging.getLogger("neuralset.extractors.base").setLevel(logging.ERROR)

# Suppress ffmpeg console windows on Windows
if os.name == "nt":
    _orig_popen = subprocess.Popen.__init__
    def _quiet_popen(self, *a, **kw):
        kw.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        _orig_popen(self, *a, **kw)
    subprocess.Popen.__init__ = _quiet_popen

from tribev2.demo_utils import TribeModel, get_audio_and_text_events, build_text_events_from_text
from tribev2.plotting import PlotBrainNilearn
from tribev2.utils import get_hcp_labels, summarize_by_roi
from tribev2.viewer import build_viewer_html

CACHE = Path("./cache")
CACHE.mkdir(exist_ok=True)
RUNS_DIR = CACHE / "runs"
RUNS_DIR.mkdir(exist_ok=True)

ZONE_GROUPS = {
    "Visual": ["V1", "V2", "V3", "V4", "V6", "V7", "V8", "MT", "MST", "LO", "FFC", "VVC", "VMV", "PIT"],
    "Auditory": ["A1", "A4", "A5", "LBelt", "MBelt", "PBelt", "RI", "STGa", "STSda", "STSdp", "STSva", "STSvp"],
    "Language": ["55b", "SFL", "PSL", "TPOJ1", "TPOJ2", "TPOJ3", "PGi", "PGs"],
    "Frontal": ["FEF", "PEF", "8Av", "8BL", "8C", "IFSa", "IFSp", "IFJa", "IFJp", "44", "45", "47l"],
    "Motor": ["1", "2", "3a", "3b", "4", "6mp", "6ma", "6d", "6v", "6a", "6r"],
    "Default mode": ["POS1", "POS2", "RSC", "v23ab", "d23ab", "31a", "31pd", "31pv", "7m", "PCV", "DVT"],
}

# ── Model loading ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_model():
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE,
        config_update={
            "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
            "data.num_workers": 0,
        },
    )
    return model

@lru_cache(maxsize=1)
def get_plotter():
    return PlotBrainNilearn(mesh="fsaverage5")

@lru_cache(maxsize=1)
def get_roi_indices():
    labels = get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both")
    zone_indices = {}
    for zone_name, keywords in ZONE_GROUPS.items():
        verts = []
        for kw in keywords:
            for label_name, indices in labels.items():
                if label_name.startswith(kw):
                    verts.append(indices)
        if verts:
            zone_indices[zone_name] = np.concatenate(verts)
    return zone_indices

# ── Brain rendering ────────────────────────────────────────────────────

def _fig_to_image(fig):
    """Convert a matplotlib figure to a numpy RGB array."""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    return img

def render_brain(preds_single, plotter, vmin=0.5, norm_pct=99):
    fig, axes = plotter.get_fig_axes(views=["left", "right", "dorsal"])
    plotter.plot_surf(
        preds_single,
        axes=axes,
        views=["left", "right", "dorsal"],
        cmap="fire",
        norm_percentile=norm_pct,
        vmin=vmin,
        alpha_cmap=(0, 0.2),
    )
    fig.set_size_inches(10, 3)
    return fig

def render_brain_row(preds, indices, plotter, time_axis=None):
    n = len(indices)
    if n == 0:
        return plt.figure()
    fig_all, axes_all = plt.subplots(1, n, figsize=(3.2 * n, 3))
    if n == 1:
        axes_all = [axes_all]
    for i, idx in enumerate(indices):
        brain_fig = render_brain(preds[idx], plotter)
        img = _fig_to_image(brain_fig)
        plt.close(brain_fig)
        axes_all[i].imshow(img)
        t_s = time_axis[idx] if time_axis and idx < len(time_axis) else idx
        axes_all[i].set_title(f"t={t_s:.1f}s", fontsize=9)
        axes_all[i].axis("off")
    fig_all.tight_layout(pad=0.3)
    return fig_all

# ── Zone time-series ───────────────────────────────────────────────────

def build_zone_timeseries(preds, time_axis=None):
    zone_indices = get_roi_indices()
    n_timesteps = preds.shape[0]
    data = {}
    for zone_name, indices in zone_indices.items():
        data[zone_name] = [float(np.abs(preds[t, indices]).mean()) for t in range(n_timesteps)]
    df = pd.DataFrame(data)
    if time_axis is not None and len(time_axis) == n_timesteps:
        df.index = time_axis
    df.index.name = "time_s"
    return df

def plot_zone_timeseries(df):
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#264653"]
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], label=col, linewidth=1.8, color=colors[i % len(colors)])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean |activation|")
    ax.set_title("Brain region activity over time")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

# ── Run persistence ────────────────────────────────────────────────────

import hashlib
from datetime import datetime

def save_run(preds, time_axis, zone_df, input_kind, source_name, video_path=None):
    """Save a run to disk for instant reload."""
    run_id = hashlib.md5(f"{source_name}_{preds.shape}_{datetime.now().isoformat()}".encode()).hexdigest()[:10]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    np.save(run_dir / "preds.npy", preds)
    np.save(run_dir / "time_axis.npy", np.array(time_axis))
    zone_df.to_csv(run_dir / "zones.csv", index=True)
    meta = {
        "id": run_id,
        "input_kind": input_kind,
        "source_name": source_name,
        "n_timesteps": int(preds.shape[0]),
        "n_vertices": int(preds.shape[1]),
        "duration_s": round(float(time_axis[-1] - time_axis[0]), 1) if len(time_axis) > 1 else 0,
        "created": datetime.now().isoformat(timespec="seconds"),
        "video_path": str(video_path) if video_path else None,
    }
    import json
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_id


def list_runs():
    """List all saved runs, newest first."""
    import json
    runs = []
    for meta_path in RUNS_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            runs.append(meta)
        except Exception:
            pass
    runs.sort(key=lambda r: r.get("created", ""), reverse=True)
    return runs


def load_run(run_id):
    """Load a saved run."""
    run_dir = RUNS_DIR / run_id
    preds = np.load(run_dir / "preds.npy")
    time_axis = np.load(run_dir / "time_axis.npy").tolist()
    zone_df = pd.read_csv(run_dir / "zones.csv", index_col=0)
    import json
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    return preds, time_axis, zone_df, meta

# ── Event preparation ─────────────────────────────────────────────────

def prepare_video_events(video_path, chunk=True):
    events = pd.DataFrame([{
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }])
    if chunk:
        return get_audio_and_text_events(events)
    from neuralset.events.transforms import (
        ExtractAudioFromVideo, AddText, AddSentenceToWords,
        AddContextToWords, RemoveMissing, ChunkEvents,
    )
    from neuralset.events.utils import standardize_events
    from tribev2.eventstransforms import ExtractWordsFromAudio
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

def prepare_audio_events(audio_path):
    events = pd.DataFrame([{
        "type": "Audio",
        "filepath": str(audio_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }])
    return get_audio_and_text_events(events)

def prepare_text_events(text):
    return build_text_events_from_text(text)

# ── Main prediction ───────────────────────────────────────────────────

log = logging.getLogger("tribe_app")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    log.addHandler(_h)


def run_prediction(video, audio, text, pipeline_backend, decoder_backend, progress=gr.Progress()):
    if video is None and audio is None and (text is None or text.strip() == ""):
        raise gr.Error("Upload a video, audio, or enter text first.")

    import time as _time

    use_fast = (pipeline_backend == "Fast (sync-free)")
    log.info("Pipeline: %s | Video decoder: %s", pipeline_backend, decoder_backend)

    if not use_fast:
        from neuralset.extractors import video as _vid_module
        _vid_module.FORCE_DECORD = (decoder_backend == "Decord (CPU)")

    log.info("Loading model...")
    progress(0.05, desc="Loading model...")
    t0 = _time.time()
    model = get_model()
    plotter = get_plotter()
    log.info("Model loaded (%.1fs)", _time.time() - t0)

    progress(0.1, desc="Preparing events (extract audio, transcribe)...")
    log.info("Preparing events...")
    t0 = _time.time()
    if video is not None:
        df = prepare_video_events(video, chunk=not use_fast)
        input_kind = "video"
    elif audio is not None:
        df = prepare_audio_events(audio)
        input_kind = "audio"
    else:
        df = prepare_text_events(text)
        input_kind = "text"
    log.info("Events ready (%.1fs) — %d events", _time.time() - t0, len(df))

    progress(0.3, desc="Extracting features & running inference...")
    t0 = _time.time()
    if use_fast:
        from tribev2.fast import FastTribePipeline
        log.info("Running FastTribePipeline.predict()...")
        fast = FastTribePipeline.from_pretrained(
            "facebook/tribev2",
            cache_dir=str(CACHE),
            text_model="unsloth/Llama-3.2-3B",
        )
        preds, segments = fast.predict(events=df)
    else:
        log.info("Running model.predict()...")
        preds, segments = model.predict(events=df, verbose=True)
    log.info("Inference done (%.1fs) — %d timesteps × %d vertices", _time.time() - t0, *preds.shape)

    progress(0.7, desc="Building zone timeseries...")
    log.info("Building zone timeseries...")
    n = preds.shape[0]
    if segments:
        seg0 = segments[0]
        video_duration = max(e.start + e.duration for seg in segments for e in seg.ns_events if hasattr(e, 'type') and e.type == 'Video') if any(e.type == 'Video' for seg in segments for e in seg.ns_events) else None
        if video_duration and video_duration > 0:
            time_axis = [i * video_duration / n for i in range(n)]
            log.info("Time axis: 0 - %.1fs (%d steps, %.2f Hz)", video_duration, n, n / video_duration)
        else:
            time_axis = [s.start for s in segments]
    else:
        time_axis = list(range(n))
    zone_df = build_zone_timeseries(preds, time_axis=time_axis)
    ts_fig = plot_zone_timeseries(zone_df)

    progress(0.8, desc="Building 3D brain viewer...")
    log.info("Building 3D brain viewer...")
    t0 = _time.time()
    zone_dict = {col: zone_df[col].tolist() for col in zone_df.columns}
    viewer_html_content = build_viewer_html(
        preds, time_axis=time_axis, zone_data=zone_dict,
        height=500,
    )
    import html as _html
    viewer_html = '<iframe srcdoc="' + _html.escape(viewer_html_content) + '" width="100%" height="750" frameborder="0" sandbox="allow-scripts"></iframe>'
    log.info("3D viewer built (%.1fs)", _time.time() - t0)

    progress(0.9, desc="Rendering brain panels...")
    log.info("Rendering brain overview...")
    t0 = _time.time()
    n = preds.shape[0]
    step = max(1, n // 8)
    sample_indices = list(range(0, n, step))[:8]
    brain_fig = render_brain_row(preds, sample_indices, plotter, time_axis=time_axis)
    log.info("Brain overview done (%.1fs)", _time.time() - t0)

    source_name = Path(video or audio or "text").name if (video or audio) else "text_input"
    run_id = save_run(preds, time_axis, zone_df, input_kind, source_name, video_path=video)
    log.info("Run saved: %s", run_id)

    log.info("All done!")
    progress(1.0, desc="Done!")

    duration_s = time_axis[-1] - time_axis[0] if len(time_axis) > 1 else 0
    hz = preds.shape[0] / duration_s if duration_s > 0 else 0
    summary = f"**{preds.shape[0]}** timesteps ({duration_s:.1f}s at {hz:.1f}Hz) × **{preds.shape[1]}** vertices ({input_kind})"

    return (
        summary,
        ts_fig,
        brain_fig,
        viewer_html,
        preds,
        _format_runs_list(),
    )


def _format_runs_list():
    runs = list_runs()
    if not runs:
        return "No saved runs yet."
    lines = []
    for r in runs:
        lines.append(f"- **{r['source_name']}** ({r['input_kind']}) — {r['n_timesteps']} steps, {r['duration_s']}s — `{r['id']}` — {r['created']}")
    return "\n".join(lines)


def load_previous_run(run_id_text):
    if not run_id_text or not run_id_text.strip():
        raise gr.Error("Enter a run ID")
    run_id = run_id_text.strip().split("`")[-2] if "`" in run_id_text else run_id_text.strip()
    try:
        preds, time_axis, zone_df, meta = load_run(run_id)
    except Exception as e:
        raise gr.Error(f"Failed to load run: {e}")

    ts_fig = plot_zone_timeseries(zone_df)

    zone_dict = {col: zone_df[col].tolist() for col in zone_df.columns}
    video_path = meta.get("video_path")
    viewer_html_content = build_viewer_html(
        preds, time_axis=time_axis, zone_data=zone_dict,
        video_path=video_path, height=500,
    )
    import html as _html
    viewer_html = '<iframe srcdoc="' + _html.escape(viewer_html_content) + '" width="100%" height="750" frameborder="0" sandbox="allow-scripts"></iframe>'

    plotter = get_plotter()
    n = preds.shape[0]
    step = max(1, n // 8)
    sample_indices = list(range(0, n, step))[:8]
    brain_fig = render_brain_row(preds, sample_indices, plotter, time_axis=time_axis)

    duration_s = time_axis[-1] - time_axis[0] if len(time_axis) > 1 else 0
    hz = n / duration_s if duration_s > 0 else 0
    summary = f"**{n}** timesteps ({duration_s:.1f}s at {hz:.1f}Hz) × **{preds.shape[1]}** vertices ({meta['input_kind']}) — loaded from `{run_id}`"

    return (
        summary,
        ts_fig,
        brain_fig,
        viewer_html,
        preds,
        _format_runs_list(),
    )

def update_timestep(timestep, preds_state):
    if preds_state is None:
        return plt.figure()
    plotter = get_plotter()
    idx = int(timestep)
    idx = min(idx, preds_state.shape[0] - 1)
    return render_brain(preds_state[idx], plotter)

# ── UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="TRIBE v2 Explorer",
        theme=gr.themes.Base(
            primary_hue="orange",
            neutral_hue="zinc",
        ),
        css="""
        .contain { max-width: 1400px; margin: auto; }
        footer { display: none !important; }
        """,
    ) as app:
        gr.Markdown("# TRIBE v2 Explorer\nPredict fMRI brain responses to video, audio, or text.")

        preds_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                video_in = gr.Video(label="Video", sources=["upload"])
                audio_in = gr.Audio(label="Audio", sources=["upload"], type="filepath")
                text_in = gr.Textbox(label="Text", lines=4, placeholder="Enter text to predict brain response...")
                with gr.Accordion("Advanced", open=False):
                    pipeline_choice = gr.Radio(
                        choices=["Standard", "Fast (sync-free)"],
                        value="Fast (sync-free)",
                        label="Pipeline",
                    )
                    decoder_choice = gr.Radio(
                        choices=["NVDEC (GPU)", "Decord (CPU)"],
                        value="Decord (CPU)",
                        label="Video decoder (Standard pipeline only)",
                    )
                run_btn = gr.Button("Run prediction", variant="primary", size="lg")
                summary_md = gr.Markdown("")

            with gr.Column(scale=3):
                gr.Markdown("### Results")

                with gr.Tab("3D Brain"):
                    brain_3d = gr.HTML(label="Interactive 3D brain viewer")

                with gr.Tab("Timeline"):
                    ts_plot = gr.Plot(label="Region activity over time")

                with gr.Tab("Brain overview"):
                    brain_row_plot = gr.Plot(label="Brain activity across timesteps")

                with gr.Tab("History"):
                    runs_md = gr.Markdown(_format_runs_list())
                    load_id = gr.Textbox(label="Run ID", placeholder="Paste run ID to load...")
                    load_btn = gr.Button("Load run", variant="secondary")

        all_outputs = [summary_md, ts_plot, brain_row_plot, brain_3d, preds_state, runs_md]

        run_btn.click(
            fn=run_prediction,
            inputs=[video_in, audio_in, text_in, pipeline_choice, decoder_choice],
            outputs=all_outputs,
        )

        load_btn.click(
            fn=load_previous_run,
            inputs=[load_id],
            outputs=all_outputs,
        )

    return app

if __name__ == "__main__":
    app = build_ui()
    gr.set_static_paths([str(CACHE.resolve())])
    app.launch(inbrowser=True, allowed_paths=[str(CACHE.resolve())])
