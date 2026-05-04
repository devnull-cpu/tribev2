"""TRIBE v2 Explorer — Flask web app."""

import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from threading import Thread

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
from flask import Flask, render_template_string, request, jsonify, send_file, redirect, url_for

from tribev2.demo_utils import TribeModel, get_audio_and_text_events, build_text_events_from_text
from tribev2.viewer import build_viewer_html
from tribev2.utils import get_hcp_labels

log = logging.getLogger("tribe_app")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    log.addHandler(_h)

app = Flask(__name__)
CACHE = Path("./cache")
CACHE.mkdir(exist_ok=True)
UPLOAD_DIR = CACHE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
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

# ── Model ──────────────────────────────────────────────────────────────

_model = None

def get_model():
    global _model
    if _model is None:
        _model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=CACHE,
            config_update={
                "data.text_feature.model_name": "unsloth/Llama-3.2-3B",
                "data.num_workers": 0,
            },
        )
    return _model

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

def build_zone_timeseries(preds, time_axis):
    zone_indices = get_roi_indices()
    n = preds.shape[0]
    data = {}
    for zone_name, indices in zone_indices.items():
        data[zone_name] = [float(np.abs(preds[t, indices]).mean()) for t in range(n)]
    df = pd.DataFrame(data)
    df.index = time_axis
    return df

# ── Run persistence ────────────────────────────────────────────────────

import json
import hashlib
from datetime import datetime

def save_run(preds, time_axis, zone_df, input_kind, source_name, video_path=None):
    run_id = hashlib.md5(f"{source_name}_{preds.shape}_{datetime.now().isoformat()}".encode()).hexdigest()[:10]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    np.save(run_dir / "preds.npy", preds)
    np.save(run_dir / "time_axis.npy", np.array(time_axis))
    zone_df.to_csv(run_dir / "zones.csv", index=True)
    meta = {
        "id": run_id, "input_kind": input_kind, "source_name": source_name,
        "n_timesteps": int(preds.shape[0]), "n_vertices": int(preds.shape[1]),
        "duration_s": round(float(time_axis[-1] - time_axis[0]), 1) if len(time_axis) > 1 else 0,
        "created": datetime.now().isoformat(timespec="seconds"),
        "video_path": str(video_path) if video_path else None,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_id

def list_runs():
    runs = []
    for meta_path in RUNS_DIR.glob("*/meta.json"):
        try:
            runs.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    runs.sort(key=lambda r: r.get("created", ""), reverse=True)
    return runs

def load_run(run_id):
    run_dir = RUNS_DIR / run_id
    preds = np.load(run_dir / "preds.npy")
    time_axis = np.load(run_dir / "time_axis.npy").tolist()
    zone_df = pd.read_csv(run_dir / "zones.csv", index_col=0)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    return preds, time_axis, zone_df, meta

# ── Routes ─────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>TRIBE v2 Explorer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; color: #eee; font-family: -apple-system, system-ui, sans-serif; }
  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 24px; font-weight: 600; margin-bottom: 4px; }
  h1 span { color: #ea580c; }
  .subtitle { color: #888; font-size: 14px; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: 300px 1fr; gap: 24px; }
  .panel { background: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 20px; }
  .panel h2 { font-size: 15px; font-weight: 600; margin-bottom: 12px; color: #ccc; }
  label { display: block; font-size: 13px; color: #999; margin-bottom: 4px; }
  input[type=file], input[type=text], textarea, select {
    width: 100%; padding: 8px 12px; background: #222; border: 1px solid #444;
    border-radius: 8px; color: #eee; font-size: 13px; margin-bottom: 12px;
  }
  textarea { height: 80px; resize: vertical; }
  button {
    padding: 10px 20px; border-radius: 8px; font-size: 14px; font-weight: 500;
    cursor: pointer; border: none; transition: background 0.2s;
  }
  .btn-primary { background: #ea580c; color: white; width: 100%; }
  .btn-primary:hover { background: #c2410c; }
  .btn-primary:disabled { background: #555; cursor: wait; }
  .btn-secondary { background: #333; color: #eee; border: 1px solid #555; }
  .btn-secondary:hover { background: #444; }
  .status { font-size: 12px; color: #888; margin-top: 8px; min-height: 20px; }
  .runs-list { list-style: none; }
  .runs-list li {
    padding: 10px; margin-bottom: 6px; background: #222; border-radius: 8px;
    border: 1px solid #333; cursor: pointer; transition: border-color 0.2s;
  }
  .runs-list li:hover { border-color: #ea580c; }
  .runs-list .name { font-weight: 500; font-size: 13px; }
  .runs-list .meta { font-size: 11px; color: #777; margin-top: 2px; }
  #viewer-frame { width: 100%; height: calc(100vh - 60px); min-height: 800px; border: none; border-radius: 12px; background: #1a1a1a; }
  .hidden { display: none; }
</style>
</head><body>
<div class="container">
  <h1><span>TRIBE v2</span> Explorer</h1>
  <p class="subtitle">Predict fMRI brain responses to video, audio, or text</p>

  <div class="grid">
    <div>
      <div class="panel">
        <h2>Input</h2>
        <form id="uploadForm" enctype="multipart/form-data">
          <label>Video</label>
          <input type="file" name="video" accept="video/*">
          <label>Audio</label>
          <input type="file" name="audio" accept="audio/*">
          <label>Text</label>
          <textarea name="text" placeholder="Enter text to predict brain response..."></textarea>
          <button type="submit" class="btn-primary" id="runBtn">Run prediction</button>
        </form>
        <div class="status" id="status"></div>
      </div>

      <div class="panel" style="margin-top: 16px;">
        <h2>Previous runs</h2>
        <ul class="runs-list" id="runsList"></ul>
      </div>
    </div>

    <div>
      <iframe id="viewer-frame" class="hidden"></iframe>
      <div id="placeholder" class="panel" style="height: 400px; display: flex; align-items: center; justify-content: center;">
        <p style="color: #555; font-size: 16px;">Upload a file and run prediction to see the brain viewer</p>
      </div>
    </div>
  </div>
</div>

<script>
const status = document.getElementById('status');
const frame = document.getElementById('viewer-frame');
const placeholder = document.getElementById('placeholder');
const runsList = document.getElementById('runsList');
const runBtn = document.getElementById('runBtn');

function showViewer(runId) {
  frame.src = '/viewer/' + runId;
  frame.classList.remove('hidden');
  placeholder.style.display = 'none';
}

function loadRuns(autoLoad) {
  fetch('/api/runs').then(r => r.json()).then(runs => {
    runsList.innerHTML = '';
    runs.forEach(r => {
      const li = document.createElement('li');
      li.innerHTML = `<div class="name">${r.source_name} (${r.input_kind})</div>
        <div class="meta">${r.n_timesteps} steps · ${r.duration_s}s · ${r.created}</div>`;
      li.onclick = () => showViewer(r.id);
      runsList.appendChild(li);
    });
    if (autoLoad && runs.length > 0) showViewer(runs[0].id);
  });
}

document.getElementById('uploadForm').onsubmit = async (e) => {
  e.preventDefault();
  runBtn.disabled = true;
  status.textContent = 'Uploading...';
  const formData = new FormData(e.target);
  try {
    const resp = await fetch('/api/predict', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
    } else {
      status.textContent = `Done! ${data.n_timesteps} timesteps, ${data.duration_s}s`;
      showViewer(data.run_id);
      loadRuns();
    }
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
  }
  runBtn.disabled = false;
};

// Poll status during prediction
async function pollStatus() {
  while (runBtn.disabled) {
    try {
      const r = await fetch('/api/status');
      const d = await r.json();
      if (d.message) status.textContent = d.message;
    } catch {}
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
}

loadRuns(true);
</script>
</body></html>"""


_current_status = {"message": "Ready"}


@app.route("/")
def index():
    return INDEX_HTML


@app.route("/api/runs")
def api_runs():
    return jsonify(list_runs())


@app.route("/api/status")
def api_status():
    return jsonify(_current_status)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global _current_status
    try:
        _current_status = {"message": "Processing upload..."}
        video_path = None
        audio_path = None
        text = request.form.get("text", "").strip()

        if "video" in request.files and request.files["video"].filename:
            f = request.files["video"]
            video_path = str(UPLOAD_DIR / f.filename)
            f.save(video_path)

        if "audio" in request.files and request.files["audio"].filename:
            f = request.files["audio"]
            audio_path = str(UPLOAD_DIR / f.filename)
            f.save(audio_path)

        if not video_path and not audio_path and not text:
            return jsonify({"error": "Upload a video, audio, or enter text"})

        _current_status = {"message": "Preparing events..."}
        if video_path:
            from neuralset.events.transforms import ExtractAudioFromVideo, ChunkEvents, AddText, AddSentenceToWords, AddContextToWords, RemoveMissing
            from neuralset.events.utils import standardize_events
            from tribev2.eventstransforms import ExtractWordsFromAudio
            events = pd.DataFrame([{"type": "Video", "filepath": video_path, "start": 0, "timeline": "default", "subject": "default"}])
            transforms = [
                ExtractAudioFromVideo(),
                ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30),
                ExtractWordsFromAudio(),
                AddText(), AddSentenceToWords(max_unmatched_ratio=0.05),
                AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""),
                RemoveMissing(),
            ]
            events = standardize_events(events)
            for t in transforms:
                events = t(events)
            df = standardize_events(events)
            input_kind = "video"
        elif audio_path:
            events = pd.DataFrame([{"type": "Audio", "filepath": audio_path, "start": 0, "timeline": "default", "subject": "default"}])
            df = get_audio_and_text_events(events)
            input_kind = "audio"
        else:
            df = build_text_events_from_text(text)
            input_kind = "text"

        _current_status = {"message": "Running inference..."}
        model = get_model()
        preds, segments = model.predict(events=df, verbose=True)

        n = preds.shape[0]
        if segments:
            try:
                video_duration = max(e.start + e.duration for seg in segments for e in seg.ns_events if hasattr(e, 'type') and e.type == 'Video')
                time_axis = [i * video_duration / n for i in range(n)]
            except ValueError:
                time_axis = [s.start for s in segments]
        else:
            time_axis = list(range(n))

        zone_df = build_zone_timeseries(preds, time_axis)
        source_name = Path(video_path or audio_path or "text").name
        run_id = save_run(preds, time_axis, zone_df, input_kind, source_name, video_path=video_path)

        duration_s = round(time_axis[-1] - time_axis[0], 1) if len(time_axis) > 1 else 0
        _current_status = {"message": "Ready"}
        return jsonify({"run_id": run_id, "n_timesteps": n, "duration_s": duration_s})

    except Exception as e:
        _current_status = {"message": "Ready"}
        log.exception("Prediction failed")
        return jsonify({"error": str(e)})


@app.route("/models/<path:filename>")
def serve_model(filename):
    return send_file(Path("models") / filename)


@app.route("/viewer/<run_id>")
def viewer(run_id):
    try:
        preds, time_axis, zone_df, meta = load_run(run_id)
        zone_dict = {col: zone_df[col].tolist() for col in zone_df.columns}
        html = build_viewer_html(
            preds, time_axis=time_axis, zone_data=zone_dict,
            video_path=meta.get("video_path"), height=500,
        )
        return html
    except Exception as e:
        return f"<h3>Error loading run: {e}</h3>", 404


if __name__ == "__main__":
    log.info("Starting TRIBE v2 Explorer...")
    log.info("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=True)
