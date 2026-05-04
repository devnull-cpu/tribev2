"""WebGL brain viewer using three.js + Plotly timeline — renders fsaverage5 mesh with vertex colors."""

import base64
import json
import struct
from pathlib import Path

import numpy as np
from nilearn import datasets, surface


def _load_mesh(mesh_type="pial"):
    """Load fsaverage5 left+right hemisphere mesh and sulcal depth."""
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    coords_l, faces_l = surface.load_surf_mesh(fs[f"{mesh_type}_left"])
    coords_r, faces_r = surface.load_surf_mesh(fs[f"{mesh_type}_right"])
    sulc_l = surface.load_surf_data(fs["sulc_left"])
    sulc_r = surface.load_surf_data(fs["sulc_right"])
    coords_r = coords_r.copy()
    coords_r[:, 0] += 2.0
    faces_r = faces_r + coords_l.shape[0]
    coords = np.concatenate([coords_l, coords_r], axis=0).astype(np.float32)
    faces = np.concatenate([faces_l, faces_r], axis=0).astype(np.uint32)
    sulc = np.concatenate([sulc_l, sulc_r]).astype(np.float32)
    center = coords.mean(axis=0)
    coords -= center
    return coords, faces, sulc


def _smooth_preds(preds, faces, n_verts, passes=3):
    """Laplacian smoothing of per-vertex predictions on the mesh."""
    from scipy import sparse
    rows, cols = [], []
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    rows.append(f[i]); cols.append(f[j])
    adj = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_verts, n_verts),
    )
    degree = np.array(adj.sum(axis=1)).ravel()
    degree[degree == 0] = 1
    smooth = preds.copy()
    for _ in range(passes):
        neighbor_sum = smooth @ adj.T  # (T, V) @ (V, V)^T
        smooth = (smooth + neighbor_sum) / (1.0 + degree[np.newaxis, :])
    return smooth.astype(np.float32)


def _pack_binary(coords, faces, sulc, preds):
    n_verts = coords.shape[0]
    n_faces = faces.shape[0]
    n_timesteps = preds.shape[0]
    header = struct.pack("<IIII", n_verts, n_faces, n_timesteps, 0)
    return (
        header
        + coords.tobytes()
        + faces.tobytes()
        + sulc.tobytes()
        + preds.astype(np.float32).tobytes()
    )


def build_viewer_html(
    preds: np.ndarray,
    time_axis: list[float] | None = None,
    zone_data: dict[str, list[float]] | None = None,
    video_path: str | None = None,
    mesh_type: str = "pial",
    height: int = 500,
) -> str:
    """Build a self-contained HTML string with three.js brain viewer + Plotly timeline.

    Parameters
    ----------
    preds : (n_timesteps, n_vertices) array
    time_axis : optional list of time values in seconds
    zone_data : optional dict of zone_name -> list of values per timestep
    video_path : optional path to video file for synced playback
    mesh_type : 'pial' or 'infl'
    height : viewer height in pixels
    """
    coords, faces, sulc = _load_mesh(mesh_type)
    preds = _smooth_preds(preds, faces, coords.shape[0], passes=3)
    blob = _pack_binary(coords, faces, sulc, preds)
    data_b64 = base64.b64encode(blob).decode("ascii")
    n_verts = coords.shape[0]
    n_faces = faces.shape[0]
    n_timesteps = preds.shape[0]

    if time_axis is None:
        time_axis = list(range(n_timesteps))
    times_json = json.dumps([round(t, 2) for t in time_axis])

    if zone_data is None:
        zone_data = {}
    zone_json = json.dumps({k: [round(v, 6) for v in vals] for k, vals in zone_data.items()})

    video_b64 = ""
    if video_path and Path(video_path).is_file():
        video_bytes = Path(video_path).read_bytes()
        video_b64 = base64.b64encode(video_bytes).decode("ascii")

    return _VIEWER_HTML.replace("__DATA_B64__", data_b64).replace(
        "__N_VERTS__", str(n_verts)
    ).replace("__N_FACES__", str(n_faces)).replace(
        "__N_TIMESTEPS__", str(n_timesteps)
    ).replace("__TIMES__", times_json).replace(
        "__ZONES__", zone_json
    ).replace("__VIDEO_B64__", video_b64).replace(
        "__HAS_VIDEO__", "true" if video_b64 else "false"
    ).replace("__HEIGHT__", str(height))


_VIEWER_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0a; font-family: -apple-system, system-ui, sans-serif; color: #fff; }

  #brain-container {
    position: relative; width: 100%; height: __HEIGHT__px;
    background: #080808; border-radius: 16px; overflow: hidden;
  }
  canvas { display: block; width: 100%; height: 100%; }

  #video-panel { display: none; width: 100%; }
  #video-panel.visible { display: flex; justify-content: center; align-items: center; padding: 12px 0; gap: 12px; }
  #video-panel video {
    max-width: 720px; width: 100%; max-height: 360px; object-fit: contain;
    border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);
  }
  #vol-control {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }
  #vol-control label { font-size: 11px; color: rgba(255,255,255,0.5); }
  #vol-control input[type=range] {
    -webkit-appearance: none; appearance: none; width: 80px; height: 4px;
    background: rgba(255,255,255,0.15); border-radius: 2px; outline: none;
  }
  #vol-control input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%;
    background: #fff; cursor: pointer;
  }
  #vol-control input[type=range]::-moz-range-thumb {
    width: 12px; height: 12px; border-radius: 50%;
    background: #fff; cursor: pointer; border: none;
  }

  #controls {
    padding: 12px 16px; display: flex; align-items: center; gap: 10px;
  }

  /* Styled range slider with orange thumb */
  #controls input[type=range] {
    flex: 1; height: 6px;
    -webkit-appearance: none; appearance: none;
    background: rgba(255,255,255,0.15); border-radius: 3px; outline: none;
    cursor: pointer; margin: 0 4px;
  }
  #controls input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 18px; height: 18px; border-radius: 50%;
    background: #ea580c; cursor: pointer; border: 2px solid #fff;
    margin-top: -6px;
  }
  #controls input[type=range]::-webkit-slider-runnable-track {
    height: 6px; border-radius: 3px;
  }
  #controls input[type=range]::-moz-range-thumb {
    width: 18px; height: 18px; border-radius: 50%;
    background: #ea580c; cursor: pointer; border: 2px solid #fff;
  }
  #controls input[type=range]::-moz-range-track {
    height: 6px; border-radius: 3px; background: rgba(255,255,255,0.15);
  }
  #controls label {
    font-size: 12px; white-space: nowrap; min-width: 70px;
    font-variant-numeric: tabular-nums; color: rgba(255,255,255,0.6);
  }

  /* Meta-style segmented pill toggle */
  .seg-control {
    position: relative; display: inline-flex; height: 34px; border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12); background: rgba(0,0,0,0.5);
    backdrop-filter: blur(12px); padding: 3px; gap: 0; overflow: hidden;
  }
  .seg-control .pill {
    position: absolute; top: 3px; bottom: 3px; border-radius: 8px;
    background: rgba(255,255,255,0.12); box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, width 0.2s ease;
  }
  .seg-control button {
    position: relative; z-index: 1; flex: 1; min-width: 56px; height: 100%;
    border: none; background: transparent; color: rgba(255,255,255,0.55);
    font-size: 12px; font-weight: 500; cursor: pointer; padding: 0 12px;
    transition: color 0.2s; white-space: nowrap;
  }
  .seg-control button:hover { color: rgba(255,255,255,0.8); }
  .seg-control button.active { color: #fff; }
  .seg-control button:active { transform: scale(0.98); }

  #play-btn {
    width: 36px; height: 36px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.08); color: #fff; font-size: 14px;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background 0.2s;
  }
  #play-btn:hover { background: rgba(255,255,255,0.15); }
  #play-btn.active { background: #ea580c; border-color: #ea580c; }

  #brain-overlay {
    position: absolute; bottom: 12px; left: 12px; right: 12px;
    display: flex; justify-content: center; gap: 8px; z-index: 20;
    pointer-events: auto;
  }
  #brain-overlay > * { pointer-events: auto; }

  #timeline { width: 100%; height: 180px; background: #0a0a0a; }
  #align-debug {
    position: absolute; top: 10px; right: 10px; z-index: 100;
    background: rgba(0,0,0,0.85); padding: 10px; border-radius: 8px;
    font-size: 11px; font-family: monospace; color: #ccc; max-height: 90%; overflow-y: auto;
  }
  #align-debug b { color: #ea580c; }
  #align-debug label { display: block; margin: 3px 0 1px; color: #888; font-size: 10px; }
  #align-debug input { width: 60px; background: #222; border: 1px solid #444; color: #eee; padding: 2px 4px; border-radius: 3px; font-size: 11px; margin-right: 2px; }
  #align-debug button { margin-top: 4px; padding: 3px 8px; font-size: 11px; border: none; border-radius: 4px; cursor: pointer; color: #fff; }
  .dbtn-apply { background: #ea580c; }
  .dbtn-print { background: #333; border: 1px solid #555 !important; }
</style>
</head><body>

<div id="brain-container">
  <div id="align-debug">
    <b>Brain</b>
    <label>Rot X/Y/Z (deg)</label>
    <input id="bRX" type="number" value="-110" step="5">
    <input id="bRY" type="number" value="0" step="5">
    <input id="bRZ" type="number" value="180" step="5">
    <hr style="border-color:#333;margin:6px 0">
    <b>Head</b>
    <label>Scale</label><input id="hScale" type="number" value="106" step="2">
    <label>Pos X/Y/Z</label>
    <input id="hX" type="number" value="0" step="2">
    <input id="hY" type="number" value="-69" step="2">
    <input id="hZ" type="number" value="-29" step="2">
    <label>Rot X/Y/Z (deg)</label>
    <input id="hRX" type="number" value="0" step="5">
    <input id="hRY" type="number" value="0" step="5">
    <input id="hRZ" type="number" value="0" step="5">
    <br>
    <button class="dbtn-apply" onclick="applyAlign()">Apply</button>
    <button class="dbtn-print" onclick="printAlign()">Print</button>
  </div>
  <div id="brain-overlay">
    <div class="seg-control" id="brainToggle">
      <div class="pill" style="width: calc(50% - 3px); transform: translateX(0);"></div>
      <button class="active" data-val="close">Close</button>
      <button data-val="open">Open</button>
    </div>
    <div class="seg-control" id="surfaceToggle">
      <div class="pill" style="width: calc(50% - 3px); transform: translateX(0);"></div>
      <button class="active" data-val="pial">Pial</button>
      <button data-val="inflated">Inflated</button>
    </div>
  </div>
</div>
<div id="video-panel">
  <video id="vidPlayer"></video>
  <div id="vol-control">
    <label>Vol</label>
    <input type="range" id="volSlider" min="0" max="1" step="0.05" value="0.5">
  </div>
</div>
<div id="controls">
  <button id="play-btn" title="Play/Pause">&#9654;</button>
  <label id="timeLabel">t=0.0s</label>
  <input type="range" id="slider" min="0" max="0" value="0" step="1">
</div>
<div id="timeline"></div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/* ── Constants from Python template ── */
const HAS_VIDEO   = __HAS_VIDEO__;
const N_VERTS     = __N_VERTS__;
const N_FACES     = __N_FACES__;
const N_TIMESTEPS = __N_TIMESTEPS__;
const TIMES       = __TIMES__;
const ZONES       = __ZONES__;
const HALF        = N_VERTS / 2;   // 10242 per hemisphere

/* ── Decode binary blob ── */
const raw = Uint8Array.from(atob("__DATA_B64__"), c => c.charCodeAt(0));
const buf = raw.buffer;
let off = 16; // skip 4x uint32 header
const coords = new Float32Array(buf, off, N_VERTS * 3); off += N_VERTS * 3 * 4;
const faces  = new Uint32Array(buf, off, N_FACES * 3);  off += N_FACES * 3 * 4;
const sulc   = new Float32Array(buf, off, N_VERTS);      off += N_VERTS * 4;
const preds  = new Float32Array(buf, off, N_TIMESTEPS * N_VERTS);

/* ── Pre-compute p99 normalization per timestep ── */
const p99s = new Float32Array(N_TIMESTEPS);
for (let t = 0; t < N_TIMESTEPS; t++) {
  const sl = preds.slice(t * N_VERTS, (t + 1) * N_VERTS);
  const sorted = Float32Array.from(sl).sort();
  p99s[t] = sorted[Math.floor(sorted.length * 0.99)] || 1;
}

/* ── Build per-face color atlas (predictions already smoothed in Python) ── */
const facesL = [], facesR = [];
for (let f = 0; f < N_FACES; f++) {
  const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
  if (a < HALF && b < HALF && c < HALF) facesL.push(a, b, c);
  else facesR.push(a - HALF, b - HALF, c - HALF);
}
const nFacesL = facesL.length / 3;
const nFacesR = facesR.length / 3;

function buildFaceAtlas(hemiFaces, nFaces, vertOffset) {
  const totalPixels = nFaces * N_TIMESTEPS;
  const W = Math.min(4096, totalPixels);
  const H = Math.ceil(totalPixels / W);
  const data = new Uint8Array(W * H * 4);
  for (let k = 3; k < data.length; k += 4) data[k] = 255;

  for (let t = 0; t < N_TIMESTEPS; t++) {
    const tOff = t * N_VERTS + vertOffset;
    const p99 = p99s[t];
    const vmin = p99 * 0.5;
    const invRange = 1.0 / (p99 - vmin + 1e-8);
    for (let f = 0; f < nFaces; f++) {
      const a = hemiFaces[f * 3], b = hemiFaces[f * 3 + 1], c = hemiFaces[f * 3 + 2];
      const val = (preds[tOff + a] + preds[tOff + b] + preds[tOff + c]) / 3.0;
      const norm = (val - vmin) * invRange;
      let r = 0, g = 0, bl = 0;
      if (norm > 0.01) {
        const n = norm > 1 ? 1 : norm;
        r = Math.min(1, n * 2.5);
        g = n > 0.4 ? Math.min(1, (n - 0.4) * 2.5) : 0;
        bl = n > 0.7 ? Math.min(1, (n - 0.7) * 3.33) : 0;
      }
      const px = (t * nFaces + f) * 4;
      data[px]     = (r * 255 + 0.5) | 0;
      data[px + 1] = (g * 255 + 0.5) | 0;
      data[px + 2] = (bl * 255 + 0.5) | 0;
    }
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
  tex.type = THREE.UnsignedByteType;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
  return { tex, W, H, numFaces: nFaces };
}

const atlasL = buildFaceAtlas(facesL, nFacesL, 0);
const atlasR = buildFaceAtlas(facesR, nFacesR, HALF);

/* ── Build face adjacency for smooth edge blending ── */
function buildFaceAdjacency(hemiFaces, nFaces) {
  const edgeMap = new Map();
  const adj = new Float32Array(nFaces * 3).fill(-1);
  for (let f = 0; f < nFaces; f++) {
    for (let e = 0; e < 3; e++) {
      const a = hemiFaces[f * 3 + e], b = hemiFaces[f * 3 + (e + 1) % 3];
      const key = a < b ? a * 100000 + b : b * 100000 + a;
      if (edgeMap.has(key)) {
        const [of, oe] = edgeMap.get(key);
        adj[f * 3 + e] = of;
        adj[of * 3 + oe] = f;
      } else {
        edgeMap.set(key, [f, e]);
      }
    }
  }
  // Pack into DataTexture: each pixel = (adj0, adj1, adj2, 0)
  const W = Math.min(4096, nFaces);
  const H = Math.ceil(nFaces / W);
  const data = new Float32Array(W * H * 4);
  for (let f = 0; f < nFaces; f++) {
    data[f * 4]     = adj[f * 3];
    data[f * 4 + 1] = adj[f * 3 + 1];
    data[f * 4 + 2] = adj[f * 3 + 2];
  }
  const tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.FloatType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
  return { tex, W, H };
}

const adjTexL = buildFaceAdjacency(facesL, nFacesL);
const adjTexR = buildFaceAdjacency(facesR, nFacesR);

/* ── Renderer setup ── */
const container = document.getElementById('brain-container');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setClearColor(0x080808);
renderer.toneMapping = THREE.NoToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.insertBefore(renderer.domElement, container.firstChild);

/* ── Scene + camera ── */
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  30, container.clientWidth / container.clientHeight, 0.01, 100
);
const ctl = new OrbitControls(camera, renderer.domElement);
ctl.enableDamping = true;
ctl.dampingFactor = 0.08;
ctl.rotateSpeed = 0.6;

/* ── Meta-style lighting ── */
scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const hemi = new THREE.HemisphereLight(0xffffff, 0x666666, 1.2);
hemi.position.set(0, 20, 0);
scene.add(hemi);

const d1 = new THREE.DirectionalLight(0xffffff, 2.0);
d1.position.set(0.418, 16.199, 0.3);
scene.add(d1);

const d2 = new THREE.DirectionalLight(0xffffff, 1.2);
d2.position.set(-0.757, 13.219, 0.717);
scene.add(d2);

const d3 = new THREE.DirectionalLight(0xffffff, 0.8);
d3.position.set(-10.906, 2.009, 1.846);
scene.add(d3);

const d4 = new THREE.DirectionalLight(0xffffff, 0.5);
d4.position.set(6.167, 0.857, 7.803);
scene.add(d4);

const d5 = new THREE.DirectionalLight(0xffffff, 0.5);
d5.position.set(-2.017, 0.018, 6.124);
scene.add(d5);

/* ── Load brain GLBs with baked AO vertex colors ── */
const gltfLoader = new GLTFLoader();
let meshL, meshR, hemiGroup, brainGroup;

function makeBrainMat(atlas, adjTex) {
  const faceUniforms = {
    uFaceTex:   { value: atlas.tex },
    uAtlasW:    { value: atlas.W },
    uAtlasH:    { value: atlas.H },
    uNumFaces:  { value: atlas.numFaces },
    uFrame0:    { value: 0 },
    uFrame1:    { value: 0 },
    uAlpha:     { value: 0 },
    uAdjTex:    { value: adjTex.tex },
    uAdjW:      { value: adjTex.W },
    uAdjH:      { value: adjTex.H },
  };
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.7,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, faceUniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
attribute float aFace;
attribute vec3 aBary;
flat varying float vFaceIndex;
varying vec3 vBary;
`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
vFaceIndex = aFace;
vBary = aBary;
`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
uniform sampler2D uFaceTex;
uniform float uAtlasW;
uniform float uAtlasH;
uniform float uNumFaces;
uniform float uFrame0;
uniform float uFrame1;
uniform float uAlpha;
uniform sampler2D uAdjTex;
uniform float uAdjW;
uniform float uAdjH;
flat varying float vFaceIndex;
varying vec3 vBary;

vec2 atlasUV(float face, float frame) {
  float f = clamp(face, 0.0, uNumFaces - 1.0);
  float idx = frame * uNumFaces + f;
  float x = mod(idx, uAtlasW);
  float y = floor(idx / uAtlasW);
  return vec2((x + 0.5) / uAtlasW, (y + 0.5) / uAtlasH);
}

vec3 sampleFace(float face) {
  if (face < 0.0) return vec3(0.0);
  vec3 a = texture2D(uFaceTex, atlasUV(face, uFrame0)).rgb;
  vec3 b = texture2D(uFaceTex, atlasUV(face, uFrame1)).rgb;
  return mix(a, b, uAlpha);
}
`)
      .replace('#include <map_fragment>', `
diffuseColor.rgb = pow(diffuseColor.rgb, vec3(1.5)) * 0.55;
vec3 myColor = sampleFace(vFaceIndex);

// Look up 3 neighbor faces from adjacency texture
float adjU = (vFaceIndex + 0.5) / uAdjW;
float adjRow = floor(vFaceIndex / uAdjW);
vec2 adjUV = vec2((mod(vFaceIndex, uAdjW) + 0.5) / uAdjW, (adjRow + 0.5) / uAdjH);
vec4 neighbors = texture2D(uAdjTex, adjUV);

// Blend near edges using barycentric coordinates
// Edge 0 (v0-v1): opposite v2 → bary.z small near this edge → neighbor[0]
// Edge 1 (v1-v2): opposite v0 → bary.x small near this edge → neighbor[1]
// Edge 2 (v2-v0): opposite v1 → bary.y small near this edge → neighbor[2]
float edgeBlend = 0.35;
vec3 blended = myColor;

float w0 = (1.0 - smoothstep(0.0, edgeBlend, vBary.z)) * step(0.0, neighbors.r);
float w1 = (1.0 - smoothstep(0.0, edgeBlend, vBary.x)) * step(0.0, neighbors.g);
float w2 = (1.0 - smoothstep(0.0, edgeBlend, vBary.y)) * step(0.0, neighbors.b);

if (w0 + w1 + w2 > 0.001) {
  vec3 n0 = sampleFace(neighbors.r);
  vec3 n1 = sampleFace(neighbors.g);
  vec3 n2 = sampleFace(neighbors.b);
  vec3 nAvg = (n0 * w0 + n1 * w1 + n2 * w2) / (w0 + w1 + w2);
  float wTotal = min(w0 + w1 + w2, 1.0);
  blended = mix(myColor, nAvg, wTotal * 0.5);
}

float faceAct = max(blended.r, max(blended.g, blended.b));
if (faceAct > 0.01) {
  float a = min(1.0, faceAct * 2.0);
  diffuseColor.rgb = mix(diffuseColor.rgb, blended, a);
  totalEmissiveRadiance += blended * smoothstep(0.3, 0.95, faceAct) * 0.5;
}
`);
  };
  mat.customProgramCacheKey = () => 'BrainFaceAtlas_v3';
  mat.__faceUniforms = faceUniforms;
  return mat;
}

function prepareGLBGeometry(gltf) {
  let srcMesh = null;
  gltf.scene.traverse((child) => { if (child.isMesh) srcMesh = child; });
  let geo = srcMesh.geometry;
  if (geo.index) geo = geo.toNonIndexed();
  if (!geo.getAttribute('normal')) geo.computeVertexNormals();
  const nVerts = geo.getAttribute('position').count;
  const nFaces = Math.floor(nVerts / 3);
  const faceAttr = new Float32Array(nVerts);
  const baryAttr = new Float32Array(nVerts * 3);
  for (let f = 0; f < nFaces; f++) {
    const i = f * 3;
    faceAttr[i] = f; faceAttr[i + 1] = f; faceAttr[i + 2] = f;
    baryAttr[i * 3]     = 1; baryAttr[i * 3 + 1] = 0; baryAttr[i * 3 + 2] = 0;
    baryAttr[i * 3 + 3] = 0; baryAttr[i * 3 + 4] = 1; baryAttr[i * 3 + 5] = 0;
    baryAttr[i * 3 + 6] = 0; baryAttr[i * 3 + 7] = 0; baryAttr[i * 3 + 8] = 1;
  }
  geo.setAttribute('aFace', new THREE.BufferAttribute(faceAttr, 1));
  geo.setAttribute('aBary', new THREE.BufferAttribute(baryAttr, 3));
  if (!geo.getAttribute('color')) {
    const colors = new Float32Array(nVerts * 3);
    colors.fill(0.15);
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }
  return geo;
}

function loadGLB(url) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, resolve, undefined, reject);
  });
}

Promise.all([
  loadGLB('/models/left.glb'),
  loadGLB('/models/right.glb'),
  loadGLB('/models/head.glb'),
]).then(([leftGltf, rightGltf, headGltf]) => {
  const geoL = prepareGLBGeometry(leftGltf);
  const geoR = prepareGLBGeometry(rightGltf);

  const matL = makeBrainMat(atlasL, adjTexL);
  const matR = makeBrainMat(atlasR, adjTexR);
  meshL = new THREE.Mesh(geoL, matL);
  meshR = new THREE.Mesh(geoR, matR);

  hemiGroup = new THREE.Group();
  hemiGroup.add(meshL);
  hemiGroup.add(meshR);
  brainGroup = new THREE.Group();
  brainGroup.add(hemiGroup);
  scene.add(brainGroup);

  /* ── Auto-fit camera to mesh bounds ── */
  const boxAll = new THREE.Box3().setFromObject(brainGroup);
  const size = new THREE.Vector3();
  boxAll.getSize(size);
  const center = new THREE.Vector3();
  boxAll.getCenter(center);

  brainGroup.position.sub(center);
  hemiGroup.rotation.x = -110 * Math.PI / 180;
  hemiGroup.rotation.z = Math.PI;

  const maxDim = Math.max(size.x, size.y, size.z);
  const fitDist = maxDim / (2 * Math.tan(Math.PI * camera.fov / 360));
  camera.position.set(0, 0, fitDist * 1.2);
  camera.near = fitDist * 0.01;
  camera.far = fitDist * 10;
  camera.updateProjectionMatrix();
  ctl.target.set(0, 0, 0);
  ctl.update();

  d1.position.set(maxDim, maxDim * 2, maxDim * 1.5);
  d2.position.set(-maxDim, -maxDim, -maxDim);
  d3.position.set(0, -maxDim * 2, maxDim);

  /* ── Head overlay ── */
  const headMesh = headGltf.scene;
  headMesh.traverse((child) => {
    if (!child.isMesh) return;
    child.material = new THREE.MeshStandardMaterial({
      color: 0xffffff, metalness: 0, roughness: 0.9,
      transparent: true, opacity: 0.06,
      depthWrite: false, depthTest: false,
      side: THREE.DoubleSide, envMapIntensity: 0,
    });
  });
  headMesh.scale.setScalar(106);
  headMesh.position.set(0, -69, -29);
  headMesh.rotation.set(0, 0, 0);
  brainGroup.add(headMesh);
  window._headMesh = headMesh;

  setTimestep(0);
  startAnimLoop();
});

window.applyAlign = () => {
  if (!hemiGroup) return;
  const d = Math.PI / 180;
  hemiGroup.rotation.set(
    parseFloat(document.getElementById('bRX').value) * d,
    parseFloat(document.getElementById('bRY').value) * d,
    parseFloat(document.getElementById('bRZ').value) * d
  );
  const h = window._headMesh;
  if (h) {
    h.scale.setScalar(parseFloat(document.getElementById('hScale').value));
    h.position.set(parseFloat(document.getElementById('hX').value), parseFloat(document.getElementById('hY').value), parseFloat(document.getElementById('hZ').value));
    h.rotation.set(parseFloat(document.getElementById('hRX').value) * d, parseFloat(document.getElementById('hRY').value) * d, parseFloat(document.getElementById('hRZ').value) * d);
  }
};
window.printAlign = () => {
  if (!hemiGroup) return;
  const d = 180 / Math.PI;
  const b = hemiGroup.rotation;
  let msg = `Brain rot: ${(b.x*d).toFixed(1)}, ${(b.y*d).toFixed(1)}, ${(b.z*d).toFixed(1)}`;
  const h = window._headMesh;
  if (h) {
    msg += `\nHead scale: ${h.scale.x.toFixed(1)}\nHead pos: ${h.position.x.toFixed(1)}, ${h.position.y.toFixed(1)}, ${h.position.z.toFixed(1)}\nHead rot: ${(h.rotation.x*d).toFixed(1)}, ${(h.rotation.y*d).toFixed(1)}, ${(h.rotation.z*d).toFixed(1)}`;
  }
  alert(msg);
};

/* ── Frame update: set face atlas uniforms ── */
function setFaceFrame(t, frac) {
  if (!meshL || !meshR) return;
  const tLo = Math.floor(t);
  const tHi = Math.min(N_TIMESTEPS - 1, tLo + 1);
  const alpha = frac !== undefined ? frac : 0;
  for (const m of [meshL, meshR]) {
    const u = m.material.__faceUniforms;
    if (!u) continue;
    u.uFrame0.value = tLo;
    u.uFrame1.value = tHi;
    u.uAlpha.value = alpha;
  }
}

function setTimestep(t) { setFaceFrame(t, 0); }
function setTimestepInterp(tLo, tHi, frac) { setFaceFrame(tLo, frac); }

/* ── Open/Close animation state ── */
let brainOpen = false;
let openAmount = 0;
let targetOpen = 0;

/* ── Segmented control logic ── */
function initSegControl(id, onChange) {
  const el = document.getElementById(id);
  const pill = el.querySelector('.pill');
  const btns = el.querySelectorAll('button');
  btns.forEach((btn, i) => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      pill.style.transform = `translateX(calc(${i * 100}% + ${i * 3}px))`;
      onChange(btn.dataset.val);
    });
  });
}

initSegControl('brainToggle', (val) => {
  brainOpen = val === 'open';
  targetOpen = brainOpen ? 1 : 0;
});

initSegControl('surfaceToggle', (val) => {
  // Placeholder for pial/inflated surface switching
});

/* ── Plotly timeline ── */
const zoneNames = Object.keys(ZONES);
const plotColors = [
  '#e63946', '#457b9d', '#2a9d8f', '#e9c46a',
  '#f4a261', '#264653', '#d62828', '#6a4c93',
  '#00b4d8', '#ff6b6b', '#48bfe3', '#72efdd',
];
const traces = zoneNames.map((name, i) => ({
  x: TIMES,
  y: ZONES[name],
  name,
  type: 'scatter',
  mode: 'lines',
  line: { color: plotColors[i % plotColors.length], width: 2 },
}));

const scrubLine = {
  type: 'line',
  x0: TIMES[0], x1: TIMES[0],
  y0: 0, y1: 1, yref: 'paper',
  line: { color: '#ea580c', width: 2, dash: 'dot' },
};

Plotly.newPlot('timeline', traces, {
  paper_bgcolor: '#0a0a0a',
  plot_bgcolor: '#0a0a0a',
  font: { color: '#aaa', size: 11 },
  margin: { l: 50, r: 20, t: 10, b: 35 },
  xaxis: { title: 'Time (s)', gridcolor: '#333', zerolinecolor: '#333' },
  yaxis: { title: 'Mean |activation|', gridcolor: '#333', zerolinecolor: '#333' },
  legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 10 } },
  shapes: [scrubLine],
  hovermode: false,
}, { responsive: true, displayModeBar: false });

/* ── Slider + controls ── */
const slider = document.getElementById('slider');
const timeLabel = document.getElementById('timeLabel');
const playBtn = document.getElementById('play-btn');
slider.max = N_TIMESTEPS - 1;

let lastRenderedTime = 0;

let plotUpdateTimer = 0;
function goToTimestep(t) {
  slider.value = t;
  setTimestep(t);
  timeLabel.textContent = `t=${TIMES[t]}s`;
  clearTimeout(plotUpdateTimer);
  plotUpdateTimer = setTimeout(() => {
    Plotly.relayout('timeline', { 'shapes[0].x0': TIMES[t], 'shapes[0].x1': TIMES[t] });
  }, 80);
}

slider.addEventListener('input', () => {
  const t = parseInt(slider.value);
  goToTimestep(t);
  lastRenderedTime = TIMES[t];
  const vid = document.getElementById('vidPlayer');
  if (vid && vid.src) vid.currentTime = TIMES[t];
  if (playing) {
    playStartWall = performance.now();
    playStartTime = TIMES[t];
  }
});

/* ── Click anywhere on timeline to jump ── */
function seekToTime(clickTime) {
  let closest = 0, minDist = Infinity;
  for (let i = 0; i < TIMES.length; i++) {
    const d = Math.abs(TIMES[i] - clickTime);
    if (d < minDist) { minDist = d; closest = i; }
  }
  goToTimestep(closest);
  lastRenderedTime = TIMES[closest];
  const vid = document.getElementById('vidPlayer');
  if (vid && vid.src) vid.currentTime = TIMES[closest];
  if (playing) {
    playStartWall = performance.now();
    playStartTime = TIMES[closest];
  }
}

const tlEl = document.getElementById('timeline');
function tlSeekFromEvent(e) {
  if (!tlEl._fullLayout) return;
  const xaxis = tlEl._fullLayout.xaxis;
  const rect = tlEl.querySelector('.main-svg').getBoundingClientRect();
  const clickTime = xaxis.p2d(e.clientX - rect.left - xaxis._offset);
  if (isFinite(clickTime)) seekToTime(clickTime);
}
let tlDragging = false;
tlEl.addEventListener('mousedown', (e) => {
  const svg = tlEl.querySelector('.main-svg');
  const plotArea = tlEl.querySelector('.draglayer');
  if (!plotArea || !plotArea.contains(e.target)) return;
  tlDragging = true;
  tlSeekFromEvent(e);
});
window.addEventListener('mousemove', (e) => {
  if (tlDragging) tlSeekFromEvent(e);
});
window.addEventListener('mouseup', () => { tlDragging = false; });

/* ── Play/pause with smooth requestAnimationFrame playback ── */
let playing = false;
let playRAF = null;
let playStartWall = 0;
let playStartTime = 0;
let lastPlotUpdate = 0;

function updateTimeline(ct) {
  const now = performance.now();
  if (now - lastPlotUpdate > 100) {
    lastPlotUpdate = now;
    let closest = 0;
    for (let i = 1; i < TIMES.length; i++) {
      if (Math.abs(TIMES[i] - ct) < Math.abs(TIMES[closest] - ct)) closest = i;
    }
    slider.value = closest;
    timeLabel.textContent = `t=${ct.toFixed(1)}s`;
    Plotly.relayout('timeline', { 'shapes[0].x0': ct, 'shapes[0].x1': ct });
  }
}

function playTick() {
  if (!playing) return;
  const elapsed = (performance.now() - playStartWall) / 1000;
  const currentTime = playStartTime + elapsed;
  const maxTime = TIMES[TIMES.length - 1];

  if (currentTime > maxTime) {
    playStartWall = performance.now();
    playStartTime = TIMES[0];
    playRAF = requestAnimationFrame(playTick);
    return;
  }

  let lo = 0, hi = 1;
  for (let i = 0; i < TIMES.length - 1; i++) {
    if (TIMES[i] <= currentTime && TIMES[i + 1] >= currentTime) {
      lo = i; hi = i + 1; break;
    }
    if (i === TIMES.length - 2) { lo = i; hi = i + 1; }
  }
  const span = TIMES[hi] - TIMES[lo];
  const frac = span > 0 ? (currentTime - TIMES[lo]) / span : 0;

  setTimestepInterp(lo, hi, frac);
  lastRenderedTime = currentTime;
  updateTimeline(currentTime);

  playRAF = requestAnimationFrame(playTick);
}

playBtn.addEventListener('click', () => {
  playing = !playing;
  playBtn.innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
  playBtn.classList.toggle('active', playing);

  if (playing) {
    playStartWall = performance.now();
    playStartTime = lastRenderedTime || TIMES[parseInt(slider.value)] || 0;
    playRAF = requestAnimationFrame(playTick);
  } else {
    if (playRAF) cancelAnimationFrame(playRAF);
  }
});

/* ── Video sync ── */
const vidPlayer = document.getElementById('vidPlayer');
const vidPanel = document.getElementById('video-panel');

if (HAS_VIDEO) {
  vidPanel.classList.add('visible');
  vidPlayer.src = 'data:video/mp4;base64,__VIDEO_B64__';
  vidPlayer.volume = 0.5;
  vidPlayer.load();

  document.getElementById('volSlider').addEventListener('input', (e) => {
    vidPlayer.volume = parseFloat(e.target.value);
  });

  function syncVideoToTimestep(t) {
    if (vidPlayer.readyState >= 2) {
      vidPlayer.currentTime = TIMES[t];
    }
  }

  // Wrap goToTimestep to also sync video
  const origGoToTimestep = goToTimestep;
  goToTimestep = function(t) {
    origGoToTimestep(t);
    syncVideoToTimestep(t);
  };

  // Play/pause video alongside brain
  playBtn.addEventListener('click', () => {
    if (playing) {
      vidPlayer.play();
    } else {
      vidPlayer.pause();
    }
  });

  // Sync video to follow playTick's time source (not the other way around)
  function videoSyncLoop() {
    if (playing && vidPlayer.readyState >= 2) {
      const drift = Math.abs(vidPlayer.currentTime - lastRenderedTime);
      if (drift > 0.3) vidPlayer.currentTime = lastRenderedTime;
    }
    requestAnimationFrame(videoSyncLoop);
  }
  videoSyncLoop();
}

/* ── Animation loop (started after GLBs load) ── */
function startAnimLoop() {
  (function animate() {
    requestAnimationFrame(animate);
    if (meshL && meshR) {
      openAmount += (targetOpen - openAmount) * 0.06;
      const spread = openAmount * 110;
      meshL.position.x = -spread;
      meshR.position.x = spread;
      meshL.rotation.z = openAmount * Math.PI / 2;
      meshR.rotation.z = -openAmount * Math.PI / 2;
      if (window._headMesh) {
        window._headMesh.visible = openAmount < 0.99;
        window._headMesh.traverse((child) => {
          if (child.isMesh) child.material.opacity = 0.06 * (1 - openAmount);
        });
      }
    }
    ctl.update();
    renderer.render(scene, camera);
  })();
}

/* ── Resize handler ── */
window.addEventListener('resize', () => {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  Plotly.Plots.resize('timeline');
});
</script>
</body></html>"""
