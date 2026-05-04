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
  #video-panel.visible { display: flex; justify-content: center; padding: 12px 0; }
  #video-panel video {
    max-width: 720px; width: 100%; max-height: 360px; object-fit: contain;
    border-radius: 12px; border: 1px solid rgba(255,255,255,0.08);
  }

  #controls {
    padding: 12px 16px; display: flex; align-items: center; gap: 10px;
  }

  /* Styled range slider with orange thumb */
  #controls input[type=range] {
    flex: 1; height: 4px;
    -webkit-appearance: none; appearance: none;
    background: rgba(255,255,255,0.1); border-radius: 2px; outline: none;
  }
  #controls input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: #ea580c; cursor: pointer; border: 2px solid #fff;
  }
  #controls input[type=range]::-moz-range-thumb {
    width: 14px; height: 14px; border-radius: 50%;
    background: #ea580c; cursor: pointer; border: 2px solid #fff;
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
    <input id="bRX" type="number" value="-111" step="5">
    <input id="bRY" type="number" value="0" step="5">
    <input id="bRZ" type="number" value="180" step="5">
    <hr style="border-color:#333;margin:6px 0">
    <b>Head</b>
    <label>Scale</label><input id="hScale" type="number" value="101" step="2">
    <label>Pos X/Y/Z</label>
    <input id="hX" type="number" value="0" step="2">
    <input id="hY" type="number" value="-69" step="2">
    <input id="hZ" type="number" value="0" step="2">
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
<div id="video-panel"><video id="vidPlayer" muted></video></div>
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

/* ── Pre-compute sulcal base colors ── */
const sulcBase = new Float32Array(N_VERTS * 3);
for (let i = 0; i < N_VERTS; i++) {
  // Sulcal depth as fake AO: deeper sulci (positive) = darker
  const s = sulc[i];
  const ao = 1.0 - Math.max(0, Math.min(1, s * 0.3 + 0.2));
  const g = (s > 0 ? 0.10 : 0.22) * ao;
  sulcBase[i * 3] = g;
  sulcBase[i * 3 + 1] = g;
  sulcBase[i * 3 + 2] = g;
}

/* ── Split binary data into L/R hemisphere arrays ── */
function splitHemisphere(fullCoords, fullFaces, startVert, numVerts, totalFaces) {
  // Extract coords for this hemisphere
  const hCoords = new Float32Array(numVerts * 3);
  for (let i = 0; i < numVerts * 3; i++) {
    hCoords[i] = fullCoords[startVert * 3 + i];
  }
  // Extract faces that belong to this hemisphere (vertex indices in [startVert, startVert+numVerts))
  const tempFaces = [];
  for (let f = 0; f < totalFaces; f++) {
    const a = fullFaces[f * 3], b = fullFaces[f * 3 + 1], c = fullFaces[f * 3 + 2];
    if (a >= startVert && a < startVert + numVerts &&
        b >= startVert && b < startVert + numVerts &&
        c >= startVert && c < startVert + numVerts) {
      tempFaces.push(a - startVert, b - startVert, c - startVert);
    }
  }
  const hFaces = new Uint32Array(tempFaces);
  return { coords: hCoords, faces: hFaces };
}

const hemiL = splitHemisphere(coords, faces, 0, HALF, N_FACES);
const hemiR = splitHemisphere(coords, faces, HALF, HALF, N_FACES);

/* ── Build three.js BufferGeometry from binary data ── */
function buildGeometry(hCoords, hFaces) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(hCoords, 3));
  geo.setIndex(new THREE.BufferAttribute(hFaces, 1));
  // Allocate vertex color buffer
  const colors = new Float32Array(hCoords.length);
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  return geo;
}

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

/* ── Build brain meshes from binary data ── */
const geoL = buildGeometry(new Float32Array(hemiL.coords), new Uint32Array(hemiL.faces));
const geoR = buildGeometry(new Float32Array(hemiR.coords), new Uint32Array(hemiR.faces));

function makeBrainMat() {
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.7,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });
  // Inject per-vertex emissive from the vertex color's brightness
  mat.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <emissivemap_fragment>',
      `#include <emissivemap_fragment>
       // Use vertex color brightness as emissive glow
       float activation = max(vColor.r, max(vColor.g, vColor.b));
       float emStr = smoothstep(0.3, 0.95, activation) * 0.5;
       totalEmissiveRadiance += vColor.rgb * emStr;`
    );
  };
  return mat;
}
const brainMat = makeBrainMat();

const meshL = new THREE.Mesh(geoL, brainMat);
const meshR = new THREE.Mesh(geoR, makeBrainMat());

const hemiGroup = new THREE.Group();
hemiGroup.add(meshL);
hemiGroup.add(meshR);
const brainGroup = new THREE.Group();
brainGroup.add(hemiGroup);
scene.add(brainGroup);

/* ── Auto-fit camera to mesh bounds ── */
const boxAll = new THREE.Box3().setFromObject(brainGroup);
const size = new THREE.Vector3();
boxAll.getSize(size);
const center = new THREE.Vector3();
boxAll.getCenter(center);

brainGroup.position.sub(center);
// Rotate brain hemispheres from RAS to screen: face forward
hemiGroup.rotation.x = -111 * Math.PI / 180;
hemiGroup.rotation.z = Math.PI;

const maxDim = Math.max(size.x, size.y, size.z);
const fitDist = maxDim / (2 * Math.tan(Math.PI * camera.fov / 360));
camera.position.set(0, 0, fitDist * 1.2);
camera.near = fitDist * 0.01;
camera.far = fitDist * 10;
camera.updateProjectionMatrix();
ctl.target.set(0, 0, 0);
ctl.update();

/* ── Scale directional lights to match mesh size ── */
d1.position.set(maxDim, maxDim * 2, maxDim * 1.5);
d2.position.set(-maxDim, -maxDim, -maxDim);
d3.position.set(0, -maxDim * 2, maxDim);

/* ── Load transparent head overlay ── */
const gltfLoader = new GLTFLoader();
gltfLoader.load('/models/head.glb', (headGltf) => {
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
  // The head GLB and brain GLB were made together by Meta.
  // Use Meta's original transform for the head, then undo brain rotation.
  headMesh.scale.setScalar(101);
  headMesh.position.set(0, -69, 0);
  headMesh.rotation.set(0, 0, 0);
  brainGroup.add(headMesh);
  window._headMesh = headMesh;
});

window.applyAlign = () => {
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
  const d = 180 / Math.PI;
  const b = hemiGroup.rotation;
  let msg = `Brain rot: ${(b.x*d).toFixed(1)}, ${(b.y*d).toFixed(1)}, ${(b.z*d).toFixed(1)}`;
  const h = window._headMesh;
  if (h) {
    msg += `\nHead scale: ${h.scale.x.toFixed(1)}\nHead pos: ${h.position.x.toFixed(1)}, ${h.position.y.toFixed(1)}, ${h.position.z.toFixed(1)}\nHead rot: ${(h.rotation.x*d).toFixed(1)}, ${(h.rotation.y*d).toFixed(1)}, ${(h.rotation.z*d).toFixed(1)}`;
  }
  alert(msg);
};

/* ── Color update: write vertex colors for both hemispheres ── */
function updateColors(tLo, tHi, frac) {
  const cL = geoL.getAttribute('color').array;
  const cR = geoR.getAttribute('color').array;
  const oLo = tLo * N_VERTS;
  const oHi = tHi * N_VERTS;
  const p99 = p99s[tLo] * (1 - frac) + p99s[tHi] * frac;
  const vmin = p99 * 0.5;
  const invRange = 1.0 / (p99 - vmin + 1e-8);
  const ifrac = 1 - frac;

  // Left hemisphere: vertices [0, HALF)
  for (let i = 0; i < HALF; i++) {
    const val = preds[oLo + i] * ifrac + preds[oHi + i] * frac;
    const norm = (val - vmin) * invRange;
    if (norm > 0.01) {
      const n = norm > 1 ? 1 : norm;
      const r = n * 2.5 > 1 ? 1 : n * 2.5;
      const g = n > 0.4 ? ((n - 0.4) * 2.5 > 1 ? 1 : (n - 0.4) * 2.5) : 0;
      const b = n > 0.7 ? ((n - 0.7) * 3.33 > 1 ? 1 : (n - 0.7) * 3.33) : 0;
      const a = n * 2 > 1 ? 1 : n * 2;
      const ia = 1 - a;
      cL[i * 3]     = sulcBase[i * 3]     * ia + r * a;
      cL[i * 3 + 1] = sulcBase[i * 3 + 1] * ia + g * a;
      cL[i * 3 + 2] = sulcBase[i * 3 + 2] * ia + b * a;
    } else {
      cL[i * 3]     = sulcBase[i * 3];
      cL[i * 3 + 1] = sulcBase[i * 3 + 1];
      cL[i * 3 + 2] = sulcBase[i * 3 + 2];
    }
  }

  // Right hemisphere: vertices [HALF, N_VERTS)
  for (let i = 0; i < HALF; i++) {
    const vi = HALF + i;  // index into global pred/sulcBase arrays
    const val = preds[oLo + vi] * ifrac + preds[oHi + vi] * frac;
    const norm = (val - vmin) * invRange;
    if (norm > 0.01) {
      const n = norm > 1 ? 1 : norm;
      const r = n * 2.5 > 1 ? 1 : n * 2.5;
      const g = n > 0.4 ? ((n - 0.4) * 2.5 > 1 ? 1 : (n - 0.4) * 2.5) : 0;
      const b = n > 0.7 ? ((n - 0.7) * 3.33 > 1 ? 1 : (n - 0.7) * 3.33) : 0;
      const a = n * 2 > 1 ? 1 : n * 2;
      const ia = 1 - a;
      cR[i * 3]     = sulcBase[vi * 3]     * ia + r * a;
      cR[i * 3 + 1] = sulcBase[vi * 3 + 1] * ia + g * a;
      cR[i * 3 + 2] = sulcBase[vi * 3 + 2] * ia + b * a;
    } else {
      cR[i * 3]     = sulcBase[vi * 3];
      cR[i * 3 + 1] = sulcBase[vi * 3 + 1];
      cR[i * 3 + 2] = sulcBase[vi * 3 + 2];
    }
  }

  geoL.getAttribute('color').needsUpdate = true;
  geoR.getAttribute('color').needsUpdate = true;
}

function setTimestep(t) { updateColors(t, t, 0); }
function setTimestepInterp(tLo, tHi, frac) { updateColors(tLo, tHi, frac); }

/* ── Open/Close animation state ── */
let brainOpen = false;
let openAmount = 0;
let targetOpen = 0;
const OPEN_DIST = 0.05;

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
  hovermode: 'x unified',
}, { responsive: true, displayModeBar: false });

/* ── Slider + controls ── */
const slider = document.getElementById('slider');
const timeLabel = document.getElementById('timeLabel');
const playBtn = document.getElementById('play-btn');
slider.max = N_TIMESTEPS - 1;

let lastRenderedTime = 0;

function goToTimestep(t) {
  slider.value = t;
  setTimestep(t);
  timeLabel.textContent = `t=${TIMES[t]}s`;
  Plotly.relayout('timeline', { 'shapes[0].x0': TIMES[t], 'shapes[0].x1': TIMES[t] });
}

slider.addEventListener('input', () => {
  const t = parseInt(slider.value);
  goToTimestep(t);
  lastRenderedTime = TIMES[t];
});

/* ── Click on timeline to jump ── */
document.getElementById('timeline').on('plotly_click', (data) => {
  if (!data.points.length) return;
  const clickTime = data.points[0].x;
  let closest = 0, minDist = Infinity;
  for (let i = 0; i < TIMES.length; i++) {
    const d = Math.abs(TIMES[i] - clickTime);
    if (d < minDist) { minDist = d; closest = i; }
  }
  goToTimestep(closest);
  lastRenderedTime = TIMES[closest];
});

/* ── Play/pause with smooth requestAnimationFrame playback ── */
let playing = false;
let playRAF = null;
let playStartWall = 0;
let playStartTime = 0;

function playTick() {
  if (!playing) return;
  const elapsed = (performance.now() - playStartWall) / 1000;
  const currentTime = playStartTime + elapsed;
  const maxTime = TIMES[TIMES.length - 1];

  if (currentTime > maxTime) {
    // Loop back to start
    playStartWall = performance.now();
    playStartTime = TIMES[0];
    playRAF = requestAnimationFrame(playTick);
    return;
  }

  // Find surrounding timesteps for interpolation
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
  slider.value = lo;
  timeLabel.textContent = `t=${currentTime.toFixed(1)}s`;
  Plotly.relayout('timeline', { 'shapes[0].x0': currentTime, 'shapes[0].x1': currentTime });

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
    // Snap to nearest discrete timestep on pause
    let closest = 0, minD = Infinity;
    for (let i = 0; i < TIMES.length; i++) {
      const d = Math.abs(TIMES[i] - lastRenderedTime);
      if (d < minD) { minD = d; closest = i; }
    }
    slider.value = closest;
    setTimestep(closest);
    timeLabel.textContent = `t=${TIMES[closest]}s`;
    Plotly.relayout('timeline', { 'shapes[0].x0': TIMES[closest], 'shapes[0].x1': TIMES[closest] });
  }
});

/* ── Video sync ── */
const vidPlayer = document.getElementById('vidPlayer');
const vidPanel = document.getElementById('video-panel');

if (HAS_VIDEO) {
  vidPanel.classList.add('visible');
  vidPlayer.src = 'data:video/mp4;base64,__VIDEO_B64__';
  vidPlayer.load();

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

  // When video is playing, use it as the time source
  function videoSyncLoop() {
    if (playing && !vidPlayer.paused) {
      const ct = vidPlayer.currentTime;
      let lo = 0, hi = 1;
      for (let i = 0; i < TIMES.length - 1; i++) {
        if (TIMES[i] <= ct && TIMES[i + 1] >= ct) { lo = i; hi = i + 1; break; }
        if (i === TIMES.length - 2) { lo = i; hi = i + 1; }
      }
      const span = TIMES[hi] - TIMES[lo];
      const frac = span > 0 ? (ct - TIMES[lo]) / span : 0;
      setTimestepInterp(lo, hi, frac);
      lastRenderedTime = ct;
      slider.value = lo;
      timeLabel.textContent = `t=${ct.toFixed(1)}s`;
      Plotly.relayout('timeline', { 'shapes[0].x0': ct, 'shapes[0].x1': ct });
    }
    requestAnimationFrame(videoSyncLoop);
  }
  videoSyncLoop();
}

/* ── Initial render ── */
setTimestep(0);

/* ── Animation loop ── */
(function animate() {
  requestAnimationFrame(animate);

  openAmount += (targetOpen - openAmount) * 0.08;
  meshL.position.x = -openAmount * OPEN_DIST;
  meshR.position.x = openAmount * OPEN_DIST;
  meshL.rotation.y = -openAmount * 0.6;
  meshR.rotation.y = openAmount * 0.6;

  ctl.update();
  renderer.render(scene, camera);
})();

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
