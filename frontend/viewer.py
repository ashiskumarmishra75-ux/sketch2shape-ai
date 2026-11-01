"""Interactive Three.js viewer for the exported GLB.

`build_viewer_html()` returns a single self-contained page (GLB embedded as base64) with
orbit rotate / zoom / pan, reset camera, wireframe, lit/unlit material view (to inspect raw
vertex colours or textures), auto-rotate and a mesh-statistics overlay. The same page is
shown inside the app (iframe) and offered as a download that opens in any browser.
Three.js is loaded from the jsDelivr CDN.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Dict, Optional

THREE_VERSION = "0.170.0"
MAX_EMBED_MB = 25

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#f4f1ea; --ink:#1d1d1b; --panel:rgba(255,255,255,.86); --accent:#e4572e; }
  @media (prefers-color-scheme: dark) { :root { --bg:#16161a; --ink:#ecebe6; --panel:rgba(30,30,36,.86); } }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--ink);
              font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; overflow:hidden; }
  #c { width:100%; height:100%; display:block; touch-action:none; }
  .bar { position:absolute; left:10px; top:10px; display:flex; gap:6px; flex-wrap:wrap; }
  .bar button { background:var(--panel); color:var(--ink); border:1px solid rgba(127,127,127,.35);
                border-radius:7px; padding:5px 9px; cursor:pointer; font:inherit; }
  .bar button.on { border-color:var(--accent); color:var(--accent); }
  #stats { position:absolute; right:10px; bottom:10px; background:var(--panel); padding:6px 9px;
           border-radius:7px; white-space:pre; font-size:11px; }
  #msg { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; }
</style></head>
<body>
<canvas id="c"></canvas>
<div class="bar">
  <button id="reset">Reset view</button><button id="rotate">Auto-rotate</button>
  <button id="wire">Wireframe</button><button id="unlit">Unlit colours</button><button id="grid" class="on">Grid</button>
</div>
<div id="stats"></div><div id="msg">Loading 3D model…</div>
<script type="importmap">{"imports":{
  "three":"https://cdn.jsdelivr.net/npm/three@__V__/build/three.module.js",
  "three/addons/":"https://cdn.jsdelivr.net/npm/three@__V__/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
const GLB_B64 = "__GLB__";
const STATS = __STATS__;
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;   // keeps colours true (ACES darkens/desaturates reds)
renderer.toneMappingExposure = 1.0;
const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8478, 0.5));
const key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(2, 3, 2); scene.add(key);
const grid = new THREE.GridHelper(2, 20, 0x999999, 0xcccccc); grid.material.opacity = .45; grid.material.transparent = true;
scene.add(grid);
let model, litMats = new Map(), unlit = false, home = null;
function resize(){ const w = canvas.clientWidth, h = canvas.clientHeight; renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h,1); camera.updateProjectionMatrix(); }
addEventListener('resize', resize);
function frame(obj){ const box = new THREE.Box3().setFromObject(obj); const size = box.getSize(new THREE.Vector3());
  const c = box.getCenter(new THREE.Vector3()); const r = size.length() * 0.5 || 1;
  const d = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
  camera.position.copy(c).add(new THREE.Vector3(0.55, 0.38, 0.75).normalize().multiplyScalar(d));
  camera.near = d / 100; camera.far = d * 100; camera.updateProjectionMatrix();
  controls.target.copy(c); controls.update(); grid.position.y = box.min.y;
  home = { p: camera.position.clone(), t: c.clone() }; }
const bytes = Uint8Array.from(atob(GLB_B64), ch => ch.charCodeAt(0));
new GLTFLoader().parse(bytes.buffer, '', gltf => {
  model = gltf.scene; scene.add(model);
  model.traverse(o => { if (o.isMesh) { litMats.set(o, o.material); o.material.side = THREE.DoubleSide; } });
  resize(); frame(model); document.getElementById('msg').remove();
  const s = STATS; document.getElementById('stats').textContent =
    `vertices ${s.vertices ?? '?'}   faces ${s.faces ?? '?'}\ncolour: ${s.visual ?? '?'}   backend: ${s.backend ?? '?'}`;
}, err => { document.getElementById('msg').textContent = 'Could not load model: ' + err; });
const toggle = (id, fn) => { const b = document.getElementById(id); b.onclick = () => { b.classList.toggle('on'); fn(b.classList.contains('on')); }; };
document.getElementById('reset').onclick = () => { if (home) { camera.position.copy(home.p); controls.target.copy(home.t); controls.update(); } };
toggle('rotate', on => controls.autoRotate = on);
toggle('grid', on => grid.visible = on);
toggle('wire', on => litMats.forEach((m, o) => { o.material.wireframe = on; m.wireframe = on; }));
toggle('unlit', on => { unlit = on; litMats.forEach((m, o) => {
  if (on) { const b = new THREE.MeshBasicMaterial({ map: m.map || null, vertexColors: !!o.geometry.attributes.color,
            color: m.map || o.geometry.attributes.color ? 0xffffff : (m.color || 0xcccccc), side: THREE.DoubleSide, wireframe: m.wireframe });
            o.material = b; } else { o.material = m; } }); });
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
resize();
</script></body></html>
"""


def build_viewer_html(glb_path: Path, title: str = "Sketch2Shape model", stats: Optional[Dict] = None) -> str:
    glb_path = Path(glb_path)
    size_mb = glb_path.stat().st_size / 1024 ** 2
    if size_mb > MAX_EMBED_MB:
        raise ValueError(f"GLB is {size_mb:.1f} MB - too large to embed; open model.glb in a 3D viewer instead.")
    b64 = base64.b64encode(glb_path.read_bytes()).decode("ascii")
    return (_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__V__", THREE_VERSION)
            .replace("__STATS__", json.dumps(stats or {})).replace("__GLB__", b64))


def write_viewer(glb_path: Path, out_path: Path, title: str = "Sketch2Shape model", stats: Optional[Dict] = None) -> Path:
    out_path = Path(out_path)
    out_path.write_text(build_viewer_html(glb_path, title, stats), encoding="utf-8")
    return out_path


def iframe(page_html: str, height: int = 520) -> str:
    """Embeds the page in Gradio via iframe srcdoc (scripts run inside the iframe)."""
    return (f'<iframe srcdoc="{html.escape(page_html, quote=True)}" '
            f'style="width:100%;height:{height}px;border:0;border-radius:10px" '
            f'sandbox="allow-scripts" title="Three.js 3D viewer"></iframe>')
