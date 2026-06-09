"""Seleccion interactiva de los 4 puntos de pista via navegador web.

Funciona en entornos headless (RunPod, Colab, SSH remoto) sin display X11.

Uso:
    python select_court_points.py --game game1 [--port 7860]

Abre http://<ip-publica>:<puerto> en tu navegador, haz clic en los 4 puntos
y pulsa "Guardar". El servidor escribe el JSON y se cierra solo.

Puntos en orden:
  1. Esquina inferior-izquierda  (bottom_left)
  2. Esquina inferior-derecha    (bottom_right)
  3. Esquina superior-derecha    (top_right)
  4. Esquina superior-izquierda  (top_left)
"""

import argparse
import base64
import json
import sys
import threading
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template_string, request

from config import (
    COURT_LENGTH,
    COURT_POINTS_FILENAME,
    COURT_TYPE,
    COURT_WIDTH,
    DATASET_ROOT,
    OUTPUTS_ROOT,
    REFERENCE_FRAME,
)

POINT_ORDER = ["bottom_left", "bottom_right", "top_right", "top_left"]
WORLD_POINTS = [
    [0.0, 0.0],
    [COURT_WIDTH, 0.0],
    [COURT_WIDTH, COURT_LENGTH],
    [0.0, COURT_LENGTH],
]

# ---------------------------------------------------------------------------
# HTML / JS de la interfaz
# ---------------------------------------------------------------------------
_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Selección de puntos de pista – {{ game }}</title>
<style>
  body { margin: 0; background: #1a1a2e; color: #eee; font-family: sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 20px; }
  h2   { margin-bottom: 6px; }
  #status { font-size: 1.1em; margin-bottom: 12px; min-height: 1.5em; color: #7ec8e3; }
  #canvas-wrap { position: relative; display: inline-block; cursor: crosshair; }
  canvas { display: block; max-width: 95vw; }
  button { margin-top: 14px; padding: 10px 32px; font-size: 1em; border: none;
           border-radius: 6px; cursor: pointer; }
  #btn-save   { background: #27ae60; color: #fff; }
  #btn-save:disabled { background: #555; cursor: default; }
  #btn-undo   { background: #e67e22; color: #fff; margin-right: 8px; }
  #loaded-pts { font-size: .85em; color: #aaa; margin-top: 8px; white-space: pre; }
</style>
</head>
<body>
<h2>Selección de puntos de pista &mdash; {{ game }}</h2>
<div id="status">Haz clic en el punto <strong>1: bottom_left (inf-izq)</strong></div>
<div id="canvas-wrap">
  <canvas id="c"></canvas>
</div>
<div>
  <button id="btn-undo" onclick="undo()">↩ Deshacer</button>
  <button id="btn-save" disabled onclick="save()">Guardar ✓</button>
</div>
<div id="loaded-pts">{{ loaded_msg }}</div>

<script>
const ORDER  = ["bottom_left","bottom_right","top_right","top_left"];
const LABELS = ["1: inf-izq","2: inf-der","3: sup-der","4: sup-izq"];
const COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12"];
const pts = [];   // [{x, y}] en coordenadas de imagen original
let imgW = 0, imgH = 0, scale = 1;

const canvas = document.getElementById("c");
const ctx    = canvas.getContext("2d");
const status = document.getElementById("status");
const img    = new Image();

img.onload = () => {
  imgW = img.naturalWidth; imgH = img.naturalHeight;
  const maxW = Math.min(window.innerWidth * 0.95, 1280);
  scale = maxW / imgW;
  canvas.width  = imgW * scale;
  canvas.height = imgH * scale;
  redraw();
};
img.src = "{{ img_src }}";

canvas.addEventListener("click", e => {
  if (pts.length >= 4) return;
  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) / scale;
  const cy = (e.clientY - rect.top)  / scale;
  pts.push({x: cx, y: cy});
  updateStatus();
  redraw();
});

function undo() {
  if (pts.length === 0) return;
  pts.pop();
  updateStatus();
  redraw();
}

function updateStatus() {
  const n = pts.length;
  document.getElementById("btn-save").disabled = (n < 4);
  if (n < 4)
    status.innerHTML = `Haz clic en el punto <strong>${n+1}: ${ORDER[n]} (${LABELS[n].split(": ")[1]})</strong>`;
  else
    status.textContent = "4 puntos seleccionados. Pulsa Guardar.";
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  pts.forEach((p, i) => {
    const sx = p.x * scale, sy = p.y * scale;
    ctx.beginPath();
    ctx.arc(sx, sy, 8, 0, 2*Math.PI);
    ctx.fillStyle = COLORS[i];
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 13px sans-serif";
    ctx.fillText(LABELS[i], sx + 10, sy - 6);
  });
  if (pts.length >= 2) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x * scale, pts[0].y * scale);
    for (let i = 1; i < pts.length; i++)
      ctx.lineTo(pts[i].x * scale, pts[i].y * scale);
    if (pts.length === 4) ctx.closePath();
    ctx.strokeStyle = "rgba(255,255,0,0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

async function save() {
  const res = await fetch("/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({points: pts.map(p => [p.x, p.y])})
  });
  const data = await res.json();
  if (data.ok) {
    status.textContent = "✓ Guardado en " + data.path;
    document.getElementById("btn-save").disabled = true;
    document.getElementById("btn-undo").disabled = true;
  } else {
    status.textContent = "Error: " + data.error;
  }
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
_state: dict = {}  # game, court_file, img_b64, frame_w, frame_h


@app.route("/")
def index():
    existing = ""
    if _state["court_file"].exists():
        with open(_state["court_file"], encoding="utf-8") as f:
            data = json.load(f)
        pts = data["image_points"]
        existing = "Puntos ya guardados:\n" + "\n".join(
            f"  P{i+1} {POINT_ORDER[i]}: ({p[0]:.1f}, {p[1]:.1f})"
            for i, p in enumerate(pts)
        )
    return render_template_string(
        _HTML,
        game=_state["game"],
        img_src=f"data:image/jpeg;base64,{_state['img_b64']}",
        loaded_msg=existing,
    )


@app.route("/save", methods=["POST"])
def save():
    body = request.get_json(force=True)
    raw_pts = body.get("points", [])
    if len(raw_pts) != 4:
        return jsonify(ok=False, error=f"Se necesitan 4 puntos, llegaron {len(raw_pts)}")

    image_points = [[float(x), float(y)] for x, y in raw_pts]
    payload = {
        "cache_key": _state["game"],
        "reference_frame": REFERENCE_FRAME,
        "frame_size": [_state["frame_w"], _state["frame_h"]],
        "court_type": COURT_TYPE,
        "image_points": image_points,
        "world_points": WORLD_POINTS,
        "point_order": POINT_ORDER,
    }
    court_file: Path = _state["court_file"]
    court_file.parent.mkdir(parents=True, exist_ok=True)
    with open(court_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nGuardado en {court_file}")
    for i, (name, (x, y)) in enumerate(zip(POINT_ORDER, image_points), 1):
        print(f"  P{i} {name}: ({x:.1f}, {y:.1f})")

    # Parar el servidor tras responder (fuera del contexto de request)
    threading.Timer(1.0, _shutdown).start()
    return jsonify(ok=True, path=str(court_file))


def _shutdown():
    import os, signal
    os.kill(os.getpid(), signal.SIGINT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Selecciona los puntos de pista via navegador.")
    parser.add_argument("--game", required=True, help="Nombre del game, p.ej. game1")
    parser.add_argument("--port", type=int, default=7860, help="Puerto del servidor (default: 7860)")
    parser.add_argument("--output-dir", default=None,
                        help="Directorio de salida del game (default: outputs/<game>)")
    args = parser.parse_args()

    game = args.game
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_ROOT / game
    court_file = out_dir / COURT_POINTS_FILENAME

    # Cargar frame de referencia
    clip_dir = DATASET_ROOT / game / "Clip1"
    frame_path = clip_dir / REFERENCE_FRAME
    if not frame_path.exists():
        sys.exit(f"Frame de referencia no encontrado: {frame_path}")

    img_bgr = cv2.imread(str(frame_path))
    if img_bgr is None:
        sys.exit(f"No se pudo leer: {frame_path}")
    h, w = img_bgr.shape[:2]

    # Codificar imagen a base64 para servirla inline
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode()

    _state.update(game=game, court_file=court_file, img_b64=img_b64, frame_w=w, frame_h=h)

    print(f"Frame de referencia: {w} x {h}")
    if court_file.exists():
        print(f"Ya existe {court_file} (puedes sobreescribirlo desde el navegador).")
    print(f"\nAbre en tu navegador:  http://0.0.0.0:{args.port}")
    print("Haz clic en los 4 puntos y pulsa Guardar. El servidor se cerrará solo.\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
