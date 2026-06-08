"""Tracking de jugadores con YOLOv8 + ByteTrack.

Procesa la secuencia completa de un clip y devuelve las posiciones del punto de
pie de los dos jugadores frame a frame. El pipeline por clip es:

1. Listar y ordenar los frames del clip.
2. Submuestrear (1 de cada ``FRAME_STEP`` frames).
3. Deteccion + tracking con YOLOv8 (clase ``person``) y ByteTrack.
4. Filtro geometrico: bandas globales + zonas de exclusion (recogepelotas de
   red, jueces, arbitro).
5. Interpolacion lineal de los frames saltados.
6. Re-mapeo a etiquetas estables ``player_top`` / ``player_bottom``.

El modelo YOLO se carga una sola vez (cache de modulo) y se reutiliza para todos
los clips del game. Toda la logica de filtrado, interpolacion y etiquetado esta
portada de ``old_notebooks/01_player_tracking.ipynb``; las bandas calibradas a
1280x720 se reescalan a la resolucion real de cada clip.
"""

import logging
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import config
from src.data import loaders

logger = logging.getLogger(__name__)

# Columnas numericas que se interpolan en los frames saltados.
_NUM_COLS = ["foot_x", "foot_y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence"]

# Cache del modelo YOLO y del device, para no recargarlos en cada clip.
_model = None
_device = None


def _get_device():
    """Devuelve el device de inferencia ('cuda' si hay GPU, si no 'cpu')."""
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = "cuda"
        logger.info("GPU detectada para YOLO: %s", torch.cuda.get_device_name(0))
    else:
        _device = "cpu"
        warnings.warn(
            "No se ha detectado GPU. YOLO correra en CPU; considera reducir IMGSZ "
            "o aumentar FRAME_STEP si tarda demasiado.",
            stacklevel=2,
        )
        logger.info("YOLO ejecutandose en CPU.")
    return _device


def _get_model():
    """Carga (una sola vez) el modelo YOLO configurado en ``config``."""
    global _model
    if _model is None:
        _model = YOLO(config.YOLO_MODEL)
        logger.info("Modelo YOLO cargado: %s | clases COCO: %d",
                    config.YOLO_MODEL, len(_model.names))
    return _model


def _compute_bands(frame_w, frame_h):
    """Convierte las bandas y zonas de exclusion (fracciones) a pixeles.

    Args:
        frame_w, frame_h: resolucion real del clip.

    Returns:
        dict con limites en pixeles: ``x_min, x_max, y_min, y_max, y_mid`` y la
        lista ``exclusion_px`` de zonas ``[x0, y0, x1, y1]``.
    """
    bands = {
        "x_min": config.X_BAND_LEFT_FRAC * frame_w,
        "x_max": config.X_BAND_RIGHT_FRAC * frame_w,
        "y_min": config.Y_TOP_FRAC * frame_h,
        "y_max": config.Y_BOTTOM_FRAC * frame_h,
        "y_mid": config.MID_LINE_Y_FRAC * frame_h,
    }
    bands["exclusion_px"] = [
        [zx0 * frame_w, zy0 * frame_h, zx1 * frame_w, zy1 * frame_h]
        for (zx0, zy0, zx1, zy1) in config.EXCLUSION_ZONES
    ]
    return bands


def _select_two_players(xyxy, area_vals, bands):
    """Selecciona hasta dos jugadores (uno por mitad) de las detecciones de un frame.

    Filtra por bandas globales, zonas de exclusion y area minima; despues elige
    la deteccion de mayor area en la mitad superior y en la inferior (respecto a
    ``y_mid``). Portado de ``select_two_players`` del notebook 01.

    Args:
        xyxy: array (N, 4) de cajas [x1, y1, x2, y2].
        area_vals: array (N,) con el area de cada caja.
        bands: dict de limites en pixeles devuelto por ``_compute_bands``.

    Returns:
        Lista de indices (0, 1 o 2) de las detecciones seleccionadas.
    """
    foot_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    foot_y = xyxy[:, 3]

    valid = (
        (foot_x >= bands["x_min"]) & (foot_x <= bands["x_max"])
        & (foot_y >= bands["y_min"]) & (foot_y <= bands["y_max"])
    )
    for (zx0, zy0, zx1, zy1) in bands["exclusion_px"]:
        valid &= ~((foot_x >= zx0) & (foot_x <= zx1) & (foot_y >= zy0) & (foot_y <= zy1))
    valid &= (area_vals >= config.MIN_BBOX_AREA)

    if not valid.any():
        return []

    idxs = np.where(valid)[0]
    selected = []
    top = idxs[foot_y[idxs] < bands["y_mid"]]
    if len(top):
        selected.append(top[np.argmax(area_vals[top])])
    bot = idxs[foot_y[idxs] >= bands["y_mid"]]
    if len(bot):
        selected.append(bot[np.argmax(area_vals[bot])])
    return selected


def _interpolate_players(df_sampled, total_frames, clip_key):
    """Interpola linealmente las posiciones en los frames saltados.

    Reconstruye, por cada ``player_id``, una fila por frame del clip
    interpolando linealmente las columnas numericas. Portado de
    ``interpolate_players`` del notebook 01.

    Args:
        df_sampled: detecciones de los frames realmente procesados.
        total_frames: numero total de frames del clip.
        clip_key: clave "game/clip" para rellenar la columna ``clip``.

    Returns:
        ``DataFrame`` con una fila por (frame, player_id).
    """
    parts = []
    for pid in df_sampled["player_id"].unique():
        sub = (df_sampled[df_sampled["player_id"] == pid]
               .set_index("frame")
               .reindex(range(total_frames)))
        sub["clip"] = clip_key
        sub[_NUM_COLS] = sub[_NUM_COLS].interpolate(method="linear").ffill().bfill()
        sub["player_id"] = pid
        parts.append(sub.reset_index().rename(columns={"index": "frame"}))
    return (pd.concat(parts, ignore_index=True)
            .sort_values(["frame", "player_id"]).reset_index(drop=True))


def _relabel_top_bottom(df_players, y_mid):
    """Asigna etiquetas estables ``player_top`` / ``player_bottom``.

    Cada ``player_id`` se etiqueta segun la mitad de pista donde pasa mas tiempo
    (moda de su mitad). Despues se elimina cualquier duplicado por
    (frame, etiqueta) quedandose con la deteccion de mayor confianza. Portado del
    notebook 01.
    """
    df_players = df_players.copy()
    df_players["half"] = np.where(df_players["foot_y"] < y_mid, "top", "bottom")
    dominant_half = df_players.groupby("player_id")["half"].agg(lambda s: s.mode().iloc[0])
    id_to_label = {pid: f"player_{h}" for pid, h in dominant_half.items()}
    df_players["player_label"] = df_players["player_id"].map(id_to_label)

    df_players = (
        df_players
        .sort_values(["frame", "player_label", "confidence"], ascending=[True, True, False])
        .drop_duplicates(subset=["frame", "player_label"], keep="first")
        .reset_index(drop=True)
    )
    return df_players


def track_clip(clip_dir):
    """Trackea los dos jugadores en un clip completo.

    Args:
        clip_dir: ruta a la carpeta del clip.

    Returns:
        ``DataFrame`` con una fila por (frame, jugador) y columnas:
        ``clip, frame, player_id, player_label, foot_x, foot_y, bbox_x1,
        bbox_y1, bbox_x2, bbox_y2, confidence``. Vacio si no hay detecciones.
    """
    clip_dir = loaders.Path(clip_dir)
    key = loaders.clip_key(clip_dir)
    frames = loaders.list_frames(clip_dir)
    if not frames:
        logger.warning("Clip sin frames: %s", key)
        return pd.DataFrame()

    frame_h, frame_w = cv2.imread(str(frames[0])).shape[:2]
    bands = _compute_bands(frame_w, frame_h)
    logger.info("Jugadores | %s | %d frames | %dx%d", key, len(frames), frame_w, frame_h)

    model = _get_model()
    device = _get_device()

    # Submuestreo de frames.
    sampled_frames = frames[::config.FRAME_STEP]
    sampled_indices = list(range(0, len(frames), config.FRAME_STEP))

    results_iter = model.track(
        source=[str(p) for p in sampled_frames],
        classes=[config.PERSON_CLASS_ID],
        conf=config.CONF_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        imgsz=config.IMGSZ,
        tracker=config.TRACKER_CFG,
        device=device,
        persist=True,
        stream=True,
        verbose=False,
    )

    records = []
    for enum_idx, result in enumerate(results_iter):
        frame_idx = sampled_indices[enum_idx]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        area_vals = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

        for i in _select_two_players(xyxy, area_vals, bands):
            x1, y1, x2, y2 = xyxy[i]
            records.append({
                "clip": key,
                "frame": frame_idx,
                "player_id": int(ids[i]) if ids is not None else int(i),
                "foot_x": float((x1 + x2) / 2),
                "foot_y": float(y2),
                "bbox_x1": float(x1), "bbox_y1": float(y1),
                "bbox_x2": float(x2), "bbox_y2": float(y2),
                "confidence": float(conf[i]),
            })

    df_sampled = pd.DataFrame.from_records(records)
    if df_sampled.empty:
        logger.warning("Sin detecciones de jugadores en %s", key)
        return df_sampled

    df_players = _interpolate_players(df_sampled, len(frames), key)
    df_players = _relabel_top_bottom(df_players, bands["y_mid"])

    # Esquema final de columnas (orden estable para el CSV master).
    cols = ["clip", "frame", "player_id", "player_label",
            "foot_x", "foot_y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence"]
    return df_players[cols]
