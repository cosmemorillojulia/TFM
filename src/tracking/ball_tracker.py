"""Inferencia de la posicion de la pelota con TrackNet.

SOLO INFERENCIA: carga el modelo ya entrenado desde ``outputs/models/
tracknet_best.pth`` y predice la posicion de la pelota frame a frame en un clip.
Aqui no hay nada de entrenamiento; el notebook de entrenamiento queda archivado
en ``old_notebooks/02-0_model_ball_train.ipynb``.

Para cada frame se apilan los ``N_INPUT_FRAMES`` ultimos frames (ventana
deslizante) como tensor de 9 canales a 640x360, se predice el heatmap y se toma
el pixel de maxima activacion como posicion de la pelota (si supera el umbral).
Las coordenadas se devuelven en la resolucion original del frame. El flag
``is_bounce`` se toma de la columna ``status`` del ``Label.csv`` del clip, igual
que hacia el notebook 02.

Logica portada de ``old_notebooks/02_ball_tracking.ipynb`` (carga del checkpoint
e inferencia).
"""

import logging
import warnings

import cv2
import numpy as np
import pandas as pd
import torch

import config
from src.data import loaders
from src.models.tracknet import TrackNetV2

logger = logging.getLogger(__name__)

# Cache del modelo y device, para no recargar el checkpoint en cada clip.
_model = None
_device = None


def _get_device():
    """Devuelve el device de inferencia ('cuda' si hay GPU, si no 'cpu')."""
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        logger.info("GPU detectada para TrackNet: %s", torch.cuda.get_device_name(0))
    else:
        _device = torch.device("cpu")
        warnings.warn(
            "No se ha detectado GPU. La inferencia de TrackNet sera lenta en CPU.",
            stacklevel=2,
        )
        logger.info("TrackNet ejecutandose en CPU.")
    return _device


def _get_model():
    """Carga (una sola vez) el modelo TrackNet desde el checkpoint entrenado."""
    global _model
    if _model is None:
        if not config.BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encuentra el modelo entrenado en {config.BEST_MODEL_PATH}. "
                "Este proyecto no entrena: coloca alli el checkpoint 'tracknet_best.pth'."
            )
        device = _get_device()
        model = TrackNetV2(in_channels=3 * config.N_INPUT_FRAMES, out_channels=1).to(device)
        ckpt = torch.load(config.BEST_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model = model
        logger.info("Checkpoint TrackNet cargado (epoch %s, val_loss %.5f)",
                    ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))
    return _model


def _load_and_prep(path):
    """Carga un frame, lo pasa a RGB, lo redimensiona a 640x360 y normaliza a [0, 1].

    Returns:
        Tupla ``(img_norm, (orig_w, orig_h))`` con la imagen ``float32`` lista
        para apilar y la resolucion original del frame.
    """
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    img_r = cv2.resize(img, (config.INPUT_W, config.INPUT_H), interpolation=cv2.INTER_AREA)
    return img_r.astype(np.float32) / 255.0, (orig_w, orig_h)


def infer_clip(clip_dir):
    """Predice la posicion de la pelota en un clip completo.

    Args:
        clip_dir: ruta a la carpeta del clip.

    Returns:
        ``DataFrame`` con una fila por frame y columnas:
        ``clip, frame, ball_x, ball_y, visibility, is_bounce``. En los frames sin
        pelota detectada (o en los primeros ``N_INPUT_FRAMES - 1``, que aun no
        completan ventana) ``ball_x``/``ball_y`` van a NaN y ``visibility=0``.
    """
    clip_dir = loaders.Path(clip_dir)
    key = loaders.clip_key(clip_dir)
    frames = loaders.list_frames(clip_dir)
    if not frames:
        logger.warning("Clip sin frames: %s", key)
        return pd.DataFrame()

    model = _get_model()
    device = _get_device()
    logger.info("Pelota | %s | %d frames", key, len(frames))

    # Mapa file -> status para el flag de rebote (ground truth del dataset).
    labels = loaders.load_labels(clip_dir)
    labels["status"] = labels["status"].fillna(0).astype(int)
    status_map = dict(zip(labels["file name"], labels["status"]))

    rows = []
    window = []
    for idx, fp in enumerate(frames):
        img_r, (orig_w, orig_h) = _load_and_prep(fp)
        window.append(img_r)
        if len(window) > config.N_INPUT_FRAMES:
            window.pop(0)

        # Aun no hay suficientes frames para completar la ventana.
        if len(window) < config.N_INPUT_FRAMES:
            ball_x, ball_y, vis = np.nan, np.nan, 0
        else:
            x = np.concatenate([f.transpose(2, 0, 1) for f in window], axis=0)
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.no_grad():
                hm = model(x_t)[0, 0].cpu().numpy()
            peak = float(hm.max())
            if peak >= config.HEATMAP_THRESHOLD:
                yy, xx = np.unravel_index(np.argmax(hm), hm.shape)
                ball_x = float(xx) * (orig_w / config.INPUT_W)
                ball_y = float(yy) * (orig_h / config.INPUT_H)
                vis = 1
            else:
                ball_x, ball_y, vis = np.nan, np.nan, 0

        rows.append({
            "clip": key,
            "frame": idx,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "visibility": vis,
            "is_bounce": int(status_map.get(fp.name, 0)),
        })

    df = pd.DataFrame(rows)
    n_detected = int((df["visibility"] == 1).sum())
    logger.info("Pelota | %s | detectada en %d/%d frames", key, n_detected, len(frames))
    return df


def detect_real_bounces(df_ball, min_prominence=20, min_distance=8):
    """Detecta botes reales en el suelo por análisis de trayectoria.

    Un bote real es un máximo local prominente de ``ball_y`` (mayor Y en píxeles
    = posición más baja en imagen = pelota más cerca del suelo). Se ignoran los
    valores ``is_bounce`` del CSV original, que mezclan golpes con raqueta y botes.

    El resultado se añade como columna ``is_real_bounce`` (0/1) al DataFrame.

    Args:
        df_ball: DataFrame con columnas ``ball_x``, ``ball_y``, ``visibility``.
        min_prominence: mínima diferencia de ``ball_y`` entre el candidato y los
            valles adyacentes para considerarlo bote (px). Evita ruido.
        min_distance: separación mínima en frames entre dos botes consecutivos.

    Returns:
        Copia del DataFrame con la columna ``is_real_bounce`` añadida.
    """
    df = df_ball.copy()
    df["is_real_bounce"] = 0

    visible = df[df["visibility"] == 1].copy()
    if len(visible) < 3:
        return df

    y = visible["ball_y"].values
    idx = visible.index.values

    for i in range(1, len(y) - 1):
        # Máximo local: ball_y mayor que sus vecinos inmediatos
        if y[i] <= y[i - 1] or y[i] <= y[i + 1]:
            continue

        # Prominencia: diferencia con el valle más alto a cada lado
        left_min  = y[:i].min()  if i > 0          else y[i]
        right_min = y[i+1:].min() if i < len(y) - 1 else y[i]
        prominence = y[i] - max(left_min, right_min)
        if prominence < min_prominence:
            continue

        # Distancia mínima al bote real anterior ya marcado
        already = df[df["is_real_bounce"] == 1].index
        if len(already) > 0:
            if (idx[i] - already[-1]) < min_distance:
                continue

        df.at[idx[i], "is_real_bounce"] = 1

    n = int(df["is_real_bounce"].sum())
    logger.info("Botes reales detectados: %d", n)
    return df
