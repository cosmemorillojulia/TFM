"""Inferencia de la posicion de la pelota con TrackNet.

SOLO INFERENCIA: carga el modelo ya entrenado desde ``outputs/models/
tracknet_best.pth`` y predice la posicion de la pelota frame a frame en un clip.
Aqui no hay nada de entrenamiento; el notebook de entrenamiento queda archivado
en ``old_notebooks/02-0_model_ball_train.ipynb``.

Para cada frame se apilan los ``N_INPUT_FRAMES`` ultimos frames (ventana
deslizante) como tensor de 9 canales a 640x360, se predice el heatmap y se toma
el pixel de maxima activacion como posicion de la pelota (si supera el umbral).
Las coordenadas se devuelven en la resolucion original del frame. El bote real
se detecta aparte, a partir de la trayectoria (ver ``detect_real_bounces``).

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
        ``clip, frame, ball_x, ball_y, visibility``. En los frames sin pelota
        detectada (o en los primeros ``N_INPUT_FRAMES - 1``, que aun no
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
        })

    df = pd.DataFrame(rows)
    n_detected = int((df["visibility"] == 1).sum())
    logger.info("Pelota | %s | detectada en %d/%d frames", key, n_detected, len(frames))
    return df


def detect_real_bounces(df_ball, min_prominence=20, min_distance=8):
    """Detecta botes reales en el suelo por cambio de signo de la velocidad vertical.

    Un bote real es un frame donde la pelota pasa de bajar a subir bruscamente:
    ``ball_y`` (Y en píxeles, crece hacia abajo en la imagen) tiene velocidad
    positiva (bajando) justo antes y negativa (subiendo) justo después. Este
    criterio es más robusto que un simple máximo local de ``ball_y``: un golpe de
    raqueta en el aire suele frenar/redirigir la pelota de forma más gradual,
    mientras que el bote en el suelo produce un cambio de velocidad más abrupto
    (mayor prominencia).

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
    vy = np.diff(y)   # velocidad vertical entre frames consecutivos (visibles)

    last_bounce_idx = None
    for i in range(1, len(y) - 1):
        # Cambio de signo: bajando (vy>0) antes, subiendo (vy<0) despues.
        if not (vy[i - 1] > 0 and vy[i] < 0):
            continue

        # Prominencia: diferencia con el valle (menor ball_y) mas alto a cada lado.
        left_min  = y[:i].min()
        right_min = y[i + 1:].min() if i < len(y) - 1 else y[i]
        prominence = y[i] - max(left_min, right_min)
        if prominence < min_prominence:
            continue

        # Distancia minima al bote real anterior ya marcado.
        if last_bounce_idx is not None and (idx[i] - last_bounce_idx) < min_distance:
            continue

        df.at[idx[i], "is_real_bounce"] = 1
        last_bounce_idx = idx[i]

    n = int(df["is_real_bounce"].sum())
    logger.info("Botes reales detectados: %d", n)
    return df


def classify_bounce_location(df_ball, court_width, court_length, margin=0.0):
    """Clasifica cada bote real como dentro o fuera de las lineas de pista.

    Requiere que ``df_ball`` ya tenga las columnas ``ball_real_x``/``ball_real_y``
    (proyeccion a metros) y ``is_real_bounce``. Un bote es "dentro" si su punto
    cae en ``[0, court_width] x [0, court_length]`` (mas el margen opcional);
    en caso contrario es "fuera" (winner/out). Los frames que no son bote real
    quedan con ``bounce_in`` en NaN.

    Args:
        df_ball: DataFrame con ``ball_real_x``, ``ball_real_y``, ``is_real_bounce``.
        court_width, court_length: dimensiones de la pista en metros.
        margin: tolerancia en metros para contar como "dentro" (por defecto 0,
            es decir, exactamente las lineas de pista).

    Returns:
        Copia del DataFrame con la columna ``bounce_in`` (1=dentro, 0=fuera,
        NaN=no es bote) añadida.
    """
    df = df_ball.copy()
    df["bounce_in"] = np.nan

    mask = df["is_real_bounce"] == 1
    if not mask.any():
        return df

    x = df.loc[mask, "ball_real_x"]
    y = df.loc[mask, "ball_real_y"]
    inside = (x >= -margin) & (x <= court_width + margin) & \
             (y >= -margin) & (y <= court_length + margin)
    df.loc[mask, "bounce_in"] = inside.astype(int)

    n_in = int(inside.sum())
    n_out = int((~inside).sum())
    logger.info("Botes clasificados | dentro: %d | fuera: %d", n_in, n_out)
    return df
