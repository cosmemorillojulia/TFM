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


def _clean_trajectory(frames, bx, by, max_jump_px):
    """Limpia outliers de la trayectoria de la pelota de un clip.

    El tracking de pelota (TrackNet) produce a veces saltos imposibles entre
    frames consecutivos (la pelota "teletransportada" cientos de px), que rompen
    cualquier regla basada en "el frame siguiente". Esta funcion marca como NO
    fiables los puntos cuyo salto respecto al frame previo visible supera
    ``max_jump_px`` (escalado a la resolucion del clip) y los reemplaza por
    interpolacion lineal entre los vecinos fiables.

    Args:
        frames: array de indices de frame (de los visibles, en orden).
        bx, by: arrays de ``ball_x`` / ``ball_y`` (px) de esos frames.
        max_jump_px: salto maximo plausible (px) entre dos frames visibles
            consecutivos; por encima se considera outlier.

    Returns:
        Tupla ``(bx_clean, by_clean)`` con los outliers interpolados.
    """
    n = len(frames)
    good = np.ones(n, dtype=bool)
    for i in range(1, n):
        # Normalizar el salto por la separacion temporal (puede haber huecos).
        step = max(1, int(frames[i] - frames[i - 1]))
        jump = np.hypot(bx[i] - bx[i - 1], by[i] - by[i - 1]) / step
        if jump > max_jump_px:
            good[i] = False

    bx_clean = bx.astype(float).copy()
    by_clean = by.astype(float).copy()
    if good.sum() >= 2:
        gi = np.where(good)[0]
        bx_clean = np.interp(np.arange(n), gi, bx[gi])
        by_clean = np.interp(np.arange(n), gi, by[gi])
    return bx_clean, by_clean


def _stays_same_court(court_y, i, net_y, lookahead, min_pts):
    """True si tras el evento en ``i`` la pelota NO se va al otro campo (es bote).

    Regla pedida: cuenta como bote si tras el evento la pelota sigue su
    trayectoria y no cambia de direccion para irse al otro campo. Se mide en
    coordenadas de PISTA (metros, eje largo ``court_y``), que es robusto y tiene
    sentido fisico: la red parte la pista en ``net_y``. Un bote ocurre en un
    campo y la pelota PERMANECE en ese campo los frames siguientes; un raquetazo
    la TRANSFIERE al otro campo (cruza la red de forma sostenida).

    Aplica por igual a botes en campo top y bottom (la comparacion es respecto a
    la red, no a un lado concreto).

    Args:
        court_y: array con la Y de pista (metros) de cada frame visible del clip.
        i: indice del extremo candidato.
        net_y: Y de la red en metros (mitad del largo de pista).
        lookahead: nº de frames visibles a mirar antes y despues.
        min_pts: nº minimo de puntos validos a cada lado para decidir; si no se
            alcanza, se acepta (no penalizar por falta de contexto).

    Returns:
        ``True`` si la pelota se queda en el mismo campo (bote), ``False`` si
        cruza al otro campo (raquetazo).
    """
    lo = max(0, i - lookahead)
    hi = min(len(court_y), i + lookahead + 1)
    before = court_y[lo:i]
    after = court_y[i + 1:hi]
    if len(before) < min_pts or len(after) < min_pts:
        return True
    # Lado de pista (campo) robusto a ruido: mediana a cada lado del evento.
    side_before = np.median(before) > net_y   # True = campo lejano (top)
    side_after = np.median(after) > net_y
    return side_before == side_after


def _local_prominence(y, i, kind="max"):
    """Prominencia LOCAL de un extremo de ``y`` en ``i``.

    Para ``kind="max"`` (bote en campo cercano: ``ball_y`` hace un maximo) mide
    cuanto sobresale el pico frente a los valles locales a cada lado. Para
    ``kind="min"`` (bote en campo lejano: ``ball_y`` hace un minimo, porque tras
    botar la pelota vuelve hacia la camara y su Y crece) mide lo simetrico.

    A diferencia de comparar contra el extremo global del array (que se diluye en
    clips largos con varios botes), camina hacia afuera a cada lado hasta el
    valle/cresta local mas cercano (definicion estandar de prominencia).
    """
    s = 1.0 if kind == "max" else -1.0   # trabajar siempre como si fuera un maximo
    yi = s * y[i]

    left_min = yi
    j = i - 1
    while j >= 0 and s * y[j] <= yi:
        left_min = min(left_min, s * y[j])
        j -= 1
    if j >= 0:
        left_min = min(left_min, s * y[j])

    right_min = yi
    j = i + 1
    while j < len(y) and s * y[j] <= yi:
        right_min = min(right_min, s * y[j])
        j += 1
    if j < len(y):
        right_min = min(right_min, s * y[j])

    return yi - max(left_min, right_min)


def _is_racket_hit(ball_x, ball_y, players_at_frame, radius, hit_zone_frac):
    """True si el candidato es, con mas probabilidad, un golpe de raqueta.

    Distingue golpe de bote por la ALTURA de la pelota respecto al cuerpo del
    jugador, no solo por proximidad. Un golpe ocurre cerca del jugador en X y a
    la altura del torso/cabeza/brazo (parte ALTA de la caja). Un bote en el
    suelo, aunque caiga cerca del jugador en X (tipico en el fondo de pista),
    esta a la altura de los PIES o por debajo: ahi NO se descarta.

    Se rechaza solo si la pelota:
      - cae en la columna del jugador en X (su bbox +/- ``radius``), y
      - esta por ENCIMA de la zona de pies, es decir su ``ball_y`` queda por
        encima de ``bbox_y2 - hit_zone_frac * altura_bbox`` (recordar: en
        pixeles, menor Y = mas arriba en la imagen).

    Asi, un bote a los pies del jugador del fondo (ball_y ~ bbox_y2) se conserva,
    mientras que un golpe a la altura del hombro (ball_y ~ bbox_y1) se descarta.
    """
    for p in players_at_frame:
        # Columna del jugador en X ampliada por ``radius``. Un raquetazo se da con
        # el brazo extendido, asi que la pelota puede caer algo fuera de la bbox.
        in_x = p["bbox_x1"] - radius <= ball_x <= p["bbox_x2"] + radius
        if not in_x:
            continue
        bbox_h = p["bbox_y2"] - p["bbox_y1"]
        # Umbral vertical: frontera entre "zona de golpeo" (arriba) y "zona de
        # pies/suelo" (abajo). Por encima de feet_zone_top => golpe.
        feet_zone_top = p["bbox_y2"] - hit_zone_frac * bbox_h
        # En la zona de golpeo (desde bastante por encima de la cabeza, para
        # cubrir globos/saques, hasta feet_zone_top) => raquetazo, no bote.
        if ball_y <= feet_zone_top:
            return True
    return False


def detect_real_bounces(df_ball, df_players=None, homography_matrix=None,
                        min_prominence=None, min_distance=None, player_exclusion_px=None,
                        hit_zone_frac=None, court_margin=None):
    """Detecta botes reales en el suelo en AMBOS campos (top y bottom).

    Un bote es un extremo local de la trayectoria vertical de la pelota:
      - Campo cercano: ``ball_y`` (px, crece hacia abajo) hace un MAXIMO local
        (la pelota baja y rebota hacia arriba).
      - Campo lejano: como la pelota se aleja de la camara, tras botar vuelve
        hacia ella y su Y crece; aparece como un MINIMO local de ``ball_y``.

    Antes de detectar, la trayectoria del clip se LIMPIA (se interpolan los
    saltos imposibles del tracking), porque la regla principal mira el frame
    siguiente y sin limpiar seria ruido.

    REGLA PRINCIPAL (pedida): un extremo cuenta como bote si tras el la pelota
    sigue su trayectoria y NO cambia de direccion para irse al otro campo. Se
    mide en coordenadas de pista (metros): la pelota debe quedarse en el MISMO
    lado de la red unos frames despues; si cruza al otro campo, es un raquetazo.
    Ademas se exige prominencia LOCAL del extremo y se descartan, como guardas
    secundarias, los golpes claros junto al jugador y los candidatos que
    proyectan fuera de la pista (ruido).

    La deteccion se hace POR CLIP de forma independiente y es GENERICA: funciona
    igual para cualquier numero de clips y cualquier game (no hay nada atado a
    clips concretos).

    El resultado se añade como columna ``is_real_bounce`` (0/1) al DataFrame.

    Args:
        df_ball: DataFrame con columnas ``clip``, ``frame``, ``ball_x``,
            ``ball_y``, ``visibility``.
        df_players: DataFrame de jugadores (columnas ``clip``, ``frame``,
            ``bbox_x1/y1/x2/y2``) para descartar golpes de raqueta por
            proximidad. Si es ``None`` o vacio, no se aplica ese filtro.
        min_prominence: minima diferencia de ``ball_y`` entre el candidato y los
            valles locales adyacentes para considerarlo bote (px). Por defecto
            ``config.BOUNCE_MIN_PROMINENCE``.
        min_distance: separacion minima en frames entre dos botes consecutivos
            del mismo clip. Por defecto ``config.BOUNCE_MIN_DISTANCE``.
        player_exclusion_px: margen (px) en X alrededor de la caja de un jugador
            para considerar el candidato "en su columna". Por defecto
            ``config.BOUNCE_PLAYER_EXCLUSION_PX``.
        hit_zone_frac: fraccion superior de la caja del jugador que se considera
            "zona de golpeo". Un candidato en la columna del jugador y por encima
            de ``bbox_y2 - hit_zone_frac*altura`` se descarta como golpe; a la
            altura de los pies se conserva como bote. Por defecto
            ``config.BOUNCE_HIT_ZONE_FRAC``.
        homography_matrix: necesaria para la regla principal (campo en metros) y
            para el filtro de fuera de pista. Si es ``None``, la regla de "mismo
            campo" y el filtro de pista se omiten (solo extremo + prominencia).
        court_margin: margen (metros) de tolerancia para el filtro de pista.
            Por defecto ``config.BOUNCE_IN_MARGIN_M``.

    Returns:
        Copia del DataFrame con la columna ``is_real_bounce`` añadida.
    """
    min_prominence = config.BOUNCE_MIN_PROMINENCE if min_prominence is None else min_prominence
    min_distance = config.BOUNCE_MIN_DISTANCE if min_distance is None else min_distance
    player_exclusion_px = (config.BOUNCE_PLAYER_EXCLUSION_PX if player_exclusion_px is None
                            else player_exclusion_px)
    hit_zone_frac = config.BOUNCE_HIT_ZONE_FRAC if hit_zone_frac is None else hit_zone_frac
    court_margin = config.BOUNCE_IN_MARGIN_M if court_margin is None else court_margin

    df = df_ball.copy()
    df["is_real_bounce"] = 0

    players_by_clip_frame = {}
    if df_players is not None and not df_players.empty:
        for (clip, frame), sub in df_players.groupby(["clip", "frame"]):
            players_by_clip_frame[(clip, frame)] = sub.to_dict("records")

    net_y = config.COURT_LENGTH / 2.0

    n_total = 0
    n_rejected_cross = 0
    n_rejected_player = 0
    n_rejected_offcourt = 0
    for clip, df_clip in df.groupby("clip"):
        visible = df_clip[df_clip["visibility"] == 1].sort_values("frame")
        if len(visible) < 3:
            continue

        frames = visible["frame"].values
        idx = visible.index.values

        # Resolucion del clip (primer frame) para escalar el umbral de outlier.
        # ``ball_x/y`` estan en px de la imagen original; el salto maximo plausible
        # se define como fraccion de la diagonal, robusto a distintas resoluciones.
        bx_raw = visible["ball_x"].values.astype(float)
        by_raw = visible["ball_y"].values.astype(float)
        diag = float(np.hypot(bx_raw.max() - bx_raw.min(), by_raw.max() - by_raw.min())) or 1.0
        max_jump = config.BOUNCE_MAX_JUMP_FRAC * diag

        # 1) Limpiar la trayectoria: sin esto, "el frame siguiente" es ruido y la
        #    regla de continuidad de avance no funciona.
        bx, y = _clean_trajectory(frames, bx_raw, by_raw, max_jump)
        vy = np.diff(y)   # velocidad vertical entre frames consecutivos (limpia)

        # 2) Y de pista (metros) de cada frame visible, para la regla de "mismo
        #    campo" y el filtro de fuera de pista (si hay homografia).
        if homography_matrix is not None:
            pts = np.stack([bx, y], axis=1).reshape(-1, 1, 2).astype(np.float64)
            court_xy = cv2.perspectiveTransform(pts, homography_matrix).reshape(-1, 2)
            court_y = court_xy[:, 1]
        else:
            court_xy = None
            court_y = None

        last_bounce_frame = None
        for i in range(1, len(y) - 1):
            # Un bote es un extremo local de la trayectoria vertical (en AMBOS
            # campos): MAXIMO de ball_y en campo cercano, MINIMO en campo lejano.
            if vy[i - 1] > 0 and vy[i] < 0:
                kind = "max"
            elif vy[i - 1] < 0 and vy[i] > 0:
                kind = "min"
            else:
                continue

            if _local_prominence(y, i, kind) < min_prominence:
                continue

            # Distancia minima al bote real anterior ya marcado (mismo clip).
            if last_bounce_frame is not None and (frames[i] - last_bounce_frame) < min_distance:
                continue

            # REGLA PRINCIPAL: cuenta como bote si tras el evento la pelota sigue
            # su trayectoria y NO se va al otro campo. Se aplica a top y bottom.
            if court_y is not None and not _stays_same_court(
                    court_y, i, net_y, config.BOUNCE_DIR_LOOKAHEAD, config.BOUNCE_DIR_MIN_PTS):
                n_rejected_cross += 1
                continue

            # Guarda secundaria: golpe de raqueta claro (pelota a altura de
            # torso/cabeza en la columna del jugador).
            players_here = players_by_clip_frame.get((clip, int(frames[i])), [])
            if _is_racket_hit(bx[i], y[i], players_here, player_exclusion_px, hit_zone_frac):
                n_rejected_player += 1
                continue

            # Descartar ruido: candidatos que proyectan fuera de la pista.
            if court_xy is not None:
                mx, my = court_xy[i]
                if not (-court_margin <= mx <= config.COURT_WIDTH + court_margin
                        and -court_margin <= my <= config.COURT_LENGTH + court_margin):
                    n_rejected_offcourt += 1
                    continue

            df.at[idx[i], "is_real_bounce"] = 1
            last_bounce_frame = frames[i]
            n_total += 1

    logger.info("Botes reales detectados: %d (descartados | cruza campo: %d | golpe raqueta: %d | "
                "fuera de pista: %d)", n_total, n_rejected_cross, n_rejected_player, n_rejected_offcourt)
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
