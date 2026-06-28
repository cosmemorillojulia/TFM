"""Homografia: proyeccion de coordenadas en pixeles a metros sobre la pista.

Estima la homografia que lleva pixeles del frame a coordenadas reales (metros)
del plano de la pista ITF, a partir de los 10 puntos de control de la pista
(4 esquinas, lineas de saque y T). Los puntos se cargan de un JSON cacheado por
game; si no existe, se abre una ventana OpenCV para seleccionarlos con el raton
(una sola vez) y se guardan.

Por defecto se usa UNA homografia por game (camara consistente). El diseno esta
preparado para usar una homografia por clip en el futuro: basta pasar una
``cache_key`` distinta (p.ej. ``"game1_Clip1"``) a ``load_or_select_corners``,
sin tocar el resto del modulo.

Logica portada de ``old_notebooks/03_homography.ipynb``.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# Orden de las esquinas exigido (mismo que en el notebook):
#   1) inferior-izquierda  2) inferior-derecha  3) superior-derecha  4) superior-izquierda
POINT_ORDER = [
    "bottom_left", "bottom_right", "top_right", "top_left",
    "service_near_left", "service_near_right",
    "service_far_left", "service_far_right",
    "T_near", "T_far",
]
_CLICK_LABELS = [
    "1: inf-izq", "2: inf-der", "3: sup-der", "4: sup-izq",
    "5: saque-cerca-izq", "6: saque-cerca-der",
    "7: saque-lejos-izq", "8: saque-lejos-der",
    "9: T-cerca", "10: T-lejos",
]

# Puntos del mundo (metros) correspondientes a los 10 puntos de control.
# Ejes: X = ancho (0..COURT_WIDTH), Y = largo (0..COURT_LENGTH, red en COURT_LENGTH/2).
_W  = config.COURT_WIDTH
_L  = config.COURT_LENGTH
_MX = config.COURT_WIDTH / 2.0          # centro en X
_SN = config.NET_Y - config.SERVICE_LINE_FROM_NET   # linea de saque lado bottom (Y bajo, cerca)
_SF = config.NET_Y + config.SERVICE_LINE_FROM_NET   # linea de saque lado top (Y alto, lejos)
WORLD_POINTS = np.array([
    [0.0, 0.0],   [_W,  0.0],   # esquinas bottom
    [_W,  _L],    [0.0, _L],    # esquinas top
    [0.0, _SN],   [_W,  _SN],   # linea de saque lado bottom (cerca)
    [0.0, _SF],   [_W,  _SF],   # linea de saque lado top (lejos)
    [_MX, _SN],   [_MX, _SF],   # T central cerca y lejos
], dtype=np.float32)


def project_points(pts_xy, matrix):
    """Proyecta un array (N, 2) de puntos con una homografia 3x3.

    Args:
        pts_xy: array (N, 2) de puntos a proyectar.
        matrix: matriz de homografia 3x3.

    Returns:
        Array (N, 2) de puntos proyectados.
    """
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, matrix)
    return out.reshape(-1, 2)


def compute_homography(image_points):
    """Calcula la homografia pixel -> mundo (metros) a partir de los puntos de control.

    Ajusta una unica homografia global por minimos cuadrados (``cv2.findHomography``)
    usando todos los puntos de control disponibles (esquinas, lineas de saque y T).

    Args:
        image_points: array (N, 2) con N >= 4 puntos en pixeles, en el orden
            de ``POINT_ORDER``.

    Returns:
        Matriz de homografia 3x3 (pixel -> metros).
    """
    image_points = np.asarray(image_points, dtype=np.float32)
    n = len(image_points)
    if n < 4:
        raise ValueError(f"Se necesitan al menos 4 puntos; recibido {n}")

    world = WORLD_POINTS[:n]
    homography, _mask = cv2.findHomography(image_points, world, method=0)

    # Sanity check: reproyectar los puntos de control debe recuperar el mundo.
    recovered = project_points(image_points, homography)
    err = np.linalg.norm(recovered - world, axis=1)
    logger.info("Homografia calculada (%d pts) | error reproyeccion medio %.4f m (max %.4f m)",
                n, err.mean(), err.max())
    return homography


def invert_homography(matrix):
    """Devuelve la homografia inversa (metros -> pixel) de una matriz pixel->metros."""
    return np.linalg.inv(np.asarray(matrix, dtype=np.float64))


def court_lines_world():
    """Devuelve las lineas de la pista ITF en coordenadas mundo (metros).

    Cada linea es un par ``((x0, y0), (x1, y1))``. Mismas referencias que el
    dibujo cenital de ``heatmaps._draw_court_2d`` (perimetro, lineas de saque,
    central de saque y red).

    Returns:
        Lista de tuplas ``((x0, y0), (x1, y1))`` en metros.
    """
    w, length = config.COURT_WIDTH, config.COURT_LENGTH
    s_near = config.NET_Y - config.SERVICE_LINE_FROM_NET
    s_far = config.NET_Y + config.SERVICE_LINE_FROM_NET
    return [
        ((0, 0), (w, 0)),                 # fondo cercano
        ((w, 0), (w, length)),            # lateral derecho
        ((w, length), (0, length)),       # fondo lejano
        ((0, length), (0, 0)),            # lateral izquierdo
        ((0, s_near), (w, s_near)),       # linea de saque cercana
        ((0, s_far), (w, s_far)),         # linea de saque lejana
        ((w / 2, s_near), (w / 2, s_far)),  # central de saque
        ((0, config.NET_Y), (w, config.NET_Y)),  # red
    ]


def load_or_select_corners(cache_path, reference_frame_path, cache_key=None):
    """Carga las esquinas de pista del cache JSON, o las selecciona interactivamente.

    Si ``cache_path`` existe se reutilizan las esquinas guardadas. Si no, se abre
    una ventana OpenCV sobre ``reference_frame_path`` para clicar las 4 esquinas
    y el resultado se guarda en ``cache_path``.

    Args:
        cache_path: ruta del JSON de esquinas (``outputs/<game>/court_points.json``).
        reference_frame_path: frame sobre el que seleccionar (primer frame del game).
        cache_key: clave identificadora a guardar en el JSON (por defecto, el
            nombre del game). Permite migrar a homografia por clip en el futuro.

    Returns:
        Array (4, 2) de esquinas en pixeles, en el orden de ``POINT_ORDER``.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_points = np.array(data["image_points"], dtype=np.float32)
        logger.info("Esquinas de pista cargadas de %s", cache_path)
        return image_points

    logger.info("No hay cache de esquinas en %s; abriendo seleccion interactiva.", cache_path)
    # El servidor web escribe el JSON directamente en cache_path; no hace falta
    # volver a guardarlo aqui.
    return _select_corners_interactive(reference_frame_path, cache_path)


def _select_corners_interactive(reference_frame_path, cache_path):
    """Lanza el servidor web de seleccion de esquinas y espera el resultado.

    Sustituye la seleccion via cv2.imshow (incompatible con entornos headless).
    Lanza select_court_points.py en un subproceso; el JSON lo escribe ese script
    y esta funcion lo lee al terminar.

    Returns:
        Array (4, 2) de esquinas en pixeles.

    Raises:
        RuntimeError: si el usuario no completa la seleccion.
    """
    import subprocess
    import sys
    from pathlib import Path

    cache_path = Path(cache_path)

    # Deducir el game a partir de la ruta del frame (outputs/<game>/... o dataset/<game>/...)
    ref = Path(reference_frame_path)
    # Buscar el game en la ruta: es el segmento que empieza por "game"
    game = next((p for p in ref.parts if p.startswith("game")), None)
    if game is None:
        raise RuntimeError(
            f"No se pudo deducir el game de la ruta {reference_frame_path}. "
            "Ejecuta manualmente: python select_court_points.py --game <game>"
        )

    script = Path(__file__).resolve().parents[2] / "select_court_points.py"
    port = 7860
    logger.info(
        "Entorno headless detectado. Abre http://0.0.0.0:%d en tu navegador para "
        "seleccionar las esquinas de %s y pulsa Guardar.", port, game
    )
    try:
        subprocess.run(
            [sys.executable, str(script),
             "--game", game,
             "--output-dir", str(cache_path.parent),
             "--port", str(port)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"El servidor de seleccion de esquinas falló: {exc}") from exc

    if not cache_path.exists():
        raise RuntimeError(
            f"El servidor terminó pero no se encontró {cache_path}. "
            "¿Guardaste los puntos antes de cerrar?"
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["image_points"], dtype=np.float32)


def _save_corners(cache_path, image_points, reference_frame_path, cache_key):
    """Guarda las esquinas seleccionadas en el JSON de cache."""
    img = cv2.imread(str(reference_frame_path))
    frame_h, frame_w = img.shape[:2]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key or cache_path.parent.name,
        "reference_frame": str(reference_frame_path),
        "frame_size": [int(frame_w), int(frame_h)],
        "court_type": config.COURT_TYPE,
        "image_points": np.asarray(image_points, dtype=float).tolist(),
        "world_points": WORLD_POINTS.tolist(),
        "point_order": POINT_ORDER,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Esquinas guardadas en %s", cache_path)


def project_dataframe(df, x_col, y_col, out_x_col, out_y_col, homography):
    """Proyecta dos columnas (x, y) en pixeles de un DataFrame a metros.

    Solo proyecta las filas con coordenadas validas (no NaN en ``x_col``/
    ``y_col``); el resto quedan como NaN en las columnas de salida.

    Args:
        df: DataFrame de entrada.
        x_col, y_col: nombres de las columnas de pixeles a proyectar.
        out_x_col, out_y_col: nombres de las columnas de metros a crear.
        homography: matriz de homografia 3x3.

    Returns:
        Copia del DataFrame con las columnas proyectadas anadidas.
    """
    df = df.copy()
    real = np.full((len(df), 2), np.nan, dtype=np.float64)
    mask = df[x_col].notna() & df[y_col].notna()
    if mask.any():
        real[mask.values] = project_points(df.loc[mask, [x_col, y_col]].to_numpy(), homography)
    df[out_x_col] = real[:, 0]
    df[out_y_col] = real[:, 1]
    return df
