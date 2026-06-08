"""Homografia: proyeccion de coordenadas en pixeles a metros sobre la pista.

Estima la homografia que lleva pixeles del frame a coordenadas reales (metros)
del plano de la pista ITF, a partir de las 4 esquinas de la pista. Las esquinas
se cargan de un JSON cacheado por game; si no existe, se abre una ventana OpenCV
para seleccionarlas con el raton (una sola vez) y se guardan.

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
POINT_ORDER = ["bottom_left", "bottom_right", "top_right", "top_left"]
_CLICK_LABELS = ["1: inf-izq", "2: inf-der", "3: sup-der", "4: sup-izq"]

# Puntos del mundo (metros) correspondientes a esas 4 esquinas.
WORLD_POINTS = np.array([
    [0.0,                0.0],
    [config.COURT_WIDTH, 0.0],
    [config.COURT_WIDTH, config.COURT_LENGTH],
    [0.0,                config.COURT_LENGTH],
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
    """Calcula la homografia pixel -> mundo (metros) a partir de las 4 esquinas.

    Con exactamente 4 puntos no degenerados el sistema tiene solucion cerrada,
    asi que no hace falta RANSAC (``method=0``).

    Args:
        image_points: array (4, 2) con las esquinas en pixeles, en el orden de
            ``POINT_ORDER``.

    Returns:
        Matriz de homografia 3x3 (pixel -> metros).
    """
    image_points = np.asarray(image_points, dtype=np.float32)
    if image_points.shape != (4, 2):
        raise ValueError(f"Se esperaban 4 esquinas (4, 2); recibido {image_points.shape}")
    homography, _mask = cv2.findHomography(image_points, WORLD_POINTS, method=0)

    # Sanity check: reproyectar las esquinas debe recuperar los puntos del mundo.
    recovered = project_points(image_points, homography)
    err = np.linalg.norm(recovered - WORLD_POINTS, axis=1)
    logger.info("Homografia calculada | error reproyeccion medio %.4f m (max %.4f m)",
                err.mean(), err.max())
    return homography


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
    image_points = _select_corners_interactive(reference_frame_path)
    _save_corners(cache_path, image_points, reference_frame_path, cache_key)
    return image_points


def _select_corners_interactive(reference_frame_path):
    """Abre una ventana OpenCV y deja clicar las 4 esquinas de la pista.

    Returns:
        Array (4, 2) de esquinas en pixeles.

    Raises:
        RuntimeError: si la ventana se cierra antes de marcar las 4 esquinas.
    """
    img = cv2.imread(str(reference_frame_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer el frame de referencia: {reference_frame_path}")

    clicked = []
    window = "Selecciona 4 esquinas: 1) inf-izq  2) inf-der  3) sup-der  4) sup-izq"

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append([float(x), float(y)])
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(img, _CLICK_LABELS[len(clicked) - 1], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if len(clicked) >= 2:
                pts = np.array(clicked, dtype=np.int32)
                cv2.polylines(img, [pts], len(clicked) == 4, (0, 255, 255), 1)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_mouse)
    print("Clica las 4 esquinas en orden (inf-izq, inf-der, sup-der, sup-izq). "
          "Pulsa una tecla al terminar.")
    while True:
        cv2.imshow(window, img)
        key = cv2.waitKey(20) & 0xFF
        # Salir al completar las 4 esquinas y pulsar tecla, o con ESC.
        if (len(clicked) == 4 and key != 255) or key == 27:
            break
    cv2.destroyWindow(window)

    if len(clicked) != 4:
        raise RuntimeError(f"Se necesitan 4 esquinas; se marcaron {len(clicked)}.")
    return np.array(clicked, dtype=np.float32)


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
