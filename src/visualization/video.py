"""Render de video con las predicciones dibujadas sobre los frames originales.

Por cada clip genera un ``.mp4`` que superpone sobre cada frame:

- las cajas y etiquetas de los jugadores (``player_top`` / ``player_bottom``) con
  su punto de pie,
- la marca de la pelota cuando es visible (resaltando los rebotes),
- las lineas de la pista proyectadas al frame con la homografia inversa,
- un minimapa cenital en una esquina con las posiciones de jugadores y pelota a
  escala sobre la pista.

Todo el dibujo es con OpenCV (sin ventanas), asi que funciona headless. Las
coordenadas de jugadores/pelota se leen de los CSV master ya acumulados; las
posiciones en metros para el minimapa se calculan proyectando con la homografia.
"""

import logging

import cv2
import numpy as np

import config
from src.data import loaders
from src.geometry import homography

logger = logging.getLogger(__name__)

# Colores BGR (OpenCV) por etiqueta de jugador.
_PLAYER_COLORS = {
    "player_top": (255, 191, 0),      # deepskyblue
    "player_bottom": (0, 165, 255),   # orange
}
_DEFAULT_PLAYER_COLOR = (255, 0, 255)
_COURT_LINE_COLOR = (255, 255, 255)
_BALL_COLOR = (0, 255, 255)
_BOUNCE_COLOR = (0, 0, 255)


def _project_court_lines_to_frame(homography_matrix):
    """Proyecta las lineas de pista (metros) a pixeles con la homografia inversa.

    Returns:
        Lista de pares ``((x0, y0), (x1, y1))`` en pixeles (enteros).
    """
    h_inv = homography.invert_homography(homography_matrix)
    lines_px = []
    for (p0, p1) in homography.court_lines_world():
        pts = homography.project_points([p0, p1], h_inv)
        (x0, y0), (x1, y1) = pts
        lines_px.append(((int(round(x0)), int(round(y0))),
                         (int(round(x1)), int(round(y1)))))
    return lines_px


def _draw_court_lines(frame, lines_px):
    """Dibuja las lineas de pista proyectadas sobre el frame."""
    for (p0, p1) in lines_px:
        cv2.line(frame, p0, p1, _COURT_LINE_COLOR, 2, cv2.LINE_AA)


def _draw_players(frame, rows):
    """Dibuja cajas, etiquetas y punto de pie de los jugadores de un frame."""
    for r in rows:
        color = _PLAYER_COLORS.get(r["player_label"], _DEFAULT_PLAYER_COLOR)
        x1, y1 = int(r["bbox_x1"]), int(r["bbox_y1"])
        x2, y2 = int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Etiqueta con fondo para legibilidad.
        label = r["player_label"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # Punto de pie.
        cv2.circle(frame, (int(r["foot_x"]), int(r["foot_y"])), 4, color, -1)


def _draw_ball(frame, ball_row):
    """Dibuja la pelota si es visible; resalta el rebote."""
    if ball_row is None or ball_row["visibility"] != 1:
        return
    if np.isnan(ball_row["ball_x"]) or np.isnan(ball_row["ball_y"]):
        return
    cx, cy = int(ball_row["ball_x"]), int(ball_row["ball_y"])
    if ball_row.get("is_bounce", 0) == 1:
        cv2.circle(frame, (cx, cy), 10, _BOUNCE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "BOTE", (cx + 12, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOUNCE_COLOR, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 6, _BALL_COLOR, -1, cv2.LINE_AA)


class _Minimap:
    """Minimapa cenital de la pista, con posiciones en metros.

    Pre-renderiza el fondo de la pista una vez y luego solo dibuja los puntos
    moviles (jugadores y pelota) por frame.
    """

    def __init__(self, homography_matrix, width_px=None):
        self.h = homography_matrix
        margin = 1.0  # metros de margen alrededor de la pista en el minimapa
        self.x_min, self.x_max = -margin, config.COURT_WIDTH + margin
        self.y_min, self.y_max = -margin, config.COURT_LENGTH + margin
        w_m = self.x_max - self.x_min
        h_m = self.y_max - self.y_min
        self.w = width_px or config.MINIMAP_WIDTH_PX
        self.scale = self.w / w_m
        self.h_px = int(round(h_m * self.scale))
        self._base = self._render_base()

    def _to_px(self, x_m, y_m):
        px = int(round((x_m - self.x_min) * self.scale))
        py = int(round((y_m - self.y_min) * self.scale))
        return px, py

    def _render_base(self):
        canvas = np.full((self.h_px, self.w, 3), (60, 90, 50), dtype=np.uint8)  # verde pista
        for (p0, p1) in homography.court_lines_world():
            cv2.line(canvas, self._to_px(*p0), self._to_px(*p1),
                     (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def render(self, player_world, ball_world):
        """Devuelve una copia del minimapa con jugadores y pelota dibujados.

        Args:
            player_world: lista de ``(x_m, y_m, label)`` de jugadores.
            ball_world: ``(x_m, y_m)`` de la pelota o ``None``.
        """
        mini = self._base.copy()
        for (x_m, y_m, label) in player_world:
            cv2.circle(mini, self._to_px(x_m, y_m), 4,
                       _PLAYER_COLORS.get(label, _DEFAULT_PLAYER_COLOR), -1, cv2.LINE_AA)
        if ball_world is not None:
            cv2.circle(mini, self._to_px(*ball_world), 3, _BALL_COLOR, -1, cv2.LINE_AA)
        return mini


def _overlay_minimap(frame, mini):
    """Pega el minimapa en la esquina superior derecha del frame con un borde."""
    fh, fw = frame.shape[:2]
    mh, mw = mini.shape[:2]
    pad = 10
    x0, y0 = fw - mw - pad, pad
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + mw + 2, y0 + mh + 2), (255, 255, 255), 1)
    frame[y0:y0 + mh, x0:x0 + mw] = mini


def render_clip_video(clip_dir, df_players_clip, df_ball_clip, homography_matrix, out_path):
    """Genera el mp4 de un clip con todas las predicciones superpuestas.

    Args:
        clip_dir: carpeta del clip (para leer los frames originales).
        df_players_clip: filas de jugadores de ESTE clip (columnas del master).
        df_ball_clip: filas de pelota de ESTE clip (columnas del master) o vacio.
        homography_matrix: homografia pixel->metros del game.
        out_path: ruta del .mp4 de salida.
    """
    frames = loaders.list_frames(clip_dir)
    if not frames:
        logger.warning("Clip sin frames, no se genera video: %s", clip_dir)
        return

    h, w = cv2.imread(str(frames[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, config.VIDEO_FPS, (w, h))

    lines_px = _project_court_lines_to_frame(homography_matrix)
    minimap = _Minimap(homography_matrix)

    # Indexar filas por frame para acceso O(1) en el bucle.
    players_by_frame = {f: sub.to_dict("records")
                        for f, sub in df_players_clip.groupby("frame")}
    ball_by_frame = ({f: sub.iloc[0].to_dict() for f, sub in df_ball_clip.groupby("frame")}
                     if not df_ball_clip.empty else {})

    for idx, fp in enumerate(frames):
        frame = cv2.imread(str(fp))
        _draw_court_lines(frame, lines_px)

        rows = players_by_frame.get(idx, [])
        _draw_players(frame, rows)

        ball_row = ball_by_frame.get(idx)
        _draw_ball(frame, ball_row)

        # Minimapa: proyectar posiciones a metros.
        player_world = []
        for r in rows:
            xm, ym = homography.project_points(
                [[r["foot_x"], r["foot_y"]]], homography_matrix)[0]
            player_world.append((xm, ym, r["player_label"]))
        ball_world = None
        if ball_row is not None and ball_row["visibility"] == 1 \
                and not np.isnan(ball_row["ball_x"]):
            ball_world = tuple(homography.project_points(
                [[ball_row["ball_x"], ball_row["ball_y"]]], homography_matrix)[0])
        _overlay_minimap(frame, minimap.render(player_world, ball_world))

        writer.write(frame)

    writer.release()
    logger.info("Video del clip guardado en %s (%d frames)", out_path, len(frames))
