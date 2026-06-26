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
_BALL_COLOR = (0, 255, 255)
_BOUNCE_COLOR = (0, 0, 255)
_MINIMAP_COURT_COLOR = (0, 0, 0)  # negro


def _draw_players(frame, rows, label_names=None):
    """Dibuja cajas, etiquetas y punto de pie de los jugadores de un frame.

    Args:
        frame: imagen BGR (se modifica in place).
        rows: filas de jugadores del frame.
        label_names: dict opcional ``{player_label: nombre_real}`` para mostrar
            el nombre del jugador (del info.json) en vez de ``player_top``/
            ``player_bottom``. Si falta o no contiene la etiqueta, se usa la
            etiqueta cruda.
    """
    label_names = label_names or {}
    for r in rows:
        color = _PLAYER_COLORS.get(r["player_label"], _DEFAULT_PLAYER_COLOR)
        x1, y1 = int(r["bbox_x1"]), int(r["bbox_y1"])
        x2, y2 = int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Etiqueta con fondo para legibilidad.
        label = label_names.get(r["player_label"], r["player_label"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # Punto de pie.
        cv2.circle(frame, (int(r["foot_x"]), int(r["foot_y"])), 4, color, -1)


def _draw_ball(frame, ball_row):
    """Dibuja la pelota si es visible."""
    if ball_row is None or ball_row["visibility"] != 1:
        return
    if np.isnan(ball_row["ball_x"]) or np.isnan(ball_row["ball_y"]):
        return
    cx, cy = int(ball_row["ball_x"]), int(ball_row["ball_y"])
    cv2.circle(frame, (cx, cy), 6, _BALL_COLOR, -1, cv2.LINE_AA)


def _draw_persistent_bounces(frame, bounces):
    """Dibuja una marca roja translucida PERSISTENTE en cada bote ya ocurrido.

    A diferencia de la marca instantanea anterior (solo visible en el frame del
    bote), estos puntos se acumulan y se mantienen en pantalla en todos los
    frames siguientes, para poder ver claramente donde ha botado la pelota a lo
    largo del punto.

    Args:
        frame: imagen BGR sobre la que dibujar (se modifica in place).
        bounces: lista de ``(x_px, y_px)`` de los botes ocurridos hasta el frame
            actual.
    """
    if not bounces:
        return
    # Capa translucida: se dibujan los circulos rellenos en una copia y se mezcla
    # con el frame con alpha, para que se vea el suelo a traves de la marca.
    overlay = frame.copy()
    for (x, y) in bounces:
        cv2.circle(overlay, (int(x), int(y)), config.BOUNCE_MARK_RADIUS_PX,
                   _BOUNCE_COLOR, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, config.BOUNCE_MARK_ALPHA,
                    frame, 1 - config.BOUNCE_MARK_ALPHA, 0, dst=frame)
    # Borde opaco para que la marca resalte sobre fondos claros.
    for (x, y) in bounces:
        cv2.circle(frame, (int(x), int(y)), config.BOUNCE_MARK_RADIUS_PX,
                   _BOUNCE_COLOR, 1, cv2.LINE_AA)


class _Minimap:
    """Minimapa cenital de la pista, con posiciones en metros.

    Pre-renderiza el fondo de la pista una vez y luego solo dibuja los puntos
    moviles (jugadores y pelota) por frame.
    """

    def __init__(self, homography_matrix, width_px=None):
        self.h = homography_matrix
        # Margenes (metros) alrededor de la pista. El margen de fondo (Y) es mayor
        # porque los jugadores sacan/restan bastante por detras de la linea de
        # fondo; el lateral (X) puede ser menor. Se reutilizan MARGIN_Y/MARGIN_X
        # de config (los mismos del canvas de heatmaps).
        margin_x = config.MINIMAP_MARGIN_X
        margin_y = config.MINIMAP_MARGIN_Y
        self.x_min, self.x_max = -margin_x, config.COURT_WIDTH + margin_x
        self.y_min, self.y_max = -margin_y, config.COURT_LENGTH + margin_y
        w_m = self.x_max - self.x_min
        h_m = self.y_max - self.y_min
        self.w = width_px or config.MINIMAP_WIDTH_PX
        self.scale = self.w / w_m
        self.h_px = int(round(h_m * self.scale))
        self._base = self._render_base()

    def _to_px(self, x_m, y_m):
        # Y de la pista invertida: Y mundo alto (fondo lejano) debe quedar
        # arriba en el minimapa (py pequeno), igual que en el frame original.
        px = int(round((x_m - self.x_min) * self.scale))
        py = int(round((self.y_max - y_m) * self.scale))
        return px, py

    def _render_base(self):
        canvas = np.full((self.h_px, self.w, 3), _MINIMAP_COURT_COLOR, dtype=np.uint8)
        for (p0, p1) in homography.court_lines_world():
            cv2.line(canvas, self._to_px(*p0), self._to_px(*p1),
                     (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def render(self, player_world, ball_world):
        """Devuelve una copia del minimapa con jugadores y pelota dibujados.

        Jugadores como cuadrados; pelota como circulo.

        Args:
            player_world: lista de ``(x_m, y_m, label)`` de jugadores.
            ball_world: ``(x_m, y_m)`` de la pelota o ``None``.
        """
        mini = self._base.copy()
        half = 4  # semilado del cuadrado del jugador (px)
        for (x_m, y_m, label) in player_world:
            cx, cy = self._to_px(x_m, y_m)
            color = _PLAYER_COLORS.get(label, _DEFAULT_PLAYER_COLOR)
            cv2.rectangle(mini, (cx - half, cy - half), (cx + half, cy + half), color, -1)
        if ball_world is not None:
            cv2.circle(mini, self._to_px(*ball_world), 3, _BALL_COLOR, -1, cv2.LINE_AA)
        return mini


def _overlay_minimap(frame, mini):
    """Pega el minimapa opaco en el lado derecho, centrado verticalmente."""
    fh, fw = frame.shape[:2]
    mh, mw = mini.shape[:2]
    pad = 10
    x0 = fw - mw - pad
    y0 = (fh - mh) // 2          # centrado verticalmente en el lado derecho
    frame[y0:y0 + mh, x0:x0 + mw] = mini
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + mw + 2, y0 + mh + 2), (255, 255, 255), 1)


def render_clip_video(clip_dir, df_players_clip, df_ball_clip, homography_matrix,
                      out_path, label_names=None):
    """Genera el mp4 de un clip con todas las predicciones superpuestas.

    Args:
        clip_dir: carpeta del clip (para leer los frames originales).
        df_players_clip: filas de jugadores de ESTE clip (columnas del master).
        df_ball_clip: filas de pelota de ESTE clip (columnas del master) o vacio.
        homography_matrix: homografia pixel->metros del game.
        out_path: ruta del .mp4 de salida.
        label_names: dict opcional ``{player_label: nombre_real}`` para etiquetar
            a los jugadores con su nombre (del info.json) en vez de top/bottom.
    """
    frames = loaders.list_frames(clip_dir)
    if not frames:
        logger.warning("Clip sin frames, no se genera video: %s", clip_dir)
        return

    h, w = cv2.imread(str(frames[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, config.VIDEO_FPS, (w, h))

    minimap = _Minimap(homography_matrix)

    # Indexar filas por frame para acceso O(1) en el bucle.
    players_by_frame = {f: sub.to_dict("records")
                        for f, sub in df_players_clip.groupby("frame")}
    ball_by_frame = ({f: sub.iloc[0].to_dict() for f, sub in df_ball_clip.groupby("frame")}
                     if not df_ball_clip.empty else {})

    # Botes que se van marcando de forma persistente: una vez ocurre un bote, su
    # punto rojo translucido se mantiene en pantalla en todos los frames siguientes.
    bounces_so_far = []

    for idx, fp in enumerate(frames):
        frame = cv2.imread(str(fp))

        rows = players_by_frame.get(idx, [])
        _draw_players(frame, rows, label_names)

        ball_row = ball_by_frame.get(idx)
        # Acumular el bote de este frame (si lo hay) antes de dibujar la capa.
        if (ball_row is not None and ball_row.get("is_real_bounce", 0) == 1
                and not np.isnan(ball_row["ball_x"]) and not np.isnan(ball_row["ball_y"])):
            bounces_so_far.append((ball_row["ball_x"], ball_row["ball_y"]))
        _draw_persistent_bounces(frame, bounces_so_far)
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
