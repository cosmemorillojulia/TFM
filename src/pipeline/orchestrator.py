"""Orquestacion del procesamiento completo de un game.

Sustituye al encadenado "Restart & Run All" de los 4 notebooks por una funcion
unica, ``run_game``, que:

1. Descubre todos los clips del game.
2. Recorre los clips uno a uno (la unidad del bucle es el CLIP, no el frame,
   porque TrackNet necesita 3 frames consecutivos y ByteTrack necesita
   continuidad temporal). Por cada clip:
      - trackea los jugadores  -> append a ``players_master.csv``
      - infiere la pelota       -> append a ``ball_master.csv``
3. Tras procesar todos los clips: calcula la homografia del game, proyecta los
   dos CSV a metros y genera los mapas de calor.
4. Opcionalmente exporta los CSV master a un Excel.

La acumulacion es siempre por append a CSV; el Excel es solo una exportacion
final. Las rutas de salida cuelgan de la carpeta del run (``outputs/<game>``).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.data import loaders
from src.geometry import homography
from src.tracking import ball_tracker, player_tracker
from src.utils import io
from src.visualization import heatmaps

logger = logging.getLogger(__name__)


def run_game(game_path, output_dir, export_excel=False):
    """Procesa un game entero y genera todos los artefactos en ``output_dir``.

    Args:
        game_path: ruta a la carpeta del game (p.ej. ``Dataset/game1``).
        output_dir: carpeta de salida del run (p.ej. ``outputs/game1``).
        export_excel: si ``True``, exporta los CSV master a un Excel al final.
    """
    game_path = Path(game_path)
    output_dir = Path(output_dir)

    # Rutas de los artefactos del run.
    tracking_dir = output_dir / config.TRACKING_SUBDIR
    projected_dir = output_dir / config.PROJECTED_SUBDIR
    heatmaps_dir = output_dir / config.HEATMAPS_SUBDIR
    players_master = tracking_dir / config.PLAYERS_MASTER_CSV
    ball_master = tracking_dir / config.BALL_MASTER_CSV

    # ---- 1. Descubrir clips ----
    clips = loaders.list_clips(game_path)
    if not clips:
        raise FileNotFoundError(f"El game {game_path} no contiene clips con Label.csv.")
    logger.info("Game %s | %d clips: %s",
                game_path.name, len(clips), ", ".join(c.name for c in clips))

    # ---- 2. Bucle clip a clip con acumulacion por append ----
    for clip_dir in clips:
        df_players = player_tracker.track_clip(clip_dir)
        if not df_players.empty:
            io.append_csv(df_players, players_master)

        df_ball = ball_tracker.infer_clip(clip_dir)
        if not df_ball.empty:
            io.append_csv(df_ball, ball_master)

    # ---- 3. Homografia + proyeccion + heatmaps (sobre el game completo) ----
    homography_matrix = _resolve_homography(game_path, clips, output_dir)
    _project_masters(players_master, ball_master, projected_dir, homography_matrix)
    _generate_heatmaps(projected_dir, heatmaps_dir)

    # ---- 4. Export final a Excel (opcional) ----
    if export_excel:
        io.export_excel(
            {players_master: "jugadores", ball_master: "pelota"},
            output_dir / config.EXCEL_FILENAME,
        )

    logger.info("Game %s procesado. Resultados en %s", game_path.name, output_dir)


def _resolve_homography(game_path, clips, output_dir):
    """Carga o selecciona las esquinas del game y devuelve la matriz de homografia.

    Las esquinas se cachean en ``output_dir/court_points.json``. El frame de
    referencia es ``REFERENCE_FRAME`` del primer clip del game.
    """
    cache_path = output_dir / config.COURT_POINTS_FILENAME
    reference_frame = clips[0] / config.REFERENCE_FRAME
    if not reference_frame.exists():
        # Si no existe el frame de referencia esperado, usa el primer frame del clip.
        reference_frame = loaders.list_frames(clips[0])[0]

    image_points = homography.load_or_select_corners(
        cache_path, reference_frame, cache_key=game_path.name,
    )
    return homography.compute_homography(image_points)


def _project_masters(players_master, ball_master, projected_dir, homography_matrix):
    """Proyecta a metros los CSV master de jugadores y pelota."""
    projected_dir.mkdir(parents=True, exist_ok=True)

    # Jugadores: proyectar el punto de pie (foot_x, foot_y).
    df_players = pd.read_csv(players_master)
    df_players_proj = homography.project_dataframe(
        df_players, "foot_x", "foot_y", "real_x", "real_y", homography_matrix,
    )
    players_out = df_players_proj[["clip", "frame", "player_label", "real_x", "real_y"]]
    players_out = players_out.rename(columns={"player_label": "player_id"})
    players_out.to_csv(projected_dir / config.PLAYER_REAL_CSV, index=False)
    logger.info("Proyectadas %d filas de jugadores a %s",
                len(players_out), config.PLAYER_REAL_CSV)

    # Pelota: proyectar (ball_x, ball_y) si existe el master.
    if Path(ball_master).exists():
        df_ball = pd.read_csv(ball_master)
        df_ball_proj = homography.project_dataframe(
            df_ball, "ball_x", "ball_y", "ball_real_x", "ball_real_y", homography_matrix,
        )
        ball_out = df_ball_proj[["clip", "frame", "ball_real_x", "ball_real_y", "is_bounce"]]
        ball_out.to_csv(projected_dir / config.BALL_REAL_CSV, index=False)
        logger.info("Proyectadas %d filas de pelota a %s",
                    len(ball_out), config.BALL_REAL_CSV)
    else:
        logger.warning("No hay %s; se omite la proyeccion de la pelota.", ball_master)


def _generate_heatmaps(projected_dir, heatmaps_dir):
    """Genera los PNG de heatmaps a partir de las coordenadas proyectadas."""
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    df_players = pd.read_csv(projected_dir / config.PLAYER_REAL_CSV)

    # Heatmap por jugador (escalas independientes).
    players_by_label = {}
    for label, sub in df_players.groupby("player_id"):
        real_x = sub["real_x"].to_numpy()
        real_y = sub["real_y"].to_numpy()
        players_by_label[label] = (real_x, real_y)
        out_path = heatmaps_dir / f"{label}_heatmap.png"
        heatmaps.export_player_heatmap(
            real_x, real_y, title=f"Heatmap — {label}  (n={len(sub)} frames)",
            out_path=out_path,
        )

    # Mapa de rebotes (si hay pelota proyectada).
    bounces_x, bounces_y = np.empty(0), np.empty(0)
    ball_real_csv = projected_dir / config.BALL_REAL_CSV
    if ball_real_csv.exists():
        df_ball = pd.read_csv(ball_real_csv).dropna(subset=["ball_real_x", "ball_real_y"])
        bounces = df_ball[df_ball["is_bounce"] == 1]
        bounces_x = bounces["ball_real_x"].to_numpy()
        bounces_y = bounces["ball_real_y"].to_numpy()
        heatmaps.export_bounce_map(bounces_x, bounces_y, heatmaps_dir / "ball_bounces_map.png")

    # Vista combinada.
    heatmaps.export_combined_view(
        players_by_label, bounces_x, bounces_y, heatmaps_dir / "combined_view.png",
    )
