"""Orquestacion incremental y reanudable del procesamiento de un game.

``run_game`` procesa un game completo en etapas con dependencia estricta
(jugadores -> pelota -> proyeccion/plots/videos) y cache por clip: antes de
computar una etapa sobre un clip se comprueba su sentinela; si existe, se salta.
Nada se borra: si una ejecucion anterior dejo resultados parciales, se continua
desde donde se quedo.

Estructura de outputs bajo ``output_dir``:

    <output-dir>/tracking/players_master.csv   -> jugadores acumulados (todos los clips)
    <output-dir>/tracking/ball_master.csv      -> pelota acumulada
    <output-dir>/tracking/.done_players/<clip> -> sentinela de etapa jugadores
    <output-dir>/tracking/.done_ball/<clip>    -> sentinela de etapa pelota
    <output-dir>/projected/player_real_coords.csv -> jugadores en metros
    <output-dir>/projected/ball_real_coords.csv   -> pelota en metros
    <output-dir>/plots/                         -> heatmaps, rebotes, vista combinada
    <output-dir>/videos/<clip>.mp4              -> predicciones dibujadas por clip
    <output-dir>/court_points.json              -> cache de esquinas

Reglas:
    - La etapa de pelota de un clip solo corre si ya esta hecha la de jugadores.
    - Proyeccion/plots/videos solo corren cuando TODOS los clips tienen jugadores
      y pelota acumulados.
    - Si todo esta completo al arrancar, no se hace nada.

La acumulacion en los CSV master es por append; los sentinelas evitan duplicar
filas al reanudar (no se puede inferir del propio master, que es concatenado).
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
from src.visualization import heatmaps, video

logger = logging.getLogger(__name__)


def _paths(output_dir):
    """Agrupa las rutas de artefactos del run en un dict para no repetirlas."""
    output_dir = Path(output_dir)
    tracking = output_dir / config.TRACKING_SUBDIR
    projected = output_dir / config.PROJECTED_SUBDIR
    return {
        "tracking": tracking,
        "projected": projected,
        "plots": output_dir / config.PLOTS_SUBDIR,
        "videos": output_dir / config.VIDEOS_SUBDIR,
        "players_master": tracking / config.PLAYERS_MASTER_CSV,
        "ball_master": tracking / config.BALL_MASTER_CSV,
        "done_players": tracking / config.DONE_PLAYERS_DIRNAME,
        "done_ball": tracking / config.DONE_BALL_DIRNAME,
        "player_real": projected / config.PLAYER_REAL_CSV,
        "ball_real": projected / config.BALL_REAL_CSV,
    }


def check_stage(output_dir, clip, stage):
    """Indica si el output de una etapa ya existe en disco.

    Centraliza TODA la logica de existencia de ficheros del pipeline.

    Args:
        output_dir: carpeta de salida del run.
        clip: directorio del clip (para ``players``/``ball``) o la LISTA de clips
            del game (para ``final``, que comprueba un video por clip).
        stage: ``"players"``, ``"ball"`` o ``"final"``.

    Returns:
        ``True`` si el output de esa etapa ya esta generado.
    """
    p = _paths(output_dir)
    if stage == "players":
        return io.is_clip_done(p["done_players"], Path(clip).name)
    if stage == "ball":
        return io.is_clip_done(p["done_ball"], Path(clip).name)
    if stage == "final":
        # La etapa final se considera completa si existen los CSV proyectados, la
        # vista combinada de los plots Y un video por cada clip. ``clip`` aqui es
        # la lista de clips del game (no un directorio), para poder contar videos.
        csv_plots_ok = (
            p["player_real"].exists()
            and p["ball_real"].exists()
            and (p["plots"] / "combined_view.png").exists()
        )
        if not csv_plots_ok:
            return False
        clips = clip or []
        return all((p["videos"] / f"{Path(c).name}.mp4").exists() for c in clips)
    raise ValueError(f"Etapa desconocida: {stage!r} (usa 'players', 'ball' o 'final').")


def run_game(game_path, output_dir, export_excel=False):
    """Procesa un game de forma incremental y reanudable.

    Args:
        game_path: ruta a la carpeta del game (p.ej. ``Dataset/game8``).
        output_dir: carpeta de salida del run.
        export_excel: si ``True``, exporta los CSV master a un Excel al final.
    """
    game_path = Path(game_path)
    output_dir = Path(output_dir)
    p = _paths(output_dir)

    # ---- Descubrir clips ----
    clips = loaders.list_clips(game_path)
    if not clips:
        raise FileNotFoundError(f"El game {game_path} no contiene clips con Label.csv.")
    logger.info("Game %s | %d clips: %s",
                game_path.name, len(clips), ", ".join(c.name for c in clips))

    # ---- Cortocircuito: todo completo ----
    all_players = all(check_stage(output_dir, c, "players") for c in clips)
    all_ball = all(check_stage(output_dir, c, "ball") for c in clips)
    final_done = check_stage(output_dir, clips, "final")
    if all_players and all_ball and final_done:
        logger.info("Nada que hacer: todas las etapas ya estan completas en %s.", output_dir)
        return

    # ---- Etapa 1: jugadores por clip (append al master + sentinela) ----
    for clip_dir in clips:
        if check_stage(output_dir, clip_dir, "players"):
            logger.info("[JUGADORES] %s ya procesado; se salta.", clip_dir.name)
            continue
        df_players = player_tracker.track_clip(clip_dir)
        if not df_players.empty:
            io.append_csv(df_players, p["players_master"])
        io.mark_clip_done(p["done_players"], clip_dir.name)
        logger.info("[JUGADORES] %s acumulado (%d filas).", clip_dir.name, len(df_players))

    # ---- Etapa 2: pelota por clip (requiere jugadores del clip) ----
    for clip_dir in clips:
        if check_stage(output_dir, clip_dir, "ball"):
            logger.info("[PELOTA] %s ya procesado; se salta.", clip_dir.name)
            continue
        if not check_stage(output_dir, clip_dir, "players"):
            logger.warning("[PELOTA] %s sin etapa de jugadores; se omite (dependencia).",
                           clip_dir.name)
            continue
        df_ball = ball_tracker.infer_clip(clip_dir)
        if not df_ball.empty:
            io.append_csv(df_ball, p["ball_master"])
        io.mark_clip_done(p["done_ball"], clip_dir.name)
        logger.info("[PELOTA] %s acumulado (%d filas).", clip_dir.name, len(df_ball))

    # ---- Etapa 3: proyeccion + plots + videos (solo si todo esta acumulado) ----
    if not all(check_stage(output_dir, c, "ball") for c in clips):
        pendientes = [c.name for c in clips if not check_stage(output_dir, c, "ball")]
        logger.warning(
            "[FINAL] No se generan proyecciones/plots/videos: faltan %d clips (%s). "
            "Vuelve a lanzar para completar.", len(pendientes), ", ".join(pendientes))
        return

    if check_stage(output_dir, clips, "final"):
        logger.info("[FINAL] Proyecciones, plots y videos ya generados; se salta.")
    else:
        _finalize(game_path, clips, output_dir, p)

    # ---- Export final a Excel (opcional) ----
    if export_excel:
        io.export_excel(
            {p["players_master"]: "jugadores", p["ball_master"]: "pelota"},
            output_dir / config.EXCEL_FILENAME,
        )

    logger.info("Game %s procesado. Resultados en %s", game_path.name, output_dir)


def _finalize(game_path, clips, output_dir, p):
    """Etapa final: homografia, proyeccion a metros, plots y videos por clip."""
    homography_matrix = _resolve_homography(game_path, clips, output_dir)

    df_players = pd.read_csv(p["players_master"]) if p["players_master"].exists() \
        else pd.DataFrame()
    df_ball = pd.read_csv(p["ball_master"]) if p["ball_master"].exists() else pd.DataFrame()

    df_players_proj, df_ball_proj = _project_masters(
        df_players, df_ball, p, homography_matrix)
    _generate_plots(df_players_proj, df_ball_proj, p["plots"])
    _generate_videos(game_path, clips, df_players, df_ball, homography_matrix, p["videos"])


def _resolve_homography(game_path, clips, output_dir):
    """Carga o selecciona las esquinas del game y devuelve la matriz de homografia."""
    cache_path = Path(output_dir) / config.COURT_POINTS_FILENAME
    reference_frame = clips[0] / config.REFERENCE_FRAME
    if not reference_frame.exists():
        reference_frame = loaders.list_frames(clips[0])[0]
    image_points = homography.load_or_select_corners(
        cache_path, reference_frame, cache_key=game_path.name,
    )
    return homography.compute_homography(image_points)


def _project_masters(df_players, df_ball, p, homography_matrix):
    """Proyecta a metros los master de jugadores y pelota y escribe los CSV."""
    p["projected"].mkdir(parents=True, exist_ok=True)

    # Jugadores: proyectar el punto de pie (foot_x, foot_y).
    df_players_proj = homography.project_dataframe(
        df_players, "foot_x", "foot_y", "real_x", "real_y", homography_matrix,
    )
    players_out = (df_players_proj[["clip", "frame", "player_label", "real_x", "real_y"]]
                   .rename(columns={"player_label": "player_id"}))
    io.write_csv(players_out, p["player_real"])
    logger.info("Proyectadas %d filas de jugadores a %s",
                len(players_out), config.PLAYER_REAL_CSV)

    # Pelota: proyectar (ball_x, ball_y).
    if not df_ball.empty:
        df_ball_proj = homography.project_dataframe(
            df_ball, "ball_x", "ball_y", "ball_real_x", "ball_real_y", homography_matrix,
        )
        ball_out = df_ball_proj[["clip", "frame", "ball_real_x", "ball_real_y", "is_bounce"]]
        io.write_csv(ball_out, p["ball_real"])
        logger.info("Proyectadas %d filas de pelota a %s",
                    len(ball_out), config.BALL_REAL_CSV)
    else:
        df_ball_proj = df_ball
        io.write_csv(pd.DataFrame(
            columns=["clip", "frame", "ball_real_x", "ball_real_y", "is_bounce"]),
            p["ball_real"])
        logger.warning("Master de pelota vacio; ball_real_coords.csv sin filas.")

    return df_players_proj, df_ball_proj


def _generate_plots(df_players_proj, df_ball_proj, plots_dir):
    """Genera los PNG de plots a partir de los DataFrames ya proyectados."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    players_by_label = {}
    for label, sub in df_players_proj.groupby("player_label"):
        real_x = sub["real_x"].to_numpy()
        real_y = sub["real_y"].to_numpy()
        players_by_label[label] = (real_x, real_y)
        heatmaps.export_player_heatmap(
            real_x, real_y, title=f"Heatmap — {label}  (n={len(sub)} frames)",
            out_path=plots_dir / f"{label}_heatmap.png",
        )

    bounces_x, bounces_y = np.empty(0), np.empty(0)
    if not df_ball_proj.empty and "ball_real_x" in df_ball_proj.columns:
        valid = df_ball_proj.dropna(subset=["ball_real_x", "ball_real_y"])
        bounces = valid[valid["is_bounce"] == 1]
        bounces_x = bounces["ball_real_x"].to_numpy()
        bounces_y = bounces["ball_real_y"].to_numpy()
        heatmaps.export_bounce_map(bounces_x, bounces_y, plots_dir / "ball_bounces_map.png")

    heatmaps.export_combined_view(
        players_by_label, bounces_x, bounces_y, plots_dir / "combined_view.png",
    )
    logger.info("Plots generados en %s", plots_dir)


def _generate_videos(game_path, clips, df_players, df_ball, homography_matrix, videos_dir):
    """Genera un video por clip con las predicciones dibujadas sobre los frames."""
    videos_dir.mkdir(parents=True, exist_ok=True)
    for clip_dir in clips:
        key = loaders.clip_key(clip_dir)
        out_path = videos_dir / f"{clip_dir.name}.mp4"
        if out_path.exists():
            logger.info("[VIDEO] %s ya existe; se salta.", out_path.name)
            continue
        df_p = df_players[df_players["clip"] == key] if not df_players.empty else df_players
        df_b = df_ball[df_ball["clip"] == key] if not df_ball.empty else df_ball
        video.render_clip_video(clip_dir, df_p, df_b, homography_matrix, out_path)
