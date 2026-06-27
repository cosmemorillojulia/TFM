"""Compara las visualizaciones de cada jugador con presion vs. sin presion.

A partir de los CSV ya proyectados de un run (``player_real_coords.csv`` y
``ball_real_coords.csv``) genera, por jugador y por estado de presion, dos
graficos:

- ``movimiento.png``: heatmap de ocupacion (posiciones del jugador).
- ``botes.png``: mapa de los botes que produce ese jugador (sus golpes caen en
  el campo CONTRARIO).

La presion es por clip y jugador (columna ``pressure`` del master, validada en
el pipeline): para cada jugador, sus clips "con presion" son aquellos donde su
propio ``pressure == 1``.

Normalizacion de lado (igual que en el pipeline de plots): se gira 180 cada clip
en el que ``player_1`` jugaba abajo, de modo que ``player_1`` queda SIEMPRE en la
mitad superior y ``player_2`` en la inferior. Asi:
  - el heatmap de cada jugador cae siempre en su mitad,
  - y el autor de un bote se deduce del campo donde cae: un bote en la mitad
    INFERIOR es un golpe de ``player_1`` (que juega arriba) y uno en la SUPERIOR
    es de ``player_2``.

Estructura de salida (bajo ``<run>/comparison/``):

    comparison/
      <Jugador_1>/
        con_presion/{movimiento,botes}.png
        sin_presion/{movimiento,botes}.png
      <Jugador_2>/
        con_presion/...
        sin_presion/...

Uso:
    python compare_pressure.py --run final_v1
    python compare_pressure.py --run outputs/final_v1 --game-path Dataset_Clutch
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

import config
from src.data import loaders
from src.pipeline import orchestrator
from src.visualization import heatmaps

logger = logging.getLogger("compare_pressure")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compara graficos de cada jugador con presion vs sin presion.",
    )
    parser.add_argument(
        "--run", required=True,
        help="Nombre (o ruta) del run bajo OUTPUTS_ROOT. Solo se usa su ultimo "
             "segmento: 'final_v1' o 'outputs/final_v1' -> OUTPUTS_ROOT/final_v1.",
    )
    parser.add_argument(
        "--game-path", default=None,
        help="Carpeta del game con los info.json de cada clip. Por defecto se "
             "deduce del nombre del run (DATASET_ROOT.parent/<game> si existe) o "
             "se usa la columna 'clip' de los CSV.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _resolve_run_dir(run):
    """Carpeta del run, siempre bajo OUTPUTS_ROOT (mismo criterio que main.py)."""
    return config.OUTPUTS_ROOT / Path(run).name


def _resolve_clips(game_path, clip_keys):
    """Lista de carpetas de clip para leer sus info.json.

    Usa ``--game-path`` si se pasa; si no, reconstruye la ruta de cada clip a
    partir de su clave ``<game>/<clip>`` (relativa a la raiz del proyecto).
    """
    if game_path is not None:
        game_path = Path(game_path)
        return sorted([p for p in game_path.glob("*") if p.is_dir()])
    # Reconstruir desde las claves 'Dataset_Clutch/Clip1' (relativas al proyecto).
    clips = []
    for key in sorted(set(clip_keys)):
        clip_dir = config.PROJECT_ROOT / key
        if clip_dir.is_dir():
            clips.append(clip_dir)
        else:
            logger.warning("No existe la carpeta del clip %s; se omite.", clip_dir)
    return clips


def _load_players_with_pressure(run_dir, names, flip_clips):
    """DataFrame de jugadores proyectado, con ``player_name``, ``pressure`` y lado
    normalizado (player_1 arriba)."""
    proj = pd.read_csv(run_dir / "projected" / "player_real_coords.csv")
    # En el CSV proyectado la etiqueta de lado se guarda como 'player_id'.
    proj = proj.rename(columns={"player_id": "player_label"})

    # Presion por (clip, player_label) desde el master.
    master = pd.read_csv(run_dir / "tracking" / "players_master.csv")
    pres = master[["clip", "player_label", "pressure"]].drop_duplicates()
    proj = proj.merge(pres, on=["clip", "player_label"], how="left")

    proj = orchestrator._normalize_player_side(proj, names, flip_clips)
    return proj


def _load_bounces_with_author(run_dir, names, flip_clips, players):
    """DataFrame de botes proyectado con ``player_name`` (autor) y ``pressure``.

    El autor se deduce del campo donde cae el bote TRAS normalizar: mitad inferior
    (y < NET_Y) -> player_1 (juega arriba); mitad superior -> player_2.
    """
    ball = pd.read_csv(run_dir / "projected" / "ball_real_coords.csv")
    ball = ball.dropna(subset=["ball_real_x", "ball_real_y"])
    ball = ball[ball["is_real_bounce"] == 1].copy()
    if ball.empty:
        return ball.assign(player_name=[], pressure=[])

    ball = orchestrator._flip_xy(ball, flip_clips, "ball_real_x", "ball_real_y")

    # Rol del autor segun el campo (opuesto al que golpea).
    role_top = "player_1"  # juega arriba; sus botes caen abajo
    role_bottom = "player_2"
    author_role = ball["ball_real_y"].apply(
        lambda y: role_bottom if y >= config.NET_Y else role_top)

    # Mapear (clip, rol) -> nombre y -> pressure usando la tabla de jugadores ya
    # normalizada: cada jugador esta en una mitad fija, asi que su rol se deduce
    # de su mitad media. Construimos el cruce por clip.
    # nombre del jugador segun rol: player_1 es quien quedo arriba tras normalizar.
    role_label = {"player_1": "player_top", "player_2": "player_bottom"}
    name_by_clip_role = {}
    pressure_by_clip_role = {}
    pmeta = players[["clip", "player_label", "player_name", "pressure"]].drop_duplicates()
    for _, r in pmeta.iterrows():
        # Tras normalizar, player_label sigue siendo el lado FISICO original; el
        # lado canonico (arriba/abajo ya normalizado) lo da el rol. Reconstruimos
        # el rol invirtiendo el flip: si el clip se giro, el lado se invierte.
        flipped = r["clip"] in flip_clips
        phys_top = r["player_label"] == "player_top"
        canon_top = phys_top != flipped  # tras girar, ¿queda arriba?
        role = "player_1" if canon_top else "player_2"
        name_by_clip_role[(r["clip"], role)] = r["player_name"]
        pressure_by_clip_role[(r["clip"], role)] = r["pressure"]

    ball["_role"] = author_role
    ball["player_name"] = [name_by_clip_role.get((c, rol))
                           for c, rol in zip(ball["clip"], ball["_role"])]
    ball["pressure"] = [pressure_by_clip_role.get((c, rol))
                        for c, rol in zip(ball["clip"], ball["_role"])]
    return ball.drop(columns="_role")


def _state_label(pressure):
    return "con_presion" if int(pressure) == 1 else "sin_presion"


def generate(run, game_path):
    run_dir = _resolve_run_dir(run)
    if not run_dir.is_dir():
        logger.error("No existe el run %s", run_dir)
        return 1

    proj_clip_keys = pd.read_csv(
        run_dir / "projected" / "player_real_coords.csv", usecols=["clip"])["clip"]
    clips = _resolve_clips(game_path, proj_clip_keys)
    if not clips:
        logger.error("No se han encontrado carpetas de clip con info.json.")
        return 1

    names, flip_clips = orchestrator._side_normalization_by_clip(clips)
    players = _load_players_with_pressure(run_dir, names, flip_clips)
    bounces = _load_bounces_with_author(run_dir, names, flip_clips, players)

    out_root = run_dir / "comparison"
    out_root.mkdir(parents=True, exist_ok=True)

    for name, p_sub in players.groupby("player_name"):
        b_sub = bounces[bounces["player_name"] == name] if not bounces.empty \
            else bounces

        # Escala comun por jugador: mismo vmax en con/sin presion para que un
        # color signifique la misma densidad en ambos estados y sean comparables.
        mov_by_state = {pr: p_sub[p_sub["pressure"] == pr] for pr in (1, 0)}
        mov_vmax = max(
            (heatmaps.kde_max(m["real_x"].to_numpy(), m["real_y"].to_numpy())
             for m in mov_by_state.values()), default=0.0) or None
        if not b_sub.empty:
            bb_by_state = {pr: b_sub[b_sub["pressure"] == pr] for pr in (1, 0)}
            bounce_vmax = max(
                (heatmaps.kde_max(b["ball_real_x"].to_numpy(), b["ball_real_y"].to_numpy())
                 for b in bb_by_state.values()), default=0.0) or None
        else:
            bb_by_state, bounce_vmax = {1: b_sub, 0: b_sub}, None

        for pressure in (1, 0):
            state = _state_label(pressure)
            out_dir = out_root / orchestrator._slug(name) / state
            out_dir.mkdir(parents=True, exist_ok=True)

            mov = mov_by_state[pressure]
            heatmaps.export_player_heatmap(
                mov["real_x"].to_numpy(), mov["real_y"].to_numpy(),
                title=f"Movimiento — {name} ({state.replace('_', ' ')})"
                      f"  (n={len(mov)} frames)",
                out_path=out_dir / "movimiento.png",
                vmax=mov_vmax,
            )

            bb = bb_by_state[pressure]
            heatmaps.export_bounce_map(
                bb["ball_real_x"].to_numpy(), bb["ball_real_y"].to_numpy(),
                out_path=out_dir / "botes.png",
                title=f"Botes — {name} ({state.replace('_', ' ')})",
                vmax=bounce_vmax,
            )
            logger.info("[%s/%s] movimiento n=%d, botes n=%d",
                        name, state, len(mov), len(bb))

    logger.info("Comparativa de presion generada en %s", out_root)
    return 0


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return generate(args.run, args.game_path)


if __name__ == "__main__":
    sys.exit(main())
