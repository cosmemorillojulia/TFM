"""Plots por jugador: movimiento + sus botes en un solo mapa, global y por presion.

Para cada jugador genera un UNICO grafico que combina en el mismo mapa de pista:

- su MOVIMIENTO, como heatmap de ocupacion (KDE inferno) en SU mitad, y
- sus BOTES, como marcadores X cyan en la mitad CONTRARIA (sus golpes botan en el
  campo del rival).

Ambos analisis caen en mitades distintas, asi que no se solapan.

Para cada jugador se generan TRES mapas, en su propia carpeta dentro de
``<run>/plots/<slug-jugador>/``:
  - general.png      -> todo el partido
  - con_presion.png  -> solo sus clips con presion
  - sin_presion.png  -> solo sus clips sin presion

La presion es por clip y jugador (columna ``pressure`` del master, validada en
el pipeline): para cada jugador, sus clips "con presion" son aquellos donde su
propio ``pressure == 1``. El heatmap de movimiento usa el MISMO ``vmax`` en global
y en con/sin presion para que un color signifique la misma densidad y sean
comparables.

Normalizacion de lado (la misma que ``orchestrator._generate_plots``): se gira 180
cada clip en el que ``player_1`` jugaba abajo, de modo que ``player_1`` queda
SIEMPRE en la mitad superior y ``player_2`` en la inferior. Asi:
  - el heatmap de cada jugador cae siempre en su mitad,
  - y el autor de un bote se deduce del campo donde cae: un bote en la mitad
    INFERIOR es un golpe de ``player_1`` (que juega arriba) y uno en la SUPERIOR
    es de ``player_2``.

Este modulo es el motor de los plots por jugador; lo invoca el pipeline
(``orchestrator._finalize``, paso 5: tras los plots generales y antes de los
videos). Para regenerarlos sobre un run ya procesado basta con relanzar el
pipeline, que es reanudable y solo rehace la etapa final.

Para evitar un import circular con ``orchestrator`` (que importa este modulo), las
utilidades de normalizacion de lado se importan de forma diferida dentro de las
funciones que las necesitan.
"""

import logging

import pandas as pd

import config
from src.visualization import heatmaps

logger = logging.getLogger(__name__)


def _load_players_with_pressure(run_dir, names, flip_clips):
    """DataFrame de jugadores proyectado, con ``player_name``, ``pressure`` y lado
    normalizado (player_1 arriba)."""
    from src.pipeline import orchestrator

    proj = pd.read_csv(run_dir / config.PROJECTED_SUBDIR / config.PLAYER_REAL_CSV)
    # En el CSV proyectado la etiqueta de lado se guarda como 'player_id'.
    proj = proj.rename(columns={"player_id": "player_label"})

    # Presion por (clip, player_label) desde el master.
    master = pd.read_csv(run_dir / config.TRACKING_SUBDIR / config.PLAYERS_MASTER_CSV)
    pres = master[["clip", "player_label", "pressure"]].drop_duplicates()
    proj = proj.merge(pres, on=["clip", "player_label"], how="left")

    return orchestrator._normalize_player_side(proj, names, flip_clips)


def _load_bounces_with_author(run_dir, names, flip_clips, players):
    """DataFrame de botes proyectado con ``player_name`` (autor) y ``pressure``.

    El autor se deduce del campo donde cae el bote TRAS normalizar: mitad inferior
    (y < NET_Y) -> player_1 (juega arriba); mitad superior -> player_2.
    """
    from src.pipeline import orchestrator

    ball = pd.read_csv(run_dir / config.PROJECTED_SUBDIR / config.BALL_REAL_CSV)
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


def _bounce_xy(df):
    """(x, y) de los botes de un DataFrame; listas vacias si no hay filas."""
    if df.empty:
        return [], []
    return df["ball_real_x"].to_numpy(), df["ball_real_y"].to_numpy()


def generate(run_dir, clips):
    """Genera los plots de movimiento+botes (global y por presion) de un run.

    Args:
        run_dir: carpeta del run (con ``projected/`` y ``tracking/`` ya escritos).
        clips: lista de carpetas de clip, para leer los ``info.json`` (nombres de
            jugador y lado de saque que definen la normalizacion).

    Lee los CSV ya proyectados del run, asi que debe llamarse DESPUES de la
    proyeccion a metros (``_project_masters``).
    """
    from src.pipeline import orchestrator

    names, flip_clips = orchestrator._side_normalization_by_clip(clips)
    players = _load_players_with_pressure(run_dir, names, flip_clips)
    bounces = _load_bounces_with_author(run_dir, names, flip_clips, players)

    plots_dir = run_dir / config.PLOTS_SUBDIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    for name, p_sub in players.groupby("player_name"):
        slug = orchestrator._slug(name)
        b_sub = bounces[bounces["player_name"] == name] if not bounces.empty \
            else bounces

        # Carpeta propia del jugador dentro de plots/: general + con/sin presion.
        out_dir = plots_dir / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # Escala comun por jugador: mismo vmax en general y en con/sin presion para
        # que un color signifique la misma densidad en todos sus graficos y sean
        # comparables. Se toma sobre el general, cota superior natural: la densidad
        # de cualquier subconjunto no la supera tras el KDE. Una para el heatmap de
        # movimiento (inferno) y otra para el KDE de botes (hot).
        mov_vmax = heatmaps.kde_max(
            p_sub["real_x"].to_numpy(), p_sub["real_y"].to_numpy()) or None
        bx, by = _bounce_xy(b_sub)
        bounce_vmax = heatmaps.kde_max(bx, by) or None

        # --- 1) General: todo el partido ---
        heatmaps.export_player_movement_and_bounces(
            p_sub["real_x"].to_numpy(), p_sub["real_y"].to_numpy(), bx, by,
            title=f"{name} — movimiento y botes"
                  f"  (n={len(p_sub)} frames, {len(b_sub)} botes)",
            out_path=out_dir / "general.png",
            vmax=mov_vmax, bounce_vmax=bounce_vmax,
        )

        # --- 2) Comparativa: con presion vs. sin presion ---
        for pressure in (1, 0):
            state = _state_label(pressure)
            mov = p_sub[p_sub["pressure"] == pressure]
            bb = b_sub[b_sub["pressure"] == pressure] if not b_sub.empty else b_sub
            bbx, bby = _bounce_xy(bb)
            heatmaps.export_player_movement_and_bounces(
                mov["real_x"].to_numpy(), mov["real_y"].to_numpy(), bbx, bby,
                title=f"{name} ({state.replace('_', ' ')})"
                      f"  (n={len(mov)} frames, {len(bb)} botes)",
                out_path=out_dir / f"{state}.png",
                vmax=mov_vmax, bounce_vmax=bounce_vmax,
            )
            logger.info("[%s/%s] movimiento n=%d, botes n=%d",
                        name, state, len(mov), len(bb))

    logger.info("Plots por jugador en %s", plots_dir)
