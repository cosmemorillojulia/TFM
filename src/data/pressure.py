"""Calculo de presion por punto a partir del info.json de cada clip.

Cada clip de ``Dataset_Clutch/<clip>/`` es UN punto de un partido. Su
``info.json`` describe el marcador y las banderas de situacion (break point,
game/set/match point, tiebreak). A partir de ahi se decide que jugador(es)
juegan ese punto bajo presion.

La logica de ``compute_pressure`` / ``resolve_pressure_for_row`` esta validada
contra los ejemplos de info.json del enunciado y NO se reescribe aqui: se usa
tal cual. Lo unico que añade este modulo alrededor es:

  - ``load_clip_info``: leer el info.json de UN clip, ignorando los campos no
    usados (video_url, start_time, end_time, tournament) y normalizando
    ``player_1_position`` (algunos clips traen ``"bottom "`` con espacio final,
    que rompería la comparacion exacta de ``resolve_pressure_for_row``).

El flujo previsto es autocontenido por clip: leer info.json -> ejecutar el
pipeline de vision sobre sus frames -> resolver la presion de cada fila con la
info ya leida -> guardar. No hace falta cargar todos los JSON por adelantado.
"""

import json
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# Campos del info.json que NO intervienen en el calculo de presion ni en el
# pipeline; se descartan al cargar para dejar claro que no se usan.
_IGNORED_FIELDS = ("video_url", "start_time", "end_time", "tournament")


# ---------------------------------------------------------------------------
# Logica de presion (validada en el enunciado; no reescribir).
# ---------------------------------------------------------------------------

_GAME_ORDER = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def _is_losing_game(score_p1: str, score_p2: str) -> bool:
    """True si player_1 va perdiendo el game actual."""
    return _GAME_ORDER[score_p1] < _GAME_ORDER[score_p2]


def compute_pressure(info: dict) -> tuple[int, int]:
    """Devuelve (pressure_p1, pressure_p2) en {0,1} para un clip,
    a partir de su info.json."""

    # Regla 1: tiebreak pisa todo lo demás
    if info["tie_break"]:
        return 1, 1

    server_is_p1 = info["server"] == "player_1"

    # Regla 2: break point -> presión solo para quien saca
    pressure_p1 = server_is_p1 and info["is_break_point"]
    pressure_p2 = (not server_is_p1) and info["is_break_point"]

    # Regla 3: game_point O set_point -> presión para quien va perdiendo
    # el game actual (mismo cálculo para ambos flags).
    if info["game_point"] or info["is_set_point"]:
        p1_losing_game = _is_losing_game(info["score_player1"], info["score_player2"])
        pressure_p1 = pressure_p1 or p1_losing_game
        pressure_p2 = pressure_p2 or (not p1_losing_game)

    # Regla 4: match point -> presión para quien va perdiendo el partido
    if info["is_match_point"]:
        s1, s2 = info["set_player1"], info["set_player2"]
        if s1 != s2:
            p1_losing_match = s1 < s2
        else:
            # sets empatados: decide el game/set en curso
            p1_losing_match = _is_losing_game(info["score_player1"], info["score_player2"])
        pressure_p1 = pressure_p1 or p1_losing_match
        pressure_p2 = pressure_p2 or (not p1_losing_match)

    return int(pressure_p1), int(pressure_p2)


def resolve_player_names(info: dict) -> dict:
    """Mapea ``player_top``/``player_bottom`` a los nombres reales del info.json.

    Cruza la posicion de player_1 (``player_1_position``) con los nombres
    ``player_1`` / ``player_2`` para saber que nombre va arriba y cual abajo.

    Args:
        info: dict del info.json del clip (ya saneado por ``load_clip_info``).

    Returns:
        dict ``{"player_top": <nombre>, "player_bottom": <nombre>}``.
    """
    p1_pos = info["player_1_position"]
    name_top = info["player_1"] if p1_pos == "top" else info["player_2"]
    name_bottom = info["player_1"] if p1_pos == "bottom" else info["player_2"]
    return {"player_top": name_top, "player_bottom": name_bottom}


def resolve_pressure_for_row(info: dict, player_label: str) -> int:
    """A partir de info.json y el player_label de una fila de players_master
    ('player_top' o 'player_bottom'), devuelve la presión (0/1) de ESE
    jugador concreto en ese clip."""
    pressure_p1, pressure_p2 = compute_pressure(info)

    row_position = "top" if player_label == "player_top" else "bottom"
    p1_is_this_row = info["player_1_position"] == row_position

    return pressure_p1 if p1_is_this_row else pressure_p2


# ---------------------------------------------------------------------------
# Carga del info.json de un clip (autocontenida, por clip).
# ---------------------------------------------------------------------------

def load_clip_info(clip_dir):
    """Lee el ``info.json`` de UN clip y lo deja listo para ``compute_pressure``.

    Lee solo el JSON de ESE clip (no carga el resto del partido), descarta los
    campos no usados (``video_url``, ``start_time``, ``end_time``,
    ``tournament``) y normaliza ``player_1_position`` recortando espacios, ya
    que algunos clips lo traen como ``"bottom "`` y la comparacion exacta de
    :func:`resolve_pressure_for_row` fallaria.

    Args:
        clip_dir: ruta a la carpeta del clip (``Dataset_Clutch/<clip>/``).

    Returns:
        dict con la info del clip, ya saneada.

    Raises:
        FileNotFoundError: si el clip no tiene ``info.json``.
    """
    info_path = Path(clip_dir) / config.CLIP_INFO_FILENAME
    if not info_path.exists():
        raise FileNotFoundError(f"El clip {Path(clip_dir).name} no tiene {config.CLIP_INFO_FILENAME}: {info_path}")

    with open(info_path, encoding="utf-8") as fh:
        info = json.load(fh)

    for field in _IGNORED_FIELDS:
        info.pop(field, None)

    # Normalizar la posicion de player_1: algunos clips traen "bottom " (con
    # espacio). resolve_pressure_for_row compara exacto contra "top"/"bottom".
    if isinstance(info.get("player_1_position"), str):
        info["player_1_position"] = info["player_1_position"].strip()

    return info
