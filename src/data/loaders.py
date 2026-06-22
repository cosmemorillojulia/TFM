"""Carga de datos del dataset de tenis.

Funciones para descubrir los clips de un game (subcarpetas con frames .jpg) y
listar los frames de un clip en orden. Logica portada de las celdas de
exploracion de ``01_player_tracking`` y ``02_ball_tracking``.
"""

import re
from pathlib import Path

# Un frame es un fichero NNNN.jpg (p.ej. 0000.jpg, 0153.jpg).
_FRAME_RE = re.compile(r"^\d+\.jpg$", re.IGNORECASE)


def list_clips(game_path):
    """Devuelve los directorios de clips de un game (subcarpetas con frames .jpg).

    Args:
        game_path: ruta a la carpeta de un game (p.ej. ``Dataset/game1``).

    Returns:
        Lista de ``Path`` de clips ordenados de forma natural (Clip1, Clip2,
        ..., Clip10) gracias a ordenar por el numero final del nombre.
    """
    game_path = Path(game_path)
    if not game_path.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del game: {game_path}")

    clips = [
        d for d in game_path.iterdir()
        if d.is_dir() and list_frames(d)
    ]
    # Orden natural por el numero del nombre del clip ("Clip2" antes que "Clip10").
    clips.sort(key=lambda d: _clip_number(d.name))
    return clips


def _clip_number(clip_name):
    """Extrae el numero final del nombre de un clip para ordenar de forma natural."""
    match = re.search(r"(\d+)$", clip_name)
    return int(match.group(1)) if match else 0


def list_frames(clip_dir):
    """Lista los frames ``NNNN.jpg`` de un clip ordenados por numero.

    Args:
        clip_dir: ruta a la carpeta de un clip.

    Returns:
        Lista de ``Path`` de frames ordenada por el numero del nombre.
    """
    clip_dir = Path(clip_dir)
    files = [p for p in clip_dir.iterdir() if _FRAME_RE.match(p.name)]
    files.sort(key=lambda p: int(p.stem))
    return files


def clip_key(clip_dir):
    """Devuelve la clave legible de un clip como ``"<game>/<clip>"``.

    Por ejemplo, para ``Dataset/game1/Clip1`` devuelve ``"game1/Clip1"``. Es el
    valor que se guarda en la columna ``clip`` de los CSV acumulados.
    """
    clip_dir = Path(clip_dir)
    return f"{clip_dir.parent.name}/{clip_dir.name}"
