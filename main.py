"""Orquestador del sistema de analisis de tenis por Computer Vision.

Procesa un game entero (todos sus clips) y genera los CSV acumulados, las
coordenadas proyectadas a metros y los mapas de calor.

Uso:
    python main.py --game-path Dataset/game1
    python main.py --game-path Dataset/game1 --output-dir outputs/game1_v2 --excel

Flujo (ver detalle en src/pipeline/orchestrator.py):
    1. Resolver la carpeta de salida del run (por defecto outputs/<game>).
    2. Limpiar los artefactos de un run previo (preservando el cache de esquinas).
    3. Procesar el game clip a clip:
         jugadores -> players_master.csv   |   pelota -> ball_master.csv
    4. Homografia del game -> proyectar a metros -> generar heatmaps.
    5. (Opcional) exportar los CSV a Excel.
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from src.pipeline import orchestrator
from src.utils import io


def parse_args(argv=None):
    """Define y parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Analisis de tenis por Computer Vision: procesa un game completo.",
    )
    parser.add_argument(
        "--game-path", required=True,
        help="Ruta a la carpeta de un game (p.ej. Dataset/game1).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Carpeta de salida del run. Por defecto: outputs/<nombre_del_game>. "
             "Si ya existe, se reescriben sus CSV y heatmaps (el cache de esquinas se conserva).",
    )
    parser.add_argument(
        "--excel", action="store_true",
        help="Exportar tambien los CSV master a un Excel al terminar.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto INFO).",
    )
    return parser.parse_args(argv)


def resolve_output_dir(game_path, output_dir):
    """Determina la carpeta de salida del run.

    Si el usuario no pasa ``--output-dir``, se usa ``outputs/<nombre_del_game>``.
    """
    if output_dir:
        return Path(output_dir)
    return config.OUTPUTS_ROOT / Path(game_path).name


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("main")

    game_path = Path(args.game_path)
    if not game_path.is_dir():
        logger.error("No existe la carpeta del game: %s", game_path)
        return 1

    output_dir = resolve_output_dir(game_path, args.output_dir)
    logger.info("Game: %s | salida: %s", game_path, output_dir)

    # Preparar la carpeta de salida (borra artefactos del run previo, conserva esquinas).
    io.reset_run_outputs(output_dir)

    # Ejecutar todo el flujo del game.
    orchestrator.run_game(game_path, output_dir, export_excel=args.excel)

    logger.info("Hecho.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
