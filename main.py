"""Orquestador del sistema de analisis de tenis por Computer Vision.

Procesa un game entero (todos sus clips) y genera los CSV acumulados, las
coordenadas proyectadas a metros y los mapas de calor.

Uso:
    python main.py --game-path Dataset/game1
    python main.py --game-path Dataset/game1 --output-dir outputs/game1_v2 --excel

El pipeline es incremental y reanudable: nunca borra ni sobreescribe outputs
existentes bajo --output-dir. Si una ejecucion anterior dejo resultados
parciales, se continua desde donde se quedo. Usar distintos --output-dir sobre el
mismo --game-path permite analizar el mismo game con varias configuraciones sin
interferencias.

Flujo (ver detalle en src/pipeline/orchestrator.py):
    1. Resolver la carpeta de salida del run (por defecto outputs/<game>).
    2. Procesar el game por etapas con cache (YOLO -> ball -> maps):
         yolo/<clip>.csv  ->  ball/<clip>.csv  ->  maps/ + CSV agregados.
    3. (Opcional) exportar los CSV agregados a Excel.
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from src.pipeline import orchestrator


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
        help="Nombre de la carpeta de salida (no ruta). Siempre se crea bajo "
             "OUTPUTS_ROOT (/workspace/TFM/outputs). Solo se usa su ultimo segmento: "
             "'/workspace/outputs/game8' -> OUTPUTS_ROOT/game8. Por defecto: "
             "OUTPUTS_ROOT/<nombre_del_game>. El pipeline es reanudable: reutiliza "
             "lo ya generado y no borra nada.",
    )
    parser.add_argument(
        "--excel", action="store_true",
        help="Exportar tambien los CSV master a un Excel al terminar.",
    )
    parser.add_argument(
        "--force-plots", action="store_true",
        help="Rehacer los plots (heatmaps generales y por jugador) aunque ya "
             "existan. No toca tracking ni videos ya generados. Util al iterar el "
             "estilo de los graficos.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto INFO).",
    )
    return parser.parse_args(argv)


def resolve_output_dir(game_path, output_dir):
    """Determina la carpeta de salida del run, siempre bajo ``OUTPUTS_ROOT``.

    La salida vive SIEMPRE dentro de ``OUTPUTS_ROOT`` (``/workspace/TFM/outputs``):

    - Sin ``--output-dir``: ``OUTPUTS_ROOT/<nombre_del_game>``.
    - Con ``--output-dir``: se ancla bajo ``OUTPUTS_ROOT`` usando solo el ULTIMO
      segmento del valor pasado. Asi, ``--output-dir /workspace/outputs/game8`` o
      ``--output-dir game8_v2`` acaban en ``OUTPUTS_ROOT/game8`` y
      ``OUTPUTS_ROOT/game8_v2`` respectivamente. Esto evita escribir fuera del
      proyecto por error (p.ej. en ``/workspace/outputs``).
    """
    if output_dir:
        return config.OUTPUTS_ROOT / Path(output_dir).name
    return config.OUTPUTS_ROOT / Path(game_path).name


def _setup_logging(level, log_file):
    """Configura el logging a consola y a un fichero dentro de la carpeta del run."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Modo append: el pipeline es reanudable, no se borra el log de runs previos.
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main(argv=None):
    args = parse_args(argv)

    game_path = Path(args.game_path)
    if not game_path.is_dir():
        # Aun sin logging configurado; usar uno minimo para el mensaje de error.
        _setup_logging(args.log_level, log_file=None)
        logging.getLogger("main").error("No existe la carpeta del game: %s", game_path)
        return 1

    output_dir = resolve_output_dir(game_path, args.output_dir)

    # El log del run vive dentro de la carpeta de salida (outputs/<game>/run.log).
    _setup_logging(args.log_level, output_dir / config.RUN_LOG_FILENAME)
    logger = logging.getLogger("main")
    logger.info("Game: %s | salida: %s", game_path, output_dir)

    # Ejecutar el flujo incremental/reanudable del game. No se borra nada:
    # las etapas ya completas se saltan y solo se computa lo que falte.
    output_dir.mkdir(parents=True, exist_ok=True)
    orchestrator.run_game(game_path, output_dir, export_excel=args.excel,
                          force_plots=args.force_plots)

    logger.info("Hecho.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
