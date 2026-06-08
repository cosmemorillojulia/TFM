"""Utilidades de entrada/salida para el pipeline.

Gestiona la escritura acumulada de los CSV master (siempre por append, nunca
sobrescribiendo a mitad de un game), el reseteo de los artefactos de un run
antes de reprocesar (preservando el cache de esquinas de pista) y la
exportacion final a Excel.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


def append_csv(df, csv_path):
    """Anade ``df`` a un CSV, escribiendo la cabecera solo si el fichero no existe.

    Esta es la operacion de acumulacion del pipeline: cada clip procesado se
    anade al CSV master del game. Nunca sobrescribe contenido previo.

    Args:
        df: ``pandas.DataFrame`` con las filas a anadir.
        csv_path: ruta del CSV master destino.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)
    logger.debug("Anadidas %d filas a %s (cabecera=%s)", len(df), csv_path.name, write_header)


def reset_run_outputs(output_dir):
    """Prepara la carpeta de salida de un game, borrando solo los artefactos del run.

    Si la carpeta ya existe, se eliminan por completo las subcarpetas
    ``tracking/``, ``projected/`` y ``heatmaps/`` para reescribirlas desde cero
    en esta ejecucion (evita CSV con filas duplicadas al reprocesar). El cache
    de esquinas ``court_points.json`` que vive en la raiz de la carpeta NUNCA se
    toca, para no obligar al usuario a reseleccionar las esquinas a mano.

    Args:
        output_dir: carpeta de salida del game (``outputs/<game>``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subdir_name in (config.TRACKING_SUBDIR, config.PROJECTED_SUBDIR, config.HEATMAPS_SUBDIR):
        subdir = output_dir / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)
            logger.info("Borrada carpeta de run previa: %s", subdir)
        subdir.mkdir(parents=True, exist_ok=True)

    court_points = output_dir / config.COURT_POINTS_FILENAME
    if court_points.exists():
        logger.info("Conservado cache de esquinas: %s", court_points)


def export_excel(csv_to_sheet, excel_path):
    """Exporta varios CSV a un unico Excel, una hoja por CSV.

    Se usa una sola vez al final del pipeline (la acumulacion durante el bucle es
    siempre por CSV). Requiere ``openpyxl``.

    Args:
        csv_to_sheet: dict ``{ruta_csv: nombre_hoja}``. Los CSV que no existan
            se omiten con un aviso.
        excel_path: ruta del ``.xlsx`` de salida.
    """
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        wrote_any = False
        for csv_path, sheet_name in csv_to_sheet.items():
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.warning("No se exporta a Excel (falta el CSV): %s", csv_path)
                continue
            pd.read_csv(csv_path).to_excel(writer, sheet_name=sheet_name, index=False)
            wrote_any = True
        if not wrote_any:
            # ExcelWriter falla al cerrar si no se escribio ninguna hoja.
            pd.DataFrame().to_excel(writer, sheet_name="vacio", index=False)
            logger.warning("Excel exportado sin datos: no habia CSV de origen.")

    logger.info("Excel exportado en %s", excel_path)
