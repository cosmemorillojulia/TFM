"""Configuracion central del proyecto.

Todas las rutas, parametros y constantes del sistema viven aqui. Ningun modulo
de ``src/`` debe contener rutas ni umbrales hardcodeados: deben importarlos de
este fichero. Los valores se han portado tal cual de las celdas de
configuracion de los notebooks originales (ahora en ``old_notebooks/``).
"""

import os
from pathlib import Path

# ===========================================================================
# Rutas base
# ===========================================================================
# En local las tres rutas se resuelven relativas al proyecto.
# En cloud (Colab, Kaggle, Lightning…) se pueden sobreescribir mediante
# variables de entorno antes de lanzar el script:
#
#   export TFM_PROJECT_ROOT=/content/ProyectoTFM
#   export TFM_DATASET_ROOT=/content/drive/MyDrive/TFM/Dataset
#   export TFM_OUTPUTS_ROOT=/content/drive/MyDrive/TFM/outputs
#
# Solo hace falta definir las que difieren del default local.

# Raiz del proyecto (carpeta que contiene este fichero).
PROJECT_ROOT = Path(os.environ.get("TFM_PROJECT_ROOT", Path(__file__).resolve().parent))

# Dataset de tenis (no versionado). Estructura: gameN/ClipM/{*.jpg}.
DATASET_ROOT = Path(os.environ.get("TFM_DATASET_ROOT", PROJECT_ROOT / "Dataset"))

# Raiz de salidas. Dentro vive el modelo global y una carpeta por game procesado.
OUTPUTS_ROOT = Path(os.environ.get("TFM_OUTPUTS_ROOT", PROJECT_ROOT / "outputs"))

# Modelo TrackNet ya entrenado. Es GLOBAL y compartido entre ejecuciones; el
# proyecto solo lo carga para inferencia (aqui NO se entrena).
MODELS_DIR = OUTPUTS_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "tracknet_best.pth"

# Subcarpetas que cuelgan de la carpeta de cada game (run). El pipeline es
# incremental y reanudable, pero acumula en CSV master (no un fichero por clip):
#   <output-dir>/tracking/   -> players_master.csv, ball_master.csv (acumulados)
#   <output-dir>/projected/  -> player_real_coords.csv, ball_real_coords.csv (metros)
#   <output-dir>/plots/      -> heatmaps, mapa de rebotes, vista combinada
#   <output-dir>/videos/     -> un mp4 por clip con las predicciones dibujadas
TRACKING_SUBDIR = "tracking"
PROJECTED_SUBDIR = "projected"
PLOTS_SUBDIR = "plots"
VIDEOS_SUBDIR = "videos"

# La reanudacion se apoya en ficheros sentinela vacios: cuando un clip termina
# una etapa, se crea tracking/.done_players/<clip> o .done_ball/<clip>. Asi el
# pipeline sabe que clips ya estan acumulados en el master sin reprocesarlos ni
# duplicar filas (no podemos inferirlo del master porque es un CSV concatenado).
DONE_PLAYERS_DIRNAME = ".done_players"
DONE_BALL_DIRNAME = ".done_ball"

# Subcarpeta del flujo antiguo. Se conserva por compatibilidad con
# reset_run_outputs, que ya no se usa en el flujo por defecto.
HEATMAPS_SUBDIR = "heatmaps"

# Video de predicciones.
# Los clips fuente estan grabados a 60fps. Se renderiza a MENOS fps a proposito
# (camara lenta) para poder ver bien las clasificaciones (botes, dentro/fuera).
# 40fps -> reproduccion al ~0.67x de la velocidad real.
VIDEO_FPS = 40                    # fotogramas por segundo del mp4 de salida
VIDEO_FOURCC = "mp4v"             # codec de cv2.VideoWriter (mp4v -> .mp4)
MINIMAP_WIDTH_PX = 190            # ancho del minimapa cenital incrustado en el video
MINIMAP_ALPHA = 0.55              # opacidad del minimapa (0=invisible, 1=opaco)
# Margenes (metros) del minimapa. Y (fondo) mayor que X (lateral) porque los
# jugadores sacan/restan por detras de la linea de fondo y se saldrian del mapa.
MINIMAP_MARGIN_Y = 5.0
MINIMAP_MARGIN_X = 2.0

# Nombre del fichero de cache de las esquinas de pista. Vive en la raiz de la
# carpeta del game (outputs/<game>/court_points.json) y NUNCA se borra al
# reescribir un run, para no tener que reseleccionar las esquinas a mano.
COURT_POINTS_FILENAME = "court_points.json"

# Nombres de los CSV acumulados (master) que se generan por game.
PLAYERS_MASTER_CSV = "players_master.csv"
BALL_MASTER_CSV = "ball_master.csv"

# CSV de presion a nivel de PUNTO (un clip = un punto = una fila). Acumula la
# salida directa de compute_pressure (pressure_p1, pressure_p2) por clip, para
# consultar de un vistazo que puntos tuvieron presion para uno, para el otro,
# para ambos (tiebreak) o para ninguno, sin reconstruirlo desde players_master.
POINTS_PRESSURE_CSV = "points_pressure.csv"

# Nombre del fichero de metadatos por clip dentro de Dataset_Clutch/<clip>/.
CLIP_INFO_FILENAME = "info.json"

# Nombres de los CSV proyectados a metros.
PLAYER_REAL_CSV = "player_real_coords.csv"
BALL_REAL_CSV = "ball_real_coords.csv"

# Nombre del Excel de exportacion final.
EXCEL_FILENAME = "resultados.xlsx"

# Nombre del fichero de log del run. Se escribe dentro de la carpeta de salida
# del game (outputs/<game>/run.log).
RUN_LOG_FILENAME = "run.log"


# ===========================================================================
# Tracking de jugadores (YOLOv8 + ByteTrack)  -- portado de 01_player_tracking
# ===========================================================================

# Modelo YOLO y parametros de inferencia.
# yolo11m: mejor deteccion de personas que yolov8s; en GPU (L4) va sobrado.
# imgsz=960: buen compromiso. Con GPU se puede subir a 1280; en CPU bajar a 640.
YOLO_MODEL = "yolo11m.pt"
IMGSZ = 960
PERSON_CLASS_ID = 0
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
TRACKER_CFG = "bytetrack.yaml"

# Tamano de lote para la inferencia YOLO. Los frames de un clip se procesan en
# grupos de este tamano para acotar el pico de VRAM (pasar el clip entero de
# golpe provocaba CUDA OOM en clips largos con yolo11m). 32 es seguro en una L4
# de 22GB; subir acelera algo a costa de mas memoria, bajar si sigue habiendo OOM.
YOLO_BATCH = 32

# Submuestreo de frames. FRAME_STEP=1 procesa todos los frames; valores mayores
# aceleran la inferencia y los frames saltados se reconstruyen por interpolacion.
FRAME_STEP = 1

# Filtro geometrico (fracciones 0.0-1.0 de la imagen). Calibrado sobre
# game1/Clip1 a 1280x720; en cada clip se multiplica por su resolucion real.
X_BAND_LEFT_FRAC = 0.15
X_BAND_RIGHT_FRAC = 0.85
Y_TOP_FRAC = 0.10
Y_BOTTOM_FRAC = 0.97
MID_LINE_Y_FRAC = 0.55
MIN_BBOX_AREA = 1500

# Zonas de exclusion [x_min, y_min, x_max, y_max] en fracciones de la imagen.
# Se descartan detecciones cuyo PUNTO DE PIE caiga dentro (recogepelotas,
# jueces de red y jueces de fondo).
EXCLUSION_ZONES = [
    [0.14, 0.28, 0.25, 0.48],   # recoge-pelotas / juez izq. de red
    [0.75, 0.28, 0.86, 0.48],   # recoge-pelotas / juez der. de red
    # Jueces de fondo: franja estrecha pegada al borde superior (10-14% Y),
    # solo zona central. Reducida para no tapar al jugador retrasado
    # (cuyo pie cae aprox. al 18-22% Y).
    [0.32, 0.10, 0.68, 0.14],
]

# ---------------------------------------------------------------------------
# Filtro temporal de jugadores (distingue jugadores de jueces por movimiento).
# ---------------------------------------------------------------------------
# Idea: a lo largo de un clip los jugadores recorren metros, mientras los jueces
# de silla/linea y recogepelotas permanecen casi estaticos. Acumulando las
# detecciones por track_id de ByteTrack y midiendo el desplazamiento total del
# punto de pie, los jueces (poco movimiento) se descartan sin depender solo de
# zonas hardcodeadas.

# Desplazamiento minimo del punto de pie (en fraccion de la diagonal de la
# imagen) para que un track se considere candidato a jugador. Un juez de silla
# se mueve muy por debajo de esto; un jugador, muy por encima.
MIN_TRACK_DISPLACEMENT_FRAC = 0.08

# Fraccion minima de frames del clip en los que el track debe aparecer. Filtra
# tracks fugaces (recogepelotas que cruzan, falsos positivos intermitentes).
MIN_TRACK_PRESENCE_FRAC = 0.15


# ===========================================================================
# Tracking de pelota (TrackNet)  -- portado de 02_ball_tracking
# ===========================================================================

# Resolucion de trabajo de la red.
INPUT_W = 640
INPUT_H = 360

# Numero de frames consecutivos apilados como entrada (3 frames -> 9 canales).
N_INPUT_FRAMES = 3

# Sigma (px, en la resolucion de trabajo) del heatmap gaussiano objetivo. Solo
# se usa para construir objetivos en entrenamiento; en inferencia no interviene,
# pero se conserva por coherencia con el checkpoint guardado.
GAUSSIAN_SIGMA = 5.0

# Umbral del pico del heatmap para considerar que hay pelota detectada.
HEATMAP_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Deteccion de botes reales (suelo) a partir de la trayectoria de la pelota.
# ---------------------------------------------------------------------------
# Prominencia minima (px) del pico de bajada->subida frente a los valles LOCALES
# mas cercanos (no el minimo global del clip). Bajado respecto al valor previo
# (20) porque exigia demasiado y perdia botes reales con rebotes poco marcados.
BOUNCE_MIN_PROMINENCE = 8

# Separacion minima en frames entre dos botes reales consecutivos del mismo clip.
BOUNCE_MIN_DISTANCE = 8

# Radio (px) alrededor de la caja de un jugador en el que un candidato a bote se
# descarta por ser, con mas probabilidad, un golpe de raqueta en el aire (el
# golpe ocurre junto al jugador; un bote en el suelo cae lejos de el, a media
# pista). Se aplica sobre la caja (bbox) de jugador, no solo el punto de pie,
# porque el golpe se da a la altura del brazo/raqueta, por encima del pie.
BOUNCE_PLAYER_EXCLUSION_PX = 60


# ===========================================================================
# Homografia y pista ITF  -- portado de 03_homography
# ===========================================================================

# Tipo de pista: "singles" o "doubles".
COURT_TYPE = "singles"

# Dimensiones reales de pista ITF (metros).
COURT_LENGTH = 23.77              # largo (eje Y en mundo)
COURT_WIDTH_SINGLES = 8.23
COURT_WIDTH_DOUBLES = 10.97
COURT_WIDTH = COURT_WIDTH_SINGLES if COURT_TYPE == "singles" else COURT_WIDTH_DOUBLES

# Referencias visuales de la pista (metros).
SERVICE_LINE_FROM_NET = 6.40      # distancia red <-> linea de saque
NET_Y = COURT_LENGTH / 2.0        # la red parte la pista por la mitad del largo

# Frame de referencia dentro de cada clip para seleccionar las esquinas.
REFERENCE_FRAME = "0000.jpg"


# ===========================================================================
# Visualizacion / heatmaps  -- portado de 04_heatmap
# ===========================================================================

# Margenes (metros) del canvas alrededor de la pista.
# MARGIN_Y: detras de las lineas de fondo (los jugadores retroceden 3-6 m).
# MARGIN_X: fuera de las lineas laterales (1-3 m).
MARGIN_Y = 5.0
MARGIN_X = 3.0

# Parametros del KDE y del render.
GRID_RES = 200                    # nº de celdas a lo largo del eje Y
KDE_BANDWIDTH = "scott"           # bandwidth de scipy.stats.gaussian_kde
PLAYER_CMAP = "magma"
BALL_CMAP = "hot"
DPI_EXPORT = 300

# Color de fondo de la pista en los plots.
COURT_FACECOLOR = "#2e6b3a"
