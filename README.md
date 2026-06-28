# Análisis de Rendimiento en Tenis mediante Computer Vision

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-TrackNet-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#)

> Trabajo de Fin de Máster — Universidad de Navarra (UNAV)

Pipeline de visión por computador que, a partir de los vídeos (secuencias de frames) de
un partido de tenis, **detecta y trackea a los jugadores y la pelota**, proyecta sus
posiciones a coordenadas reales de la pista (metros) y genera **mapas de calor** de
ocupación y de rebotes, además de un **análisis bajo presión** (puntos clave:
break/set/match points) por jugador y un **vídeo** por clip con las predicciones dibujadas.

El proyecto era originalmente un conjunto de 4 notebooks de Jupyter encadenados. Se ha
reorganizado como un **proyecto Python modular** que se ejecuta de una sola orden:

```bash
python main.py --game-path Dataset_Clutch --output-dir outputs/final_v1
```

Este comando procesa el *game* entero (todos sus clips), acumula los resultados en CSV,
los proyecta a metros y genera los mapas de calor, el análisis de presión y los vídeos,
sin tocar nada más.

> Los notebooks originales se conservan, archivados, en [`old_notebooks/`](old_notebooks/).
> El proyecto **no entrena** ningún modelo en su ejecución normal: la red de pelota
> (TrackNet) ya entrenada se carga desde `outputs/models/tracknet_best.pth` y se usa solo
> para inferencia.

---

## Cómo ejecutar

1. **Instala las dependencias** (idealmente en un entorno virtual):
   ```bash
   pip install -r requirements.txt
   ```
   > Para GPU, instala la build CUDA de `torch`/`torchvision` desde
   > [pytorch.org](https://pytorch.org). El sistema funciona en CPU, pero la inferencia
   > de pelota es más lenta.

2. **Coloca el dataset** en `Dataset/` (un directorio por game, con clips y su `Label.csv`)
   y el **modelo entrenado** en `outputs/models/tracknet_best.pth`.

3. **Ejecuta** sobre un game:
   ```bash
   python main.py --game-path Dataset_Clutch --output-dir outputs/final_v1
   ```
   La **primera vez** por game se abrirá una ventana para que cliques las **4 esquinas de
   la pista** (en orden: inferior-izq, inferior-der, superior-der, superior-izq). Quedan
   cacheadas y no se vuelven a pedir.

   > También funciona con los games del dataset TrackNet clásico (`Dataset/game1`, …), pero
   > esos clips **no traen `info.json`**: en ese caso no hay análisis de presión ni nombres
   > reales, y los jugadores se etiquetan como `player_top` / `player_bottom`.

### Argumentos de `main.py`

| Argumento | Descripción |
|---|---|
| `--game-path` | (obligatorio) Carpeta del game a procesar, p.ej. `Dataset_Clutch`. |
| `--output-dir` | Carpeta de salida. Por defecto `outputs/<nombre_del_game>`. |
| `--excel` | Exporta además los CSV master a un Excel (`resultados.xlsx`). |
| `--force-plots` | Rehace los plots (heatmaps generales y por jugador) aunque ya existan. No toca tracking ni vídeos. Útil al iterar el estilo de los gráficos. |
| `--log-level` | Detalle del log: `DEBUG`, `INFO` (defecto), `WARNING`, `ERROR`. Ver tabla de niveles abajo. |

| Nivel | Qué muestra |
|---|---|
| `DEBUG` | Todo: cada frame procesado, detecciones individuales, valores intermedios. Muy verbose. |
| `INFO` | Progreso normal: qué clip se está procesando, cuántos frames detectados, artefactos generados. Es el útil para seguir la ejecución. |
| `WARNING` | Solo avisos: CSV que faltan, GPU no detectada, datos insuficientes para KDE. |
| `ERROR` | Solo errores graves que interrumpen el pipeline. |

> **Re-ejecución:** el pipeline es **incremental y reanudable a todos los niveles**. En
> tracking, cada clip deja un sentinela al terminar (`tracking/.done_players/`, `.done_ball/`)
> y los ya hechos se saltan sin duplicar filas. La etapa final solo corre cuando **todos**
> los clips están acumulados, y dentro de ella **cada sub-paso se salta si su salida ya
> existe** (detección por existencia de fichero): si solo faltan vídeos, solo se generan los
> vídeos; los plots ya hechos no se rehacen. Para forzar rehacer los plots, usa
> `--force-plots`. El cache de esquinas `court_points.json` **nunca** se borra.

---

## El flujo, paso a paso

El orquestador ([`src/pipeline/orchestrator.py`](src/pipeline/orchestrator.py)) ejecuta:

```
Dado --game-path Dataset_Clutch:

1. Descubrir todos los clips del game (Clip1, Clip2, ...).

2. Bucle CLIP A CLIP  (la unidad del bucle es el clip, no el frame: TrackNet
   necesita 3 frames consecutivos y ByteTrack continuidad temporal):
     a. player_tracker  -> posiciones de los 2 jugadores -> APPEND players_master.csv
     b. ball_tracker     -> posición de la pelota          -> APPEND ball_master.csv
     c. pressure         -> presión del punto (info.json)  -> APPEND points_pressure.csv

3. Tras TODOS los clips del game:
     a. homography  -> cargar/seleccionar las 4 esquinas, proyectar a metros
     b. ball_tracker.detect_real_bounces -> botes reales desde la trayectoria
     c. plots       -> mapas de calor de ocupación, mapa de rebotes y vista combinada
     d. pressure_compare -> por jugador, mapa movimiento+botes (general / con / sin presión)
     e. videos      -> un mp4 por clip con las predicciones dibujadas (siempre lo último)

4. (Opcional, --excel) exportar los CSV master a resultados.xlsx
```

Cada fila de los CSV master lleva las columnas `clip` y `frame`, de modo que cada
posición queda identificada de forma única y se puede filtrar por clip.

### Las etapas y la técnica de cada una

| Etapa | Módulo | Técnica |
|---|---|---|
| Tracking de jugadores | [`src/tracking/player_tracker.py`](src/tracking/player_tracker.py) | YOLO11m + ByteTrack, filtrado geométrico (zonas de exclusión de recogepelotas/jueces) y temporal (desplazamiento del track), interpolación y etiquetado `player_top`/`player_bottom` |
| Tracking de pelota | [`src/tracking/ball_tracker.py`](src/tracking/ball_tracker.py) | TrackNet (solo inferencia) sobre ventanas de 3 frames; los botes reales (`is_real_bounce`) se detectan de la **trayectoria** (picos de bajada→subida) y se clasifican dentro/fuera de pista |
| Homografía | [`src/geometry/homography.py`](src/geometry/homography.py) | `cv2.findHomography` (4 esquinas) y proyección píxel → metros |
| Presión | [`src/data/pressure.py`](src/data/pressure.py) | Deriva la presión por jugador del `info.json` del clip (break/set/match point, marcador) |
| Heatmaps | [`src/visualization/heatmaps.py`](src/visualization/heatmaps.py) | KDE 2D (`scipy.stats.gaussian_kde`) por jugador + mapa de rebotes + vista combinada |
| Plots por jugador | [`src/visualization/pressure_compare.py`](src/visualization/pressure_compare.py) | Mapa que combina movimiento (KDE) y botes del jugador, general y con/sin presión |
| Vídeos | [`src/visualization/video.py`](src/visualization/video.py) | Un mp4 por clip con jugadores, pelota, botes y minimapa cenital dibujados |

---

## Estructura del proyecto

```
ProyectoTFM/
├── main.py                     # Orquestador. python main.py --game-path Dataset_Clutch
├── config.py                   # TODAS las rutas, parámetros y constantes (pista ITF, TrackNet, KDE...)
├── requirements.txt
├── README.md
│
├── src/
│   ├── data/
│   │   ├── loaders.py          # Descubrir clips de un game, listar frames, leer Label.csv
│   │   └── pressure.py         # Presión por jugador a partir del info.json del clip
│   ├── models/tracknet.py      # Arquitectura TrackNet (encoder VGG16 9 canales + decoder U-Net)
│   ├── tracking/
│   │   ├── player_tracker.py   # YOLO11m + ByteTrack + filtrado/interpolación/etiquetado
│   │   └── ball_tracker.py     # Inferencia TrackNet + detección de botes reales. SOLO inferencia
│   ├── geometry/homography.py  # findHomography + proyección píxel→metros + selección de esquinas
│   ├── visualization/
│   │   ├── heatmaps.py         # KDE de jugadores + mapa de rebotes + vista combinada
│   │   ├── pressure_compare.py # Motor de los plots por jugador (movimiento+botes, con/sin presión)
│   │   └── video.py            # Render del mp4 por clip (predicciones + minimapa cenital)
│   ├── pipeline/orchestrator.py  # El flujo: llama en orden a cada etapa (jugadores→pelota→…→vídeos)
│   └── utils/io.py             # Append seguro a CSV, reseteo de run, export a Excel
│
├── old_notebooks/              # Los 4 notebooks originales + el de entrenamiento + demo (archivados)
│
├── outputs/
│   ├── models/
│   │   └── tracknet_best.pth   # Modelo entrenado (global, compartido entre ejecuciones)
│   └── <game>/                 # Una carpeta por game procesado (p.ej. final_v1)
│       ├── court_points.json   # Esquinas de pista cacheadas (NO se borra al reejecutar)
│       ├── tracking/           # players_master.csv, ball_master.csv, points_pressure.csv
│       ├── projected/          # *_real_coords.csv (coordenadas en metros)
│       ├── plots/              # combinados (sueltos) + una carpeta por jugador
│       └── videos/             # un mp4 por clip
│
└── Dataset_Clutch/             # Dataset de clips "clutch" (no versionado)
    └── ClipM/                  # Frames 0000.jpg... + Label.csv + info.json (metadatos del punto)
```

### Configuración

Toda la parametrización vive en [`config.py`](config.py): rutas, modelo y umbrales de YOLO,
bandas y zonas de exclusión del filtro de jugadores, resolución y umbral de TrackNet,
dimensiones de la pista ITF, márgenes y parámetros del KDE. Los módulos de `src/` no
contienen rutas ni umbrales hardcodeados: todo se importa de aquí.

### El dataset

Se usa **`Dataset_Clutch`**: clips de puntos "clutch" (`Clip1`, `Clip2`, …) de un partido.
Cada clip incluye sus frames como `.jpg`, un `Label.csv` con una fila por frame y un
`info.json` con los metadatos del punto.

`Label.csv` (una fila por frame):

| Columna | Significado |
|---------|-------------|
| `file name` | Nombre del frame (`0000.jpg`, …) |
| `visibility` | 1 = pelota visible, 0 = no visible |
| `x-coordinate`, `y-coordinate` | Posición de la pelota en píxeles |
| `status` | Estado de la jugada en la anotación original |

> Los botes que se usan en los análisis **no** se leen del `status`: se detectan de la
> trayectoria proyectada de la pelota (`ball_tracker.detect_real_bounces`).

`info.json` (metadatos del punto, base del análisis de presión y del etiquetado de lado):

| Campo | Significado |
|-------|-------------|
| `player_1`, `player_2` | Nombres reales de los jugadores |
| `player_1_position` | Lado de `player_1` en el clip (`top` / `bottom`); define la normalización |
| `server` | Quién saca (`player_1` / `player_2`) |
| `is_break_point`, `is_set_point`, `is_match_point`, `tie_break` | Contexto de presión del punto |
| `score_player1`, `score_player2`, `set_player1`, `set_player2` | Marcador, para decidir quién va por detrás |

---

## Salidas

Para un game procesado en `outputs/<game>/`:

- **`tracking/players_master.csv`** — punto de pie, bounding box y `pressure` de cada jugador, por clip y frame.
- **`tracking/ball_master.csv`** — posición de la pelota por clip y frame.
- **`tracking/points_pressure.csv`** — una fila por punto con `pressure_p1` / `pressure_p2`.
- **`projected/*_real_coords.csv`** — las mismas posiciones proyectadas a metros sobre la pista (la de pelota, solo los botes reales).
- **`plots/`** — mapas de calor:
  - sueltos, combinados de ambos jugadores: `players_combined_heatmap.png`, `ball_bounces_map.png`, `combined_view.png`.
  - una carpeta por jugador (`<jugador>/`) con `general.png`, `con_presion.png` y `sin_presion.png` (movimiento + sus botes).
- **`videos/`** — un `<clip>.mp4` por clip con las predicciones dibujadas.
- **`resultados.xlsx`** — (solo con `--excel`) los CSV master en un Excel, una hoja cada uno.

> Para **regenerar** los plots de un run ya procesado, relanza `main.py` sobre él con
> `--force-plots`: el tracking y los vídeos ya hechos se respetan y solo se rehacen los
> plots.

---

## Notas

- El entrenamiento de TrackNet **no forma parte de este pipeline**; se hizo aparte, desde
  [`model_ball_train.ipynb`](model_ball_train.ipynb), en una instancia remota de
  [RunPod](https://www.runpod.io/) con GPU **NVIDIA A100** para acelerar tanto el
  entrenamiento del modelo como las pruebas del pipeline completo.
- La selección interactiva de esquinas requiere un entorno con interfaz gráfica (la primera
  vez por game). En ejecuciones posteriores se reutiliza el `court_points.json` cacheado.
- Por defecto se usa una homografía por game (cámara consistente). El diseño está preparado
  para pasar a una homografía por clip cambiando la clave de cache, sin reescribir el módulo.
