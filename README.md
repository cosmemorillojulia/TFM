# TFM — Análisis de rendimiento en tenis mediante Computer Vision

Sistema que, a partir de los vídeos (secuencias de frames) de un partido de tenis,
**detecta y trackea a los jugadores y la pelota**, proyecta sus posiciones a coordenadas
reales de la pista (metros) y genera **mapas de calor** de ocupación y de rebotes.

El proyecto era originalmente un conjunto de 4 notebooks de Jupyter encadenados. Se ha
reorganizado como un **proyecto Python modular** que se ejecuta de una sola orden:

```bash
python main.py --game-path Dataset/game0 --output-dir outputs/game0
```

Este comando procesa el *game* entero (todos sus clips), acumula los resultados en CSV,
los proyecta a metros y genera los mapas de calor, sin tocar nada más.

> Los notebooks originales se conservan, archivados, en [`old_notebooks/`](old_notebooks/).
> El proyecto **no entrena** ningún modelo: la red de pelota (TrackNet) ya entrenada se
> carga desde `outputs/models/tracknet_best.pth` y se usa solo para inferencia.

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
   python main.py --game-path Dataset/game1
   ```
   La **primera vez** por game se abrirá una ventana para que cliques las **4 esquinas de
   la pista** (en orden: inferior-izq, inferior-der, superior-der, superior-izq). Quedan
   cacheadas y no se vuelven a pedir.

### Argumentos de `main.py`

| Argumento | Descripción |
|---|---|
| `--game-path` | (obligatorio) Carpeta del game a procesar, p.ej. `Dataset/game1`. |
| `--output-dir` | Carpeta de salida. Por defecto `outputs/<nombre_del_game>`. |
| `--excel` | Exporta además los CSV master a un Excel (`resultados.xlsx`). |
| `--log-level` | Detalle del log: `DEBUG`, `INFO` (defecto), `WARNING`, `ERROR`. Ver tabla de niveles abajo. |

| Nivel | Qué muestra |
|---|---|
| `DEBUG` | Todo: cada frame procesado, detecciones individuales, valores intermedios. Muy verbose. |
| `INFO` | Progreso normal: qué clip se está procesando, cuántos frames detectados, artefactos generados. Es el útil para seguir la ejecución. |
| `WARNING` | Solo avisos: CSV que faltan, GPU no detectada, datos insuficientes para KDE. |
| `ERROR` | Solo errores graves que interrumpen el pipeline. |

> **Re-ejecución:** si la carpeta de salida ya existe, se reescriben desde cero los CSV de
> `tracking/`, `projected/` y los PNG de `heatmaps/` (no se duplican filas). El cache de
> esquinas `court_points.json` **nunca** se borra.

---

## El flujo, paso a paso

El orquestador ([`src/pipeline/orchestrator.py`](src/pipeline/orchestrator.py)) ejecuta:

```
Dado --game-path Dataset/game1:

1. Descubrir todos los clips del game (Clip1, Clip2, ...).

2. Bucle CLIP A CLIP  (la unidad del bucle es el clip, no el frame: TrackNet
   necesita 3 frames consecutivos y ByteTrack continuidad temporal):
     a. player_tracker  -> posiciones de los 2 jugadores -> APPEND players_master.csv
     b. ball_tracker     -> posición de la pelota          -> APPEND ball_master.csv

3. Tras TODOS los clips del game:
     a. homography  -> cargar/seleccionar las 4 esquinas, proyectar a metros
     b. heatmaps    -> generar los PNG sobre la pista 2D

4. (Opcional, --excel) exportar los CSV master a resultados.xlsx
```

Cada fila de los CSV master lleva las columnas `clip` y `frame`, de modo que cada
posición queda identificada de forma única y se puede filtrar por clip.

### Las etapas y la técnica de cada una

| Etapa | Módulo | Técnica |
|---|---|---|
| Tracking de jugadores | [`src/tracking/player_tracker.py`](src/tracking/player_tracker.py) | YOLOv8s + ByteTrack, filtrado geométrico (zonas de exclusión de recogepelotas/jueces), interpolación y etiquetado `player_top`/`player_bottom` |
| Tracking de pelota | [`src/tracking/ball_tracker.py`](src/tracking/ball_tracker.py) | TrackNet (solo inferencia) sobre ventanas de 3 frames; el `is_bounce` se toma del `status` del `Label.csv` |
| Homografía | [`src/geometry/homography.py`](src/geometry/homography.py) | `cv2.findHomography` (4 esquinas) y proyección píxel → metros |
| Heatmaps | [`src/visualization/heatmaps.py`](src/visualization/heatmaps.py) | KDE 2D (`scipy.stats.gaussian_kde`) por jugador + mapa de rebotes |

---

## Estructura del proyecto

```
ProyectoTFM/
├── main.py                     # Orquestador. python main.py --game-path Dataset/game1
├── config.py                   # TODAS las rutas, parámetros y constantes (pista ITF, TrackNet, KDE...)
├── requirements.txt
├── README.md
│
├── src/
│   ├── data/loaders.py         # Descubrir clips de un game, listar frames, leer Label.csv
│   ├── models/tracknet.py      # Arquitectura TrackNet (encoder VGG16 9 canales + decoder U-Net)
│   ├── tracking/
│   │   ├── player_tracker.py   # YOLOv8 + ByteTrack + filtrado/interpolación/etiquetado
│   │   └── ball_tracker.py     # Inferencia TrackNet (ventana de 3 frames). SOLO inferencia
│   ├── geometry/homography.py  # findHomography + proyección píxel→metros + selección de esquinas
│   ├── visualization/heatmaps.py  # KDE de jugadores + mapa de rebotes + vista combinada
│   ├── pipeline/orchestrator.py   # Bucle clip a clip y escritura acumulada
│   └── utils/io.py             # Append seguro a CSV, reseteo de run, export a Excel
│
├── old_notebooks/              # Los 4 notebooks originales + el de entrenamiento + demo (archivados)
│
├── outputs/
│   ├── models/
│   │   └── tracknet_best.pth   # Modelo entrenado (global, compartido entre ejecuciones)
│   └── <game>/                 # Una carpeta por game procesado (p.ej. game1)
│       ├── court_points.json   # Esquinas de pista cacheadas (NO se borra al reejecutar)
│       ├── tracking/           # players_master.csv, ball_master.csv (acumulados)
│       ├── projected/          # *_real_coords.csv (coordenadas en metros)
│       └── heatmaps/           # *.png (mapas de calor)
│
└── Dataset/                    # Dataset TrackNet de tenis (no versionado)
    └── gameN/ClipM/            # Frames 0000.jpg... + Label.csv
```

### Configuración

Toda la parametrización vive en [`config.py`](config.py): rutas, modelo y umbrales de YOLO,
bandas y zonas de exclusión del filtro de jugadores, resolución y umbral de TrackNet,
dimensiones de la pista ITF, márgenes y parámetros del KDE. Los módulos de `src/` no
contienen rutas ni umbrales hardcodeados: todo se importa de aquí.

### El dataset

Se usa el **TrackNet tennis dataset**: partidos (`game1`, `game2`, …) divididos en clips.
Cada clip incluye sus frames como `.jpg` y un `Label.csv` con una fila por frame:

| Columna | Significado |
|---------|-------------|
| `file name` | Nombre del frame (`0000.jpg`, …) |
| `visibility` | 1 = pelota visible, 0 = no visible |
| `x-coordinate`, `y-coordinate` | Posición de la pelota en píxeles |
| `status` | Estado de la jugada (`1` = rebote) |

---

## Salidas

Para un game procesado en `outputs/<game>/`:

- **`tracking/players_master.csv`** — punto de pie y bounding box de cada jugador, por clip y frame.
- **`tracking/ball_master.csv`** — posición de la pelota por clip y frame (con `visibility` e `is_bounce`).
- **`projected/*_real_coords.csv`** — las mismas posiciones proyectadas a metros sobre la pista.
- **`heatmaps/`** — `player_top_heatmap.png`, `player_bottom_heatmap.png`, `ball_bounces_map.png` y `combined_view.png`.
- **`resultados.xlsx`** — (solo con `--excel`) los CSV master en un Excel, una hoja cada uno.

---

## Notas

- El entrenamiento de TrackNet **no forma parte de este proyecto**; se hizo en Google Colab
  con GPU. Ese notebook queda archivado en `old_notebooks/02-0_model_ball_train.ipynb`.
- La selección interactiva de esquinas requiere un entorno con interfaz gráfica (la primera
  vez por game). En ejecuciones posteriores se reutiliza el `court_points.json` cacheado.
- Por defecto se usa una homografía por game (cámara consistente). El diseño está preparado
  para pasar a una homografía por clip cambiando la clave de cache, sin reescribir el módulo.
```
