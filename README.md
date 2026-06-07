# TFM — Análisis de rendimiento en tenis mediante Computer Vision

Sistema que, a partir de los vídeos (secuencias de frames) de un partido de tenis, **detecta y trackea a los jugadores y la pelota**, proyecta sus posiciones a coordenadas reales de la pista (metros) y genera **mapas de calor** de ocupación y de rebotes.

El proyecto está organizado como un **pipeline de 4 notebooks** que se ejecutan en orden. Cada notebook lee los artefactos (CSV / PNG) que produjo el anterior en la carpeta `outputs/`, de modo que las etapas están desacopladas y se pueden re-ejecutar de forma independiente.

```
Frames de vídeo
      │
      ▼
[1] Tracking de jugadores ──▶ player_positions.csv
      │
[2] Tracking de pelota ─────▶ ball_positions.csv   (+ modelo TrackNet entrenado)
      │
[3] Homografía ────────────▶ *_real_coords.csv     (píxeles ▶ metros)
      │
[4] Heatmaps ──────────────▶ *.png                 (mapas de calor sobre la pista 2D)
```

---

## Pipeline

| # | Notebook | Qué hace | Técnica | Salida principal |
|---|----------|----------|---------|------------------|
| 1 | `01_player_tracking.ipynb` | Detecta y trackea a los 2 jugadores en cada frame. Filtra detecciones espurias (recogepelotas, jueces, árbitro) con bandas geométricas y zonas de exclusión, interpola los frames saltados y asigna etiquetas estables `player_top` / `player_bottom`. | YOLOv8s + ByteTrack | `outputs/tracking/player_positions.csv` |
| 2 | `02_ball_tracking.ipynb` | Entrena una red **TrackNet** sobre el dataset y, con el modelo entrenado, predice la posición de la pelota frame a frame. | TrackNet (encoder VGG16 + decoder U-Net, salida heatmap) | `outputs/models/tracknet_best.pth`, `outputs/tracking/ball_positions.csv` |
| 3 | `03_homography.ipynb` | Calcula la homografía píxel→mundo a partir de las 4 esquinas de la pista (seleccionadas manualmente) y proyecta las posiciones de jugadores y pelota a metros sobre el plano real de la pista ITF. | `cv2.findHomography` | `outputs/projected/player_real_coords.csv`, `ball_real_coords.csv` |
| 4 | `04_heatmap.ipynb` | Genera los mapas de calor finales: densidad de ocupación por jugador (KDE 2D) y mapa de rebotes de la pelota, superpuestos sobre una pista 2D dibujada a escala. | `scipy.stats.gaussian_kde` | `outputs/heatmaps/*.png` |

> Cada notebook está pensado para ejecutarse con **`Kernel → Restart & Run All`**. Toda la configuración (rutas, parámetros, clip de prueba) vive en la celda **"1. Configuración"** al principio de cada uno.

### Detalle de la red de pelota (TrackNet)

- **Entrada:** 3 frames consecutivos `(t-2, t-1, t)` apilados como tensor de 9 canales (RGB×3), a resolución 640×360.
- **Salida:** un heatmap 2D donde la pelota es un pico gaussiano (σ = 5 px); la posición se extrae como el píxel de máxima activación.
- **Arquitectura:** encoder VGG16 preentrenado en ImageNet (primera capa adaptada de 3 a 9 canales) + decoder tipo U-Net con skip connections.
- **Entrenamiento:** se hace por GPU en Google Colab mediante el notebook auxiliar `02-0_model_ball_train.ipynb` (variante de `02` con montaje de Drive, *mixed precision* y *loss* ponderada). El checkpoint resultante (`tracknet_best.pth`) se copia a `outputs/models/`.

---

## Estructura del proyecto

```
ProyectoTFM/
├── 01_player_tracking.ipynb        # Etapa 1 — jugadores (YOLOv8 + ByteTrack)
├── 02_ball_tracking.ipynb          # Etapa 2 — pelota (TrackNet: entreno + inferencia, local)
├── 02-0_model_ball_train.ipynb     # Etapa 2 — entrenamiento de TrackNet en Google Colab (GPU)
├── 03_homography.ipynb             # Etapa 3 — proyección píxel ▶ metros
├── 04_heatmap.ipynb                # Etapa 4 — mapas de calor
├── Easy_Demo_CV_With_Pretrained_YOLO.ipynb   # Demo independiente (no forma parte del pipeline)
│
├── Dataset/                        # Dataset TrackNet de tenis (no versionado)
│   ├── game1/ … game10/
│   │   └── ClipN/
│   │       ├── 0000.jpg … NNNN.jpg # Frames del clip
│   │       └── Label.csv           # Etiquetas de la pelota por frame
│   └── Readme.docx
│
├── outputs/                        # Artefactos generados por el pipeline
│   ├── tracking/                   # player_positions.csv, ball_positions.csv
│   ├── models/                     # tracknet_best.pth
│   ├── homography/                 # court_points.json (esquinas de pista)
│   ├── projected/                  # *_real_coords.csv (coordenadas en metros)
│   └── heatmaps/                   # *.png (mapas de calor)
│
├── yolov8s.pt                      # Pesos YOLOv8s (no versionado)
└── README.md
```

### El dataset

Se usa el **TrackNet tennis dataset**: 10 partidos (`game1`–`game10`) divididos en clips, ~15 700 frames de 1280×720 en total. Cada clip incluye sus frames como `.jpg` y un `Label.csv` con una fila por frame:

| Columna | Significado |
|---------|-------------|
| `file name` | Nombre del frame (`0000.jpg`, …) |
| `visibility` | 1 = pelota visible, 0 = no visible |
| `x-coordinate`, `y-coordinate` | Posición de la pelota en píxeles |
| `status` | Estado de la jugada (`1` = rebote) |

---

## Requisitos

- **Python 3.9+** y Jupyter / VS Code.
- Para el tracking de jugadores (notebook 1): `ultralytics`, `opencv-python`, `torch`, `pandas`, `numpy`, `matplotlib`, `lap`.
- Para la pelota (notebook 2): `torch`, `torchvision`, `opencv-python`, `numpy`, `pandas`, `matplotlib`.
- Para homografía y heatmaps (3 y 4): `opencv-python`, `numpy`, `pandas`, `scipy`, `matplotlib`.

```bash
pip install ultralytics opencv-python torch torchvision pandas numpy scipy matplotlib lap
```

> **GPU:** los notebooks funcionan en CPU, pero el **entrenamiento de TrackNet es muy lento sin GPU**. Por eso el entrenamiento se realiza en Google Colab (`02-0_model_ball_train.ipynb`); el resto de etapas corren bien en local.

---

## Cómo ejecutar

1. Coloca el dataset en `Dataset/` (un directorio por partido, con clips y su `Label.csv`).
2. Asegúrate de tener `yolov8s.pt` en la raíz (se descarga automáticamente la primera vez que `ultralytics` lo necesita).
3. Ejecuta los notebooks **en orden** (1 → 2 → 3 → 4), cada uno con *Restart & Run All*. En el notebook 3 deberás **clicar las 4 esquinas de la pista** la primera vez (luego se guardan en `outputs/homography/court_points.json`).
4. Los resultados aparecen en `outputs/`: los mapas de calor finales, en `outputs/heatmaps/`.

> El clip de referencia para la proyección es `game1/Clip1`. Si cambias de clip, recuerda re-seleccionar las esquinas de la pista en el notebook 3.

---

## Salidas

- **`outputs/tracking/player_positions.csv`** — posición del punto de pie y bounding box de cada jugador, por frame.
- **`outputs/tracking/ball_positions.csv`** — posición de la pelota por frame (con `visibility` e `is_bounce`).
- **`outputs/projected/*_real_coords.csv`** — las mismas posiciones proyectadas a metros sobre la pista.
- **`outputs/heatmaps/`** — `player_1_heatmap.png`, `player_2_heatmap.png`, `ball_bounces_map.png` y `combined_view.png`.
