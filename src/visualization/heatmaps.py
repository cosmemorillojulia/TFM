"""Mapas de calor sobre la pista 2D en coordenadas reales (metros).

Genera, a partir de las coordenadas ya proyectadas a metros:
- un heatmap de ocupacion por jugador (KDE 2D),
- un mapa de rebotes de la pelota,
- una vista combinada con ambos jugadores y los rebotes.

La pista se dibuja a escala de forma programatica. Logica portada de
``old_notebooks/04_heatmap.ipynb`` (mismos cmaps, margenes, resolucion del KDE y
DPI de exportacion).
"""

import logging

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: el pipeline corre headless desde main.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from scipy.stats import gaussian_kde

import config

logger = logging.getLogger(__name__)

# Dimensiones del canvas (pista + margenes) y figura proporcional.
_CANVAS_W = config.COURT_WIDTH + 2 * config.MARGIN_X
_CANVAS_H = config.COURT_LENGTH + 2 * config.MARGIN_Y
_FIG_H = 10
_FIG_W = round(_FIG_H * _CANVAS_W / _CANVAS_H, 1)


def _make_grid(n=None):
    """Construye la rejilla regular (en metros) sobre la que se evalua el KDE."""
    n = n or config.GRID_RES
    width, length = config.COURT_WIDTH, config.COURT_LENGTH
    aspect = (width + 2 * config.MARGIN_X) / (length + 2 * config.MARGIN_Y)
    nx = max(20, int(round(n * aspect)))
    ny = n
    xs = np.linspace(-config.MARGIN_X, width + config.MARGIN_X, nx)
    ys = np.linspace(-config.MARGIN_Y, length + config.MARGIN_Y, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return xs, ys, grid_x, grid_y


def _draw_court_2d(ax, lw=1.6):
    """Dibuja las lineas de la pista de tenis (metros) sobre un eje matplotlib."""
    width, length = config.COURT_WIDTH, config.COURT_LENGTH
    s_near = config.NET_Y - config.SERVICE_LINE_FROM_NET
    s_far = config.NET_Y + config.SERVICE_LINE_FROM_NET
    lines = [
        ((0, 0), (width, 0)),
        ((width, 0), (width, length)),
        ((width, length), (0, length)),
        ((0, length), (0, 0)),
        ((0, s_near), (width, s_near)),
        ((0, s_far), (width, s_far)),
        ((width / 2, s_near), (width / 2, s_far)),
    ]
    for (p0, p1) in lines:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="white", lw=lw)
    # La red, mas gruesa.
    ax.plot([0, width], [config.NET_Y, config.NET_Y], color="white", lw=lw + 0.8)
    ax.set_xlim(-config.MARGIN_X, width + config.MARGIN_X)
    ax.set_ylim(-config.MARGIN_Y, length + config.MARGIN_Y)
    ax.set_aspect("equal")


def _kde_grid(real_x, real_y):
    """Evalua el KDE de una nube de puntos sobre la rejilla. Devuelve ``z`` o None.

    Devuelve None si no hay datos suficientes (menos de 2 puntos o varianza nula).
    El ancho de banda (``scott``) se calcula con la covarianza de ESTOS puntos, por
    lo que el detalle es el propio de cada jugador; para el mapa combinado se suman
    los ``z`` por jugador en vez de re-estimar un KDE sobre la nube global (que
    saldria mucho mas suavizado y no coincidiria con los heatmaps individuales).
    """
    pts = np.vstack([real_x, real_y])
    if pts.shape[1] < 2 or np.allclose(pts.std(axis=1), 0):
        return None
    _, _, grid_x, grid_y = _make_grid()
    kde = gaussian_kde(pts, bw_method=config.KDE_BANDWIDTH)
    return kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)


def _draw_kde_grid(ax, z, cmap, emphasize_low=False):
    """Pinta una densidad ``z`` ya evaluada en la rejilla. Devuelve la imagen o None.

    Si ``emphasize_low`` es True, se realza el rango bajo de densidad para que
    cualquier zona pisada alguna vez quede visible (no solo las mas frecuentadas):
    se aplica una normalizacion no lineal (``PowerNorm`` con gamma < 1) y se funde
    con el fondo negro unicamente la densidad practicamente nula.
    """
    if z is None:
        return None
    xs, ys, _, _ = _make_grid()

    if not emphasize_low:
        return ax.imshow(
            z, origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            cmap=cmap, alpha=0.85, aspect="equal",
        )

    # Realce del rango bajo: gamma<1 expande las densidades pequeñas hacia
    # valores de color visibles, manteniendo el gradiente hacia las zonas mas
    # frecuentadas. Lo que esta por debajo de un umbral minimo (zonas nunca
    # pisadas) se funde con el fondo negro de la pista.
    zmax = z.max()
    if zmax <= 0:
        return None
    thresh = zmax * config.HEATMAP_LOW_THRESHOLD
    z_masked = np.ma.masked_less(z, thresh)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(config.PLAYER_HEATMAP_FACECOLOR)  # sin pisar -> color de fondo
    return ax.imshow(
        z_masked, origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap=cmap_obj, aspect="equal",
        norm=PowerNorm(gamma=config.HEATMAP_GAMMA, vmin=thresh, vmax=zmax),
    )


def _kde_layer(ax, real_x, real_y, cmap, emphasize_low=False):
    """Evalua y pinta el KDE de una nube de puntos. Devuelve la imagen o None."""
    return _draw_kde_grid(ax, _kde_grid(real_x, real_y), cmap, emphasize_low)


def _render_occupancy_heatmap(z, title, out_path):
    """Dibuja y guarda un heatmap de ocupacion (fondo negro, rampa inferno).

    Recibe la densidad ``z`` ya evaluada en la rejilla (``_kde_grid``). Logica
    comun al heatmap de un jugador y al combinado de ambos: solo cambia como se
    obtiene ``z`` (KDE de un jugador, o suma de los KDE por jugador).
    """
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.set_facecolor(config.PLAYER_HEATMAP_FACECOLOR)
    im = _draw_kde_grid(ax, z, config.PLAYER_CMAP, emphasize_low=True)
    _draw_court_2d(ax)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Densidad de ocupación (norm.)")
        # La densidad KDE absoluta no es interpretable; normalizamos la escala a
        # [0, 1] (1 = celda mas frecuentada) con ticks numericos, convencion
        # estandar en analitica de tracking deportivo. Como la norma es no lineal
        # (PowerNorm), situamos los ticks en posiciones VISUALMENTE equiespaciadas
        # de la barra invirtiendo la norma, para que no queden huecos.
        fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
        cbar.set_ticks([im.norm.inverse(f) for f in fracs])
        cbar.set_ticklabels([f"{f:.2f}" for f in fracs])
    else:
        ax.text(0.5, 0.5, "datos insuficientes", transform=ax.transAxes,
                ha="center", va="center", color="white")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)


def export_player_heatmap(real_x, real_y, title, out_path):
    """Genera y guarda el heatmap KDE de ocupacion de un jugador."""
    _render_occupancy_heatmap(_kde_grid(real_x, real_y), title, out_path)
    logger.info("Heatmap de jugador guardado en %s", out_path)


def export_combined_player_heatmap(players_by_label, out_path,
                                   title="Heatmap de ocupación — ambos jugadores"):
    """Genera un unico heatmap de ocupacion con los dos jugadores en el mismo mapa.

    Para que sea coherente con los heatmaps individuales, NO se re-estima un KDE
    sobre la nube global de ambos jugadores (eso usaria un ancho de banda mucho
    mayor y saldria mas suavizado), sino que se SUMAN las densidades evaluadas por
    jugador: cada mitad conserva el mismo detalle que su heatmap individual. Como
    cada jugador ocupa su propio campo, sus densidades apenas se solapan.

    Args:
        players_by_label: dict ``{etiqueta: (real_x, real_y)}`` por jugador.
        out_path: ruta del PNG de salida.
        title: titulo del grafico.
    """
    z_total = None
    for real_x, real_y in players_by_label.values():
        z = _kde_grid(real_x, real_y)
        if z is None:
            continue
        z_total = z if z_total is None else z_total + z
    _render_occupancy_heatmap(z_total, title, out_path)
    logger.info("Heatmap combinado de jugadores guardado en %s", out_path)


def export_bounce_map(bounces_x, bounces_y, out_path, title="Rebotes de pelota"):
    """Genera y guarda el mapa de rebotes de la pelota (puntos + KDE de densidad)."""
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.set_facecolor(config.COURT_FACECOLOR)
    im = _kde_layer(ax, bounces_x, bounces_y, config.BALL_CMAP)
    _draw_court_2d(ax)
    ax.scatter(bounces_x, bounces_y, s=55, marker="X", color="cyan",
               edgecolors="white", linewidths=1.2, label=f"rebotes (n={len(bounces_x)})")
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("densidad de rebotes")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Mapa de rebotes guardado en %s", out_path)


def export_combined_view(players_by_label, bounces_x, bounces_y, out_path):
    """Genera una vista combinada: posiciones de los jugadores y rebotes de pelota.

    Args:
        players_by_label: dict ``{etiqueta: (real_x, real_y)}`` por jugador.
        bounces_x, bounces_y: coordenadas (m) de los rebotes de pelota.
        out_path: ruta del PNG de salida.
    """
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.set_facecolor(config.PLAYER_HEATMAP_FACECOLOR)
    _draw_court_2d(ax)
    # Las etiquetas son nombres reales de jugador; se asigna un color por orden
    # (uno por mitad de pista, no se solapan) en lugar de mapear por top/bottom.
    palette = ["deepskyblue", "orange", "magenta", "lime"]
    for i, (label, (real_x, real_y)) in enumerate(players_by_label.items()):
        ax.scatter(real_x, real_y, s=14, alpha=0.7,
                   color=palette[i % len(palette)], label=label,
                   edgecolors="black", linewidths=0.3)
    if len(bounces_x):
        ax.scatter(bounces_x, bounces_y, s=70, marker="X", color="red",
                   edgecolors="white", linewidths=1.0, label=f"rebotes (n={len(bounces_x)})")
    ax.legend(loc="upper right")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Vista combinada — jugadores y rebotes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Vista combinada guardada en %s", out_path)
