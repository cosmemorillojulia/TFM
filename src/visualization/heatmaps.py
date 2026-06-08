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


def _kde_layer(ax, real_x, real_y, cmap):
    """Pinta una capa de densidad KDE sobre el eje. Devuelve la imagen o None.

    Devuelve None si no hay datos suficientes para estimar la densidad (menos de
    2 puntos o varianza nula), en cuyo caso solo se dibuja la pista.
    """
    pts = np.vstack([real_x, real_y])
    if pts.shape[1] < 2 or np.allclose(pts.std(axis=1), 0):
        return None
    xs, ys, grid_x, grid_y = _make_grid()
    kde = gaussian_kde(pts, bw_method=config.KDE_BANDWIDTH)
    z = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
    return ax.imshow(
        z, origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap=cmap, alpha=0.85, aspect="equal",
    )


def export_player_heatmap(real_x, real_y, title, out_path):
    """Genera y guarda el heatmap KDE de ocupacion de un jugador."""
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
    ax.set_facecolor(config.COURT_FACECOLOR)
    im = _kde_layer(ax, real_x, real_y, config.PLAYER_CMAP)
    _draw_court_2d(ax)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("densidad")
    else:
        ax.text(0.5, 0.5, "datos insuficientes", transform=ax.transAxes,
                ha="center", va="center", color="white")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.DPI_EXPORT, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap de jugador guardado en %s", out_path)


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
    ax.set_facecolor(config.COURT_FACECOLOR)
    _draw_court_2d(ax)
    colors = {"player_top": "deepskyblue", "player_bottom": "orange"}
    for label, (real_x, real_y) in players_by_label.items():
        ax.scatter(real_x, real_y, s=14, alpha=0.7,
                   color=colors.get(label, "magenta"), label=label,
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
