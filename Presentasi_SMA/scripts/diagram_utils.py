"""Shared vector-drawing utilities for Mobile-FOG architecture diagrams.

Inspired by clean hand-drawn / Excalidraw-style technical tensor diagrams:
rounded pastel containers, 3D stacked tensor cubes, dashed projection cones,
orange monospace layer labels, shape annotations, and SMA-friendly analogy pills.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

# Configure crisp system fonts (verified available on Linux)
FONT_SANS = "Inter"
FONT_MONO = "JetBrains Mono"

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [FONT_SANS, "Lato", "DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["axes.unicode_minus"] = False

# Palette matching reference style + clean semantic accents
PALETTE = {
    "bg": "#FFFFFF",
    "frame_border": "#3B3E47",
    "text_dark": "#1E293B",
    "text_muted": "#555D6E",
    "code_orange": "#C95B2B",
    "mint_fill": "#C8E8BA",
    "mint_edge": "#5A8F49",
    "blue_fill": "#E8F1FD",
    "blue_edge": "#6B8CC7",
    "peach_fill": "#FDECDC",
    "peach_edge": "#D4935E",
    "gray_fill": "#E5E8EE",
    "gray_edge": "#7C8494",
    "yellow_fill": "#FEF3C7",
    "yellow_edge": "#D97706",
    "purple_fill": "#EFE9FD",
    "purple_edge": "#7C5CBF",
    "red_fill": "#FEE2E2",
    "red_edge": "#DC2626",
    "charcoal_fill": "#434750",
    "charcoal_edge": "#272A30",
    "white_fill": "#FFFFFF",
    "white_edge": "#4A4E59",
    "arrow": "#5A6070",
    "sma_bg": "#FFFBEB",
    "sma_edge": "#F59E0B",
    "sma_text": "#92400E",
}

CUBE_PALETTES = {
    "blue": {
        "front": ["#9EC1F5", "#6B9CEB", "#4B7ED8", "#3566C4"],
        "back": ["#769FE0", "#4F7ED0", "#3765BA", "#244FA3"],
        "edge": "#37568F",
    },
    "yellow": {
        "front": ["#FDE047", "#FACC15"],
        "back": ["#EAB308", "#CA8A04"],
        "edge": "#92610A",
    },
    "mint": {
        "front": ["#A7E3B0", "#6ECF82"],
        "back": ["#78C486", "#4BA85E"],
        "edge": "#2F733E",
    },
    "peach": {
        "front": ["#FDBA74", "#FB923C"],
        "back": ["#EA8C3A", "#DC6814"],
        "edge": "#9A4A10",
    },
    "purple": {
        "front": ["#D8B4FE", "#C084FC"],
        "back": ["#A855F7", "#9333EA"],
        "edge": "#6B21A8",
    },
}


def new_canvas(width: float, height: float, dpi: int = 260):
    """Create a clean white canvas with exact metric coordinate bounds."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.015)
    return fig, ax


def draw_header(
    ax,
    width: float,
    top_y: float,
    title: str,
    subtitle: str | None = None,
    badge: str | None = None,
):
    """Draw a clean top header title, subtitle, and optional badge pill."""
    ax.text(
        0.35,
        top_y,
        title,
        fontsize=15.5,
        fontweight="bold",
        color=PALETTE["text_dark"],
        ha="left",
        va="top",
        fontfamily=FONT_SANS,
        zorder=20,
    )
    if subtitle:
        ax.text(
            0.35,
            top_y - 0.36,
            subtitle,
            fontsize=10.2,
            color=PALETTE["text_muted"],
            ha="left",
            va="top",
            fontfamily=FONT_SANS,
            zorder=20,
        )
    if badge:
        bw = max(2.2, len(badge) * 0.115 + 0.4)
        bx = width - 0.35 - bw
        by = top_y - 0.44
        draw_box(
            ax,
            bx,
            by,
            bw,
            0.42,
            badge,
            facecolor=PALETTE["blue_fill"],
            edgecolor=PALETTE["blue_edge"],
            textcolor="#1E3A8A",
            fontsize=9.2,
            bold=True,
            rounding=0.14,
            lw=1.3,
            zorder=20,
        )


def draw_outer_frame(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str = "",
    subtitle: str | None = None,
    facecolor: str = "#FFFFFF",
    edgecolor: str = PALETTE["frame_border"],
    lw: float = 2.0,
    ls: str = "-",
    rounding: float = 0.28,
    title_size: float = 12.0,
    zorder: int = 2,
):
    """Draw a rounded container frame with title in top-left corner."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if title:
        ax.text(
            x + 0.24,
            y + h - 0.20,
            title,
            fontsize=title_size,
            fontweight="bold",
            color=PALETTE["text_dark"],
            ha="left",
            va="top",
            fontfamily=FONT_SANS,
            zorder=zorder + 2,
        )
    if subtitle:
        ax.text(
            x + 0.24,
            y + h - 0.48,
            subtitle,
            fontsize=8.8,
            color=PALETTE["text_muted"],
            ha="left",
            va="top",
            fontfamily=FONT_SANS,
            zorder=zorder + 2,
        )
    return patch


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str = "#FFFFFF",
    edgecolor: str = PALETTE["white_edge"],
    linestyle: str = "-",
    textcolor: str = PALETTE["text_dark"],
    fontsize: float = 10.0,
    bold: bool = False,
    subtext: str | None = None,
    subsize: float = 8.2,
    subcolor: str | None = None,
    rounding: float = 0.14,
    lw: float = 1.6,
    fontfamily: str = FONT_SANS,
    zorder: int = 5,
):
    """Draw a rounded box with main centered text and optional subtext."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)

    cx = x + w / 2.0
    cy = y + h / 2.0
    weight = "bold" if bold else "medium"
    if subtext:
        ax.text(
            cx,
            cy + h * 0.14,
            text,
            fontsize=fontsize,
            fontweight=weight,
            color=textcolor,
            ha="center",
            va="center",
            fontfamily=fontfamily,
            zorder=zorder + 2,
        )
        sc = subcolor or (
            "#E2E8F0" if facecolor == PALETTE["charcoal_fill"] else PALETTE["text_muted"]
        )
        ax.text(
            cx,
            cy - h * 0.19,
            subtext,
            fontsize=subsize,
            color=sc,
            ha="center",
            va="center",
            fontfamily=FONT_SANS,
            zorder=zorder + 2,
        )
    else:
        ax.text(
            cx,
            cy,
            text,
            fontsize=fontsize,
            fontweight=weight,
            color=textcolor,
            ha="center",
            va="center",
            fontfamily=fontfamily,
            zorder=zorder + 2,
        )
    return patch


def draw_tensor_bar(
    ax,
    cx: float,
    cy: float,
    n_blocks: int = 4,
    cell: float = 0.25,
    depth: int = 2,
    palette: str = "blue",
    label_code: str | None = None,
    label_shape: str | None = None,
    shape_pos: str = "right",
    code_pos: str = "left",
    zorder: int = 8,
):
    """Draw 3D stacked rounded tensor cubes matching the reference VisionMamba diagram.

    Returns (x_left, x_right, y_bottom, y_top) bounding box of the main front row.
    """
    pal = CUBE_PALETTES.get(palette, CUBE_PALETTES["blue"])
    fronts = pal["front"]
    backs = pal["back"]
    edge = pal["edge"]

    gap = cell * 0.06
    total_w = n_blocks * cell + max(0, n_blocks - 1) * gap
    x0 = cx - total_w / 2.0
    y0 = cy - cell / 2.0

    # Draw depth layers from back to front (up-right isometric step)
    dx_step = cell * 0.34
    dy_step = cell * 0.34

    for layer in range(depth - 1, -1, -1):
        ox = layer * dx_step
        oy = layer * dy_step
        colors = backs if layer > 0 else fronts
        for i in range(n_blocks):
            c = colors[(i + layer) % len(colors)]
            bx = x0 + i * (cell + gap) + ox
            by = y0 + oy
            cube = FancyBboxPatch(
                (bx, by),
                cell,
                cell,
                boxstyle=f"round,pad=0,rounding_size={cell * 0.18}",
                facecolor=c,
                edgecolor=edge,
                linewidth=1.0,
                zorder=zorder + (depth - layer),
            )
            ax.add_patch(cube)

    top_y = y0 + cell + (depth - 1) * dy_step
    right_x = x0 + total_w + (depth - 1) * dx_step

    if label_shape:
        if shape_pos == "right":
            ax.text(
                right_x + 0.14,
                cy + (depth - 1) * dy_step * 0.4,
                label_shape,
                fontsize=8.8,
                color=PALETTE["text_muted"],
                ha="left",
                va="center",
                fontfamily=FONT_MONO,
                zorder=zorder + 5,
            )
        elif shape_pos == "bottom":
            ax.text(
                cx,
                y0 - 0.14,
                label_shape,
                fontsize=8.6,
                color=PALETTE["text_muted"],
                ha="center",
                va="top",
                fontfamily=FONT_MONO,
                zorder=zorder + 5,
            )
        elif shape_pos == "top":
            ax.text(
                cx,
                top_y + 0.12,
                label_shape,
                fontsize=8.6,
                color=PALETTE["text_muted"],
                ha="center",
                va="bottom",
                fontfamily=FONT_MONO,
                zorder=zorder + 5,
            )

    if label_code:
        if code_pos == "left":
            ax.text(
                x0 - 0.14,
                cy,
                label_code,
                fontsize=8.8,
                color=PALETTE["code_orange"],
                fontweight="bold",
                ha="right",
                va="center",
                fontfamily=FONT_MONO,
                zorder=zorder + 5,
            )
        elif code_pos == "top":
            ax.text(
                cx + (depth - 1) * dx_step * 0.4,
                top_y + 0.12,
                label_code,
                fontsize=8.8,
                color=PALETTE["code_orange"],
                fontweight="bold",
                ha="center",
                va="bottom",
                fontfamily=FONT_MONO,
                zorder=zorder + 5,
            )
        elif code_pos == "bottom":
            ax.text(
                cx,
                y0 - 0.14,
                label_code,
                fontsize=8.8,
                color=PALETTE["code_orange"],
                fontweight="bold",
                ha="center",
                va="top",
                fontfamily=FONT_MONO,
                zorder=zorder + 5,
            )

    return (x0, x0 + total_w, y0, top_y)


def draw_projection_cone(
    ax,
    x1_l: float,
    x1_r: float,
    y1: float,
    x2_l: float,
    x2_r: float,
    y2: float,
    text: str = "Projection",
    code_label: str | None = None,
    zorder: int = 4,
):
    """Draw dashed trapezoid connector lines between two tensor bars."""
    ax.plot(
        [x1_l, x2_l],
        [y1, y2],
        linestyle=(0, (3, 3)),
        color="#8B93A3",
        linewidth=1.25,
        zorder=zorder,
    )
    ax.plot(
        [x1_r, x2_r],
        [y1, y2],
        linestyle=(0, (3, 3)),
        color="#8B93A3",
        linewidth=1.25,
        zorder=zorder,
    )
    mid_x = 0.25 * (x1_l + x1_r + x2_l + x2_r)
    mid_y = 0.5 * (y1 + y2)
    ax.text(
        mid_x,
        mid_y,
        text,
        fontsize=8.3,
        color="#6E7787",
        ha="center",
        va="center",
        fontfamily=FONT_SANS,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="#FFFFFF", edgecolor="none", alpha=0.85),
        zorder=zorder + 2,
    )
    if code_label:
        left_x = min(x1_l, x2_l) - 0.16
        ax.text(
            left_x,
            mid_y,
            code_label,
            fontsize=8.8,
            fontweight="bold",
            color=PALETTE["code_orange"],
            ha="right",
            va="center",
            fontfamily=FONT_MONO,
            zorder=zorder + 2,
        )


def draw_op_circle(
    ax,
    cx: float,
    cy: float,
    symbol: str = "+",
    r: float = 0.19,
    facecolor: str = "#FFFFFF",
    edgecolor: str = "#6B7280",
    zorder: int = 10,
):
    """Draw crisp vector ⊕ or ⊗ circle operator without font glyph issues."""
    circ = Circle(
        (cx, cy),
        radius=r,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
        zorder=zorder,
    )
    ax.add_patch(circ)
    s = r * 0.58
    if symbol == "+":
        ax.plot([cx - s, cx + s], [cy, cy], color=edgecolor, lw=1.6, zorder=zorder + 1)
        ax.plot([cx, cx], [cy - s, cy + s], color=edgecolor, lw=1.6, zorder=zorder + 1)
    elif symbol == "x":
        d = s * 0.78
        ax.plot([cx - d, cx + d], [cy - d, cy + d], color=edgecolor, lw=1.6, zorder=zorder + 1)
        ax.plot([cx - d, cx + d], [cy + d, cy - d], color=edgecolor, lw=1.6, zorder=zorder + 1)
    return circ


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = PALETTE["arrow"],
    lw: float = 1.6,
    ls: str = "-",
    rad: float = 0.0,
    mutation_scale: float = 13,
    zorder: int = 6,
):
    """Draw a clean straight or curved arrow."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0",
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def draw_poly_arrow(
    ax,
    pts: list[tuple[float, float]],
    color: str = PALETTE["arrow"],
    lw: float = 1.6,
    ls: str = "-",
    mutation_scale: float = 13,
    zorder: int = 6,
):
    """Draw a multi-segment orthogonal path ending with an arrowhead."""
    if len(pts) < 2:
        return
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=zorder)
    draw_arrow(
        ax,
        pts[-2],
        pts[-1],
        color=color,
        lw=lw,
        ls=ls,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )


def draw_sma_pill(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fontsize: float = 9.4,
    zorder: int = 12,
):
    """Draw a friendly warm yellow analogy pill for high-school students."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0,rounding_size=0.16",
        facecolor=PALETTE["sma_bg"],
        edgecolor=PALETTE["sma_edge"],
        linewidth=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        fontsize=fontsize,
        fontweight="medium",
        color=PALETTE["sma_text"],
        ha="center",
        va="center",
        fontfamily=FONT_SANS,
        zorder=zorder + 2,
    )
    return patch


def draw_code_tag(
    ax,
    x: float,
    y: float,
    text: str,
    ha: str = "center",
    va: str = "center",
    fontsize: float = 8.8,
    color: str = PALETTE["code_orange"],
    zorder: int = 15,
):
    """Draw a monospace variable/layer tag."""
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha=ha,
        va=va,
        fontfamily=FONT_MONO,
        zorder=zorder,
    )


def draw_shape_tag(
    ax,
    x: float,
    y: float,
    text: str,
    ha: str = "center",
    va: str = "center",
    fontsize: float = 8.6,
    color: str = PALETTE["text_muted"],
    zorder: int = 15,
):
    """Draw a monospace tensor shape tag like [60, 64]."""
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        fontfamily=FONT_MONO,
        zorder=zorder,
    )
