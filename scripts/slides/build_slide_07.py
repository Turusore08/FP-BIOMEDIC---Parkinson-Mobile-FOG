"""Generate Slide 7 design (Dampak Penghapusan Data Noise) in preview & Canva-importable formats.

Outputs written to outputs/slides/:
  - chart_progresi_3tahap.png               (300 DPI grouped bar chart, 7.1 x 3.55 in)
  - slide_07_dampak_penghapusan_noise.pptx  (16:9 editable PPTX for Canva import)
  - slide_07_dampak_penghapusan_noise.png   (200 DPI 1:1 visual preview)
  - slide_07_dampak_penghapusan_noise.pdf   (16:9 vector/high-res PDF)
"""

from __future__ import annotations

import glob
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

from mobilefog.paths import SLIDES_OUTPUT_DIR

OUT_DIR = SLIDES_OUTPUT_DIR
CHART_PATH = OUT_DIR / "chart_progresi_3tahap.png"
PPTX_PATH = OUT_DIR / "slide_07_dampak_penghapusan_noise.pptx"
PNG_PATH = OUT_DIR / "slide_07_dampak_penghapusan_noise.png"
PDF_PATH = OUT_DIR / "slide_07_dampak_penghapusan_noise.pdf"

SLIDE_W = 13.333
SLIDE_H = 7.500


# ---------------------------------------------------------------------------
# Color tokens (matched to Presentasi_SMA/scripts/build_pptx.py)
# ---------------------------------------------------------------------------
HEX = {
    "bg": "#F8FAFC",
    "white": "#FFFFFF",
    "border": "#CBD5E1",
    "row_alt": "#F8FAFC",
    "navy": "#0F172A",
    "slate_dark": "#1E293B",
    "slate_mid": "#475569",
    "muted": "#64748B",
    "raw": "#94A3B8",
    "blue": "#2563EB",
    "blue_bg": "#EFF6FF",
    "blue_dark": "#1E3A8A",
    "green": "#16A34A",
    "green_bg": "#F0FDF4",
    "green_dark": "#14532D",
    "purple": "#7C3AED",
    "purple_bg": "#F5F3FF",
    "purple_dark": "#5B21B6",
    "amber": "#D97706",
    "amber_bg": "#FFFBEB",
    "amber_dark": "#92400E",
    "red": "#DC2626",
    "red_bg": "#FEF2F2",
    "grid": "#E2E8F0",
}


def hex_to_rgb(h: str) -> RGBColor:
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------
def configure_fonts() -> str:
    """Register Lato & JetBrains Mono fonts if available and return primary family."""
    for folder in [
        "/usr/share/fonts/truetype/lato",
        "/usr/share/fonts/truetype/jetbrains-mono",
        "/usr/share/fonts/truetype/dejavu",
    ]:
        for ttf in sorted(glob.glob(f"{folder}/*.ttf")):
            try:
                font_manager.fontManager.addfont(ttf)
            except Exception:
                pass

    available = {f.name for f in font_manager.fontManager.ttflist}
    primary = "Lato" if "Lato" in available else "DejaVu Sans"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [primary, "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    return primary


# ---------------------------------------------------------------------------
# Slide content data (verified against results_summary.csv & loso_oof_summary)
# ---------------------------------------------------------------------------
MODELS_CHART = ["Light_ST-GCN", "LightGBM+FFT", "Pure_Bi-Mamba", "Hybrid\nGCN+Mamba"]
VALS_RAW = [0.5756, 0.6294, 0.6631, 0.6065]
VALS_BBOX = [0.8366, 0.8365, 0.6802, 0.6699]
VALS_CLEAN = [0.9036, 0.8861, 0.8292, 0.6642]
DELTAS_CB = [c - b for c, b in zip(VALS_CLEAN, VALS_BBOX)]

KPI_CARDS = [
    {
        "x": 0.42,
        "fill": HEX["green_bg"],
        "border": HEX["green"],
        "lw": 1.6,
        "label": "BEST TEST PR-AUC",
        "label_color": HEX["green_dark"],
        "value": "0.9036",
        "value_color": HEX["green"],
        "sub": "+0.0670 vs BBox  •  Light_ST-GCN (B_15hz)",
    },
    {
        "x": 3.60,
        "fill": HEX["blue_bg"],
        "border": HEX["blue"],
        "lw": 1.4,
        "label": "BEST TEST F1-FoG (PP)",
        "label_color": HEX["blue_dark"],
        "value": "0.8293",
        "value_color": HEX["blue"],
        "sub": "+0.0764 vs BBox  •  Menang 16/16 Run",
    },
    {
        "x": 6.78,
        "fill": HEX["white"],
        "border": HEX["green"],
        "lw": 1.4,
        "label": "BEST BALANCED ACCURACY",
        "label_color": HEX["green_dark"],
        "value": "90.54%",
        "value_color": HEX["navy"],
        "sub": "Sens 87.2% | Spec 93.9%  •  Light_ST-GCN",
    },
    {
        "x": 9.96,
        "fill": HEX["purple_bg"],
        "border": HEX["purple"],
        "lw": 1.4,
        "label": "EFISIENSI WAKTU LATIH",
        "label_color": HEX["purple_dark"],
        "value": "4.2× Lebih Cepat",
        "value_color": HEX["purple"],
        "sub": "418.5 dtk → 100.0 dtk (Mean 16 Run)",
    },
]

LOSO_ROWS = [
    ("Light_ST-GCN", "0.7284", "0.9170", "0.6711", "79.5%", HEX["green_bg"], True),
    ("LightGBM+FFT", "0.5357*", "0.8406", "0.7515", "82.4%", HEX["blue_bg"], True),
    ("Pure_Bi-Mamba", "0.6706", "0.8072", "0.6006", "73.6%", HEX["white"], False),
    ("Hybrid_GCN+Mamba", "0.6021", "0.7817", "0.5502", "71.7%", HEX["row_alt"], False),
]

INSIGHTS = [
    (
        "Pure_Bi-Mamba melonjak +0.139 mean PR-AUC:",
        "model sequence (211k params) paling sensitif terhadap outlier tracking/label.",
        HEX["purple_dark"],
    ),
    (
        "Model ringan dominan & konsisten:",
        "Light_ST-GCN (38.8k) & LightGBM+FFT (1.3 dtk CPU) unggul di Fixed Split & LOSO; Hybrid (216k) overfit.",
        HEX["blue_dark"],
    ),
    (
        "Skenario B_15hz (Hampel + Butterworth 9 Hz) terbaik:",
        "meredam jitter antar-frame dan menonjolkan pita getar FoG 3–7.5 Hz.",
        HEX["green_dark"],
    ),
]


# ---------------------------------------------------------------------------
# 1. Generate high-res grouped bar chart (7.1 x 3.55 in, 300 DPI)
# ---------------------------------------------------------------------------
def build_chart(font_family: str) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.55), dpi=300)
    fig.patch.set_facecolor(HEX["white"])
    ax.set_facecolor(HEX["white"])

    x = np.arange(len(MODELS_CHART))
    w = 0.245
    gap = 0.02

    bars_raw = ax.bar(
        x - (w + gap),
        VALS_RAW,
        width=w,
        color=HEX["raw"],
        edgecolor="#64748B",
        linewidth=0.7,
        label="Baseline (35 Subj)",
        zorder=3,
    )
    bars_bbox = ax.bar(
        x,
        VALS_BBOX,
        width=w,
        color=HEX["blue"],
        edgecolor="#1E40AF",
        linewidth=0.7,
        label="+ BBox (35 Subj)",
        zorder=3,
    )
    bars_clean = ax.bar(
        x + (w + gap),
        VALS_CLEAN,
        width=w,
        color=HEX["green"],
        edgecolor="#15803D",
        linewidth=0.8,
        label="+ Without Noise (32 Subj)",
        zorder=3,
    )

    # Reference threshold line y=0.90
    ax.axhline(0.90, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        len(MODELS_CHART) - 0.38,
        0.912,
        "0.90",
        fontsize=8.0,
        color=HEX["muted"],
        fontweight="bold",
        ha="right",
        va="bottom",
        fontfamily=font_family,
    )

    # Value labels above each bar (white halo prevents dashed 0.90 line from striking through)
    halo = dict(boxstyle="square,pad=0.04", facecolor=HEX["white"], edgecolor="none", alpha=0.90)
    for rect in bars_raw:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + 0.012,
            f".{int(round(h * 1000)):03d}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=HEX["slate_mid"],
            fontfamily=font_family,
            bbox=halo,
            zorder=4,
        )

    for rect in bars_bbox:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + 0.012,
            f".{int(round(h * 1000)):03d}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=HEX["blue_dark"],
            fontfamily=font_family,
            bbox=halo,
            zorder=4,
        )

    for idx, rect in enumerate(bars_clean):
        h = rect.get_height()
        d = DELTAS_CB[idx]
        cx = rect.get_x() + rect.get_width() / 2.0
        # Value label on top of green bar
        ax.text(
            cx,
            h + 0.012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=HEX["green_dark"],
            fontfamily=font_family,
            bbox=halo,
            zorder=4,
        )
        # Delta pill badge above value label
        if idx == 2:  # Pure_Bi-Mamba huge jump highlight
            badge_bg, badge_edge, badge_fg = HEX["purple_bg"], HEX["purple"], HEX["purple_dark"]
        elif d >= 0:
            badge_bg, badge_edge, badge_fg = HEX["green_bg"], HEX["green"], HEX["green_dark"]
        else:
            badge_bg, badge_edge, badge_fg = HEX["red_bg"], HEX["red"], HEX["red"]

        sign = "+" if d >= 0 else ""
        ax.text(
            cx,
            h + 0.088,
            f"{sign}{d:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color=badge_fg,
            fontfamily=font_family,
            bbox=dict(
                boxstyle="round,pad=0.22,rounding_size=0.2",
                facecolor=badge_bg,
                edgecolor=badge_edge,
                linewidth=0.9,
            ),
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS_CHART, fontsize=9.5, fontweight="bold", color=HEX["navy"], fontfamily=font_family)
    ax.set_ylim(0.0, 1.15)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.5, color=HEX["slate_mid"], fontfamily=font_family)
    ax.set_ylabel("Best Test PR-AUC", fontsize=9.2, fontweight="bold", color=HEX["slate_dark"], fontfamily=font_family)

    ax.yaxis.grid(True, linestyle=":", color=HEX["grid"], linewidth=0.9, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(HEX["border"])
    ax.spines["bottom"].set_color(HEX["border"])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", color=HEX["border"])

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
        facecolor=HEX["bg"],
        edgecolor=HEX["border"],
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.4,
    )

    plt.tight_layout(pad=0.35)
    fig.savefig(CHART_PATH, dpi=300, facecolor=HEX["white"])
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Build 100% editable 16:9 PPTX for Canva import
# ---------------------------------------------------------------------------
def set_cell_margins(cell, left=0.06, right=0.06, top=0.03, bottom=0.03):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.set("marL", str(int(Inches(left))))
    tcPr.set("marR", str(int(Inches(right))))
    tcPr.set("marT", str(int(Inches(top))))
    tcPr.set("marB", str(int(Inches(bottom))))


def set_cell_border(cell, color_hex: str = "#CBD5E1", width_pt: float = 0.6):
    tcPr = cell._tc.get_or_add_tcPr()
    w = str(int(width_pt * 12700))
    rgb = color_hex.lstrip("#")
    for side in ("lnL", "lnR", "lnT", "lnB"):
        for old in tcPr.findall(qn(f"a:{side}")):
            tcPr.remove(old)
        ln = OxmlElement(f"a:{side}")
        ln.set("w", w)
        ln.set("cap", "flat")
        ln.set("cmpd", "sng")
        ln.set("algn", "ctr")
        solidFill = OxmlElement("a:solidFill")
        srgbClr = OxmlElement("a:srgbClr")
        srgbClr.set("val", rgb)
        solidFill.append(srgbClr)
        ln.append(solidFill)
        tcPr.append(ln)


def add_pptx_rect(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    fill_hex: str,
    line_hex: str | None = None,
    line_pt: float = 1.2,
    rounded: bool = True,
    radius_adj: float = 0.05,
):
    stype = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(stype, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = hex_to_rgb(fill_hex)
    if line_hex:
        shp.line.color.rgb = hex_to_rgb(line_hex)
        shp.line.width = Pt(line_pt)
    else:
        shp.line.fill.background()
    if rounded and shp.adjustments:
        shp.adjustments[0] = radius_adj
    return shp


def add_pptx_text(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    runs: list[tuple[str, float, bool, str]],
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    font_name: str = "Lato",
    mx: float = 0.04,
    my: float = 0.02,
):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(mx)
    tf.margin_right = Inches(mx)
    tf.margin_top = Inches(my)
    tf.margin_bottom = Inches(my)
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(0)
    for txt, fsize, is_bold, col_hex in runs:
        if txt == "\n":
            p = tf.add_paragraph()
            p.alignment = align
            p.space_after = Pt(1.5)
            continue
        r = p.add_run()
        r.text = txt
        r.font.name = font_name
        r.font.size = Pt(fsize)
        r.font.bold = is_bold
        r.font.color.rgb = hex_to_rgb(col_hex)
    return tb


def build_pptx(font_family: str) -> None:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Background
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = hex_to_rgb(HEX["bg"])

    # 1. Top dual accent bar
    add_pptx_rect(slide, 0.0, 0.0, SLIDE_W / 2.0, 0.08, HEX["blue"], rounded=False)
    add_pptx_rect(slide, SLIDE_W / 2.0, 0.0, SLIDE_W / 2.0, 0.08, HEX["green"], rounded=False)

    # 2. Header
    add_pptx_rect(slide, 0.42, 0.18, 2.95, 0.28, HEX["green_bg"], HEX["green"], line_pt=1.0, radius_adj=0.35)
    add_pptx_text(
        slide, 0.42, 0.18, 2.95, 0.28,
        [("EVALUASI DATA CLEANING  •  SLIDE 7", 8.5, True, HEX["green_dark"])],
        align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.02, my=0.01,
    )
    add_pptx_text(
        slide, 0.42, 0.48, 9.25, 0.40,
        [("Dampak Penghapusan Data Noise terhadap Deteksi FoG", 21.5, True, HEX["navy"])],
        valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.0, my=0.0,
    )
    add_pptx_text(
        slide, 0.42, 0.87, 9.30, 0.26,
        [("Pembuangan 3 subjek bermasalah (PDFE15_1 Train · PDFE27_1–3 Val · PDFE24_1 Test) di atas PatientLock-2Pass BBox", 9.8, False, HEX["slate_mid"])],
        valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.0, my=0.0,
    )

    # Right summary pill
    add_pptx_rect(slide, 9.85, 0.22, 3.06, 0.84, HEX["amber_bg"], HEX["amber"], line_pt=1.3, radius_adj=0.14)
    add_pptx_text(
        slide, 9.95, 0.27, 2.86, 0.74,
        [
            ("35 → 32 Subjek Bersih (-5 Trial)", 10.5, True, HEX["amber_dark"]),
            ("\n", 10.5, False, HEX["amber_dark"]),
            ("71 → 66 Video  (22 Train / 4 Val / 6 Test)", 9.0, False, HEX["slate_mid"]),
        ],
        align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.04, my=0.02,
    )

    # 3. Row 1: 4 KPI Cards
    for kpi in KPI_CARDS:
        kx = kpi["x"]
        add_pptx_rect(slide, kx, 1.24, 2.95, 1.06, kpi["fill"], kpi["border"], line_pt=kpi["lw"], radius_adj=0.10)
        add_pptx_text(
            slide, kx + 0.12, 1.31, 2.71, 0.22,
            [(kpi["label"], 8.2, True, kpi["label_color"])],
            font_name=font_family, mx=0.0, my=0.0,
        )
        add_pptx_text(
            slide, kx + 0.12, 1.54, 2.71, 0.44,
            [(kpi["value"], 22.0, True, kpi["value_color"])],
            valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.0, my=0.0,
        )
        add_pptx_text(
            slide, kx + 0.12, 2.00, 2.71, 0.24,
            [(kpi["sub"], 8.4, False, HEX["slate_dark"])],
            valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.0, my=0.0,
        )

    # 4. Row 2 Left: Chart Card
    add_pptx_rect(slide, 0.42, 2.44, 7.10, 3.80, HEX["white"], HEX["border"], line_pt=1.2, radius_adj=0.04)
    add_pptx_text(
        slide, 0.56, 2.51, 6.82, 0.25,
        [("Progresi Best Test PR-AUC per Model (Baseline → BBox → Without Noise)", 11.2, True, HEX["navy"])],
        font_name=font_family, mx=0.0, my=0.0,
    )
    add_pptx_text(
        slide, 0.56, 2.75, 6.82, 0.20,
        [("Mean 16-Run PR-AUC: 0.538 → 0.707 → 0.766   |   Mean ROC-AUC: 0.753 → 0.844 → 0.877", 8.4, False, HEX["muted"])],
        font_name=font_family, mx=0.0, my=0.0,
    )
    # Fit 2:1 chart image inside (0.56, 2.96, 6.82, 3.20) -> 6.40 x 3.20 centered at x=0.77
    with Image.open(CHART_PATH) as im:
        ratio = im.size[0] / im.size[1]
    box_w, box_h = 6.82, 3.20
    if ratio >= box_w / box_h:
        cw, ch = box_w, box_w / ratio
    else:
        ch, cw = box_h, box_h * ratio
    cx = 0.56 + (box_w - cw) / 2.0
    cy = 2.96 + (box_h - ch) / 2.0
    slide.shapes.add_picture(str(CHART_PATH), Inches(cx), Inches(cy), width=Inches(cw), height=Inches(ch))

    # 5. Row 2 Right Top: LOSO Table Card
    add_pptx_rect(slide, 7.70, 2.44, 5.21, 2.04, HEX["white"], HEX["border"], line_pt=1.2, radius_adj=0.05)
    add_pptx_text(
        slide, 7.84, 2.51, 4.93, 0.24,
        [("Validasi Objektif: LOSO 32-Fold (B_15hz, 100% Selesai)", 10.8, True, HEX["navy"])],
        font_name=font_family, mx=0.0, my=0.0,
    )

    tbl_shape = slide.shapes.add_table(5, 5, Inches(7.84), Inches(2.80), Inches(4.93), Inches(1.28))
    tbl = tbl_shape.table
    col_widths = [1.59, 0.82, 0.86, 0.80, 0.86]
    for ci, cw_in in enumerate(col_widths):
        tbl.columns[ci].width = Inches(cw_in)

    headers = ["Model", "OOF PR", "OOF ROC", "F1-PP", "Bal-Acc"]
    for ci, htxt in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = hex_to_rgb(HEX["navy"])
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_cell_margins(cell, 0.05, 0.05, 0.02, 0.02)
        set_cell_border(cell, HEX["navy"], 0.5)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
        r = p.add_run()
        r.text = htxt
        r.font.name = font_family
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = hex_to_rgb(HEX["white"])

    for ri, row in enumerate(LOSO_ROWS, start=1):
        mod, pr, roc, f1, ba, row_bg, highlight = row
        vals = [mod, pr, roc, f1, ba]
        for ci, val in enumerate(vals):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = hex_to_rgb(row_bg)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            set_cell_margins(cell, 0.05, 0.05, 0.02, 0.02)
            set_cell_border(cell, HEX["border"], 0.6)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            r.font.name = font_family
            r.font.size = Pt(8.4)
            is_best = (ri == 1 and ci in (0, 1, 2)) or (ri == 2 and ci in (0, 3, 4))
            r.font.bold = is_best
            if ri == 1 and ci in (1, 2):
                r.font.color.rgb = hex_to_rgb(HEX["green_dark"])
            elif ri == 2 and ci in (3, 4):
                r.font.color.rgb = hex_to_rgb(HEX["blue_dark"])
            else:
                r.font.color.rgb = hex_to_rgb(HEX["navy"] if ci == 0 else HEX["slate_dark"])

    add_pptx_text(
        slide, 7.84, 4.13, 4.93, 0.30,
        [("*Light_ST-GCN unggul kalibrasi lintas subjek (PR/ROC); LightGBM+FFT unggul keputusan biner (F1/Bal-Acc, Spec 97.2%).", 7.6, False, HEX["slate_mid"])],
        font_name=font_family, mx=0.0, my=0.0,
    )

    # 6. Row 2 Right Bottom: Key Insights Card
    add_pptx_rect(slide, 7.70, 4.60, 5.21, 1.64, HEX["white"], HEX["border"], line_pt=1.2, radius_adj=0.05)
    add_pptx_text(
        slide, 7.84, 4.67, 4.93, 0.24,
        [("Temuan Kunci Penghapusan Noise", 10.8, True, HEX["navy"])],
        font_name=font_family, mx=0.0, my=0.0,
    )
    bullet_runs = []
    for idx, (lead, body, lead_col) in enumerate(INSIGHTS):
        if idx > 0:
            bullet_runs.append(("\n", 8.5, False, HEX["slate_dark"]))
        bullet_runs.append((f"• {lead} ", 8.5, True, lead_col))
        bullet_runs.append((body, 8.5, False, HEX["slate_dark"]))
    add_pptx_text(
        slide, 7.84, 4.94, 4.93, 1.24,
        bullet_runs,
        font_name=font_family, mx=0.0, my=0.0,
    )

    # 7. Bottom Transparency Callout
    add_pptx_rect(slide, 0.42, 6.36, 12.49, 0.58, HEX["amber_bg"], HEX["amber"], line_pt=1.3, radius_adj=0.14)
    add_pptx_text(
        slide, 0.56, 6.40, 12.21, 0.50,
        [
            ("Catatan Transparansi Metodologis: ", 8.6, True, HEX["amber_dark"]),
            (
                "Karena PDFE24_1 berada di Test Set (7 → 6 subjek) dan PDFE27 di Val Set (5 → 4 subjek), sebagian kenaikan Fixed Split "
                "berasal dari penghapusan outlier evaluasi. Validasi penuh LOSO 32-Fold membuktikan generalisasi tetap kuat (ROC-AUC 0.917, Bal-Acc 82.4%).",
                8.6,
                False,
                HEX["slate_dark"],
            ),
        ],
        valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.0, my=0.0,
    )

    # 8. Footer
    add_pptx_rect(slide, 0.42, 7.06, 12.49, 0.015, HEX["border"], rounded=False)
    add_pptx_text(
        slide, 0.42, 7.11, 8.50, 0.28,
        [("Mobile-FOG  •  Deteksi Freezing of Gait Pasien Parkinson via Video Smartphone  •  Kelompok 4 ITS", 8.8, False, HEX["muted"])],
        valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.0, my=0.0,
    )
    add_pptx_text(
        slide, 9.80, 7.11, 3.11, 0.28,
        [("Slide 7", 8.8, True, HEX["muted"])],
        align=PP_ALIGN.RIGHT, valign=MSO_ANCHOR.MIDDLE, font_name=font_family, mx=0.0, my=0.0,
    )

    prs.save(PPTX_PATH)


# ---------------------------------------------------------------------------
# 3. Render 1:1 full-slide preview PNG (200 DPI) & PDF using Matplotlib
# ---------------------------------------------------------------------------
def mpl_round_rect(
    ax,
    x: float,
    y_top: float,
    w: float,
    h: float,
    fill: str,
    edge: str | None = None,
    lw: float = 1.2,
    rounding: float = 0.10,
    zorder: int = 2,
):
    y_bot = SLIDE_H - y_top - h
    if rounding <= 0:
        patch = Rectangle(
            (x, y_bot),
            w,
            h,
            facecolor=fill,
            edgecolor=edge if edge else "none",
            linewidth=lw if edge else 0,
            zorder=zorder,
        )
    else:
        patch = FancyBboxPatch(
            (x, y_bot),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=fill,
            edgecolor=edge if edge else "none",
            linewidth=lw if edge else 0,
            zorder=zorder,
        )
    ax.add_patch(patch)


def wrap_text_lines(text: str, max_chars: int) -> list[str]:
    words = text.split(" ")
    lines, cur = [], ""
    for word in words:
        cand = f"{cur} {word}".strip() if cur else word
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def build_preview_and_pdf(font_family: str) -> None:
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=200)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor(HEX["bg"])
    ax.set_facecolor(HEX["bg"])
    ax.set_xlim(0.0, SLIDE_W)
    ax.set_ylim(0.0, SLIDE_H)
    ax.axis("off")

    # 1. Top accent bar
    mpl_round_rect(ax, 0.0, 0.0, SLIDE_W / 2.0, 0.08, HEX["blue"], rounding=0.0, zorder=10)
    mpl_round_rect(ax, SLIDE_W / 2.0, 0.0, SLIDE_W / 2.0, 0.08, HEX["green"], rounding=0.0, zorder=10)

    # 2. Header
    mpl_round_rect(ax, 0.42, 0.18, 2.95, 0.28, HEX["green_bg"], HEX["green"], lw=1.0, rounding=0.10, zorder=4)
    ax.text(
        0.42 + 2.95 / 2.0,
        SLIDE_H - (0.18 + 0.14),
        "EVALUASI DATA CLEANING  •  SLIDE 7",
        fontsize=8.5, fontweight="bold", color=HEX["green_dark"],
        ha="center", va="center", fontfamily=font_family, zorder=6,
    )
    ax.text(
        0.42,
        SLIDE_H - 0.68,
        "Dampak Penghapusan Data Noise terhadap Deteksi FoG",
        fontsize=21.5, fontweight="bold", color=HEX["navy"],
        ha="left", va="center", fontfamily=font_family, zorder=6,
    )
    ax.text(
        0.42,
        SLIDE_H - 1.00,
        "Pembuangan 3 subjek bermasalah (PDFE15_1 Train · PDFE27_1–3 Val · PDFE24_1 Test) di atas PatientLock-2Pass BBox",
        fontsize=9.8, color=HEX["slate_mid"],
        ha="left", va="center", fontfamily=font_family, zorder=6,
    )

    # Right summary pill
    mpl_round_rect(ax, 9.85, 0.22, 3.06, 0.84, HEX["amber_bg"], HEX["amber"], lw=1.3, rounding=0.12, zorder=4)
    ax.text(
        9.85 + 3.06 / 2.0,
        SLIDE_H - 0.49,
        "35 → 32 Subjek Bersih (-5 Trial)",
        fontsize=10.5, fontweight="bold", color=HEX["amber_dark"],
        ha="center", va="center", fontfamily=font_family, zorder=6,
    )
    ax.text(
        9.85 + 3.06 / 2.0,
        SLIDE_H - 0.81,
        "71 → 66 Video  (22 Train / 4 Val / 6 Test)",
        fontsize=9.0, color=HEX["slate_mid"],
        ha="center", va="center", fontfamily=font_family, zorder=6,
    )

    # 3. Row 1: 4 KPI Cards
    for kpi in KPI_CARDS:
        kx = kpi["x"]
        mpl_round_rect(ax, kx, 1.24, 2.95, 1.06, kpi["fill"], kpi["border"], lw=kpi["lw"], rounding=0.10, zorder=3)
        ax.text(
            kx + 0.14, SLIDE_H - 1.41, kpi["label"],
            fontsize=8.2, fontweight="bold", color=kpi["label_color"],
            ha="left", va="center", fontfamily=font_family, zorder=6,
        )
        ax.text(
            kx + 0.14, SLIDE_H - 1.76, kpi["value"],
            fontsize=22.0, fontweight="bold", color=kpi["value_color"],
            ha="left", va="center", fontfamily=font_family, zorder=6,
        )
        ax.text(
            kx + 0.14, SLIDE_H - 2.12, kpi["sub"],
            fontsize=8.4, color=HEX["slate_dark"],
            ha="left", va="center", fontfamily=font_family, zorder=6,
        )

    # 4. Row 2 Left: Chart Card
    mpl_round_rect(ax, 0.42, 2.44, 7.10, 3.80, HEX["white"], HEX["border"], lw=1.2, rounding=0.12, zorder=3)
    ax.text(
        0.56, SLIDE_H - 2.63,
        "Progresi Best Test PR-AUC per Model (Baseline → BBox → Without Noise)",
        fontsize=11.2, fontweight="bold", color=HEX["navy"],
        ha="left", va="center", fontfamily=font_family, zorder=6,
    )
    ax.text(
        0.56, SLIDE_H - 2.85,
        "Mean 16-Run PR-AUC: 0.538 → 0.707 → 0.766   |   Mean ROC-AUC: 0.753 → 0.844 → 0.877",
        fontsize=8.4, color=HEX["muted"],
        ha="left", va="center", fontfamily=font_family, zorder=6,
    )

    chart_img = np.asarray(Image.open(CHART_PATH))
    ratio = chart_img.shape[1] / chart_img.shape[0]
    box_w, box_h = 6.82, 3.20
    if ratio >= box_w / box_h:
        cw, ch = box_w, box_w / ratio
    else:
        ch, cw = box_h, box_h * ratio
    cx = 0.56 + (box_w - cw) / 2.0
    cy = 2.96 + (box_h - ch) / 2.0
    ax.imshow(
        chart_img,
        extent=(cx, cx + cw, SLIDE_H - cy - ch, SLIDE_H - cy),
        aspect="auto",
        zorder=5,
        interpolation="lanczos",
    )

    # 5. Row 2 Right Top: LOSO Table Card
    mpl_round_rect(ax, 7.70, 2.44, 5.21, 2.04, HEX["white"], HEX["border"], lw=1.2, rounding=0.10, zorder=3)
    ax.text(
        7.84, SLIDE_H - 2.62,
        "Validasi Objektif: LOSO 32-Fold (B_15hz, 100% Selesai)",
        fontsize=10.8, fontweight="bold", color=HEX["navy"],
        ha="left", va="center", fontfamily=font_family, zorder=6,
    )

    tx0, ty0 = 7.84, 2.80
    col_ws = [1.59, 0.82, 0.86, 0.80, 0.86]
    row_h = 0.256
    headers = ["Model", "OOF PR", "OOF ROC", "F1-PP", "Bal-Acc"]
    # Table header
    cx_cur = tx0
    for ci, (htxt, cw_i) in enumerate(zip(headers, col_ws)):
        mpl_round_rect(ax, cx_cur, ty0, cw_i, row_h, HEX["navy"], HEX["navy"], lw=0.6, rounding=0.0, zorder=5)
        ax.text(
            cx_cur + (0.06 if ci == 0 else cw_i / 2.0),
            SLIDE_H - (ty0 + row_h / 2.0),
            htxt,
            fontsize=8.4, fontweight="bold", color=HEX["white"],
            ha="left" if ci == 0 else "center", va="center",
            fontfamily=font_family, zorder=7,
        )
        cx_cur += cw_i

    for ri, (mod, pr, roc, f1, ba, row_bg, _) in enumerate(LOSO_ROWS, start=1):
        ry = ty0 + ri * row_h
        cx_cur = tx0
        vals = [mod, pr, roc, f1, ba]
        for ci, (val, cw_i) in enumerate(zip(vals, col_ws)):
            mpl_round_rect(ax, cx_cur, ry, cw_i, row_h, row_bg, HEX["border"], lw=0.6, rounding=0.0, zorder=5)
            is_best = (ri == 1 and ci in (0, 1, 2)) or (ri == 2 and ci in (0, 3, 4))
            if ri == 1 and ci in (1, 2):
                tcol = HEX["green_dark"]
            elif ri == 2 and ci in (3, 4):
                tcol = HEX["blue_dark"]
            else:
                tcol = HEX["navy"] if ci == 0 else HEX["slate_dark"]
            ax.text(
                cx_cur + (0.06 if ci == 0 else cw_i / 2.0),
                SLIDE_H - (ry + row_h / 2.0),
                val,
                fontsize=8.2, fontweight="bold" if is_best else "normal", color=tcol,
                ha="left" if ci == 0 else "center", va="center",
                fontfamily=font_family, zorder=7,
            )
            cx_cur += cw_i

    footnote_lines = wrap_text_lines(
        "*Light_ST-GCN unggul kalibrasi lintas subjek (PR/ROC); LightGBM+FFT unggul keputusan biner (F1/Bal-Acc, Spec 97.2%).",
        86,
    )
    ax.text(
        7.84, SLIDE_H - 4.18,
        "\n".join(footnote_lines),
        fontsize=7.5, color=HEX["slate_mid"], linespacing=1.25,
        ha="left", va="top", fontfamily=font_family, zorder=6,
    )

    # 6. Row 2 Right Bottom: Key Insights Card
    mpl_round_rect(ax, 7.70, 4.60, 5.21, 1.64, HEX["white"], HEX["border"], lw=1.2, rounding=0.10, zorder=3)
    ax.text(
        7.84, SLIDE_H - 4.78,
        "Temuan Kunci Penghapusan Noise",
        fontsize=10.8, fontweight="bold", color=HEX["navy"],
        ha="left", va="center", fontfamily=font_family, zorder=6,
    )

    cur_y_top = 4.96
    for lead, body, lead_col in INSIGHTS:
        full = f"• {lead} {body}"
        wrapped = wrap_text_lines(full, 72)
        # Render lead bold on line 1, rest normal
        first_line = wrapped[0]
        lead_prefix = f"• {lead} "
        if first_line.startswith(lead_prefix):
            rest_first = first_line[len(lead_prefix):]
            t_lead = ax.text(
                7.84, SLIDE_H - cur_y_top, lead_prefix,
                fontsize=8.3, fontweight="bold", color=lead_col,
                ha="left", va="top", fontfamily=font_family, zorder=6,
            )
            # Measure lead width using renderer via transform later or simple char estimation?
            # Use canvas draw once or exact text split for clean alignment:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = t_lead.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
            ax.text(
                bb.x1, SLIDE_H - cur_y_top, rest_first,
                fontsize=8.3, color=HEX["slate_dark"],
                ha="left", va="top", fontfamily=font_family, zorder=6,
            )
        else:
            ax.text(
                7.84, SLIDE_H - cur_y_top, first_line,
                fontsize=8.3, fontweight="bold", color=lead_col,
                ha="left", va="top", fontfamily=font_family, zorder=6,
            )
        cur_y_top += 0.178
        for extra_line in wrapped[1:]:
            ax.text(
                7.96, SLIDE_H - cur_y_top, extra_line,
                fontsize=8.3, color=HEX["slate_dark"],
                ha="left", va="top", fontfamily=font_family, zorder=6,
            )
            cur_y_top += 0.175
        cur_y_top += 0.075

    # 7. Bottom Transparency Callout
    mpl_round_rect(ax, 0.42, 6.36, 12.49, 0.58, HEX["amber_bg"], HEX["amber"], lw=1.3, rounding=0.10, zorder=3)
    callout_lead = "Catatan Transparansi Metodologis: "
    callout_body = (
        "Karena PDFE24_1 berada di Test Set (7 → 6 subjek) dan PDFE27 di Val Set (5 → 4 subjek), "
        "sebagian kenaikan Fixed Split berasal dari penghapusan outlier evaluasi. "
        "Validasi penuh LOSO 32-Fold membuktikan generalisasi tetap kuat (ROC-AUC 0.917, Bal-Acc 82.4%)."
    )
    cl_lines = wrap_text_lines(callout_lead + callout_body, 152)
    t_cl = ax.text(
        0.58, SLIDE_H - 6.475, callout_lead,
        fontsize=8.5, fontweight="bold", color=HEX["amber_dark"],
        ha="left", va="top", fontfamily=font_family, zorder=6,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb_cl = t_cl.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
    ax.text(
        bb_cl.x1, SLIDE_H - 6.475, cl_lines[0][len(callout_lead):],
        fontsize=8.5, color=HEX["slate_dark"],
        ha="left", va="top", fontfamily=font_family, zorder=6,
    )
    if len(cl_lines) > 1:
        ax.text(
            0.58, SLIDE_H - 6.675, " ".join(cl_lines[1:]),
            fontsize=8.5, color=HEX["slate_dark"],
            ha="left", va="top", fontfamily=font_family, zorder=6,
        )

    # 8. Footer
    mpl_round_rect(ax, 0.42, 7.06, 12.49, 0.015, HEX["border"], rounding=0.0, zorder=4)
    ax.text(
        0.42, SLIDE_H - 7.25,
        "Mobile-FOG  •  Deteksi Freezing of Gait Pasien Parkinson via Video Smartphone  •  Kelompok 4 ITS",
        fontsize=8.8, color=HEX["muted"],
        ha="left", va="center", fontfamily=font_family, zorder=6,
    )
    ax.text(
        12.91, SLIDE_H - 7.25,
        "Slide 7",
        fontsize=8.8, fontweight="bold", color=HEX["muted"],
        ha="right", va="center", fontfamily=font_family, zorder=6,
    )

    fig.savefig(PNG_PATH, dpi=200, facecolor=HEX["bg"])
    fig.savefig(PDF_PATH, facecolor=HEX["bg"])
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font_family = configure_fonts()
    print(f"Using primary font: {font_family}")
    build_chart(font_family)
    build_pptx(font_family)
    build_preview_and_pdf(font_family)
    for p in (CHART_PATH, PPTX_PATH, PNG_PATH, PDF_PATH):
        print(f"Generated: {p} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
