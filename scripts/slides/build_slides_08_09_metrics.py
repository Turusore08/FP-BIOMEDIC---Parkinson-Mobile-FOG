"""Generate Slide 8 & Slide 9 from REAL Mobile-FOG evaluation data (no placeholders).

Reads directly from:
  - results/metrics/baseline_35subj_results_summary.csv
  - results/metrics/bbox_35subj_results_summary.csv
  - results/metrics/clean_32subj_results_summary.csv
  - results/metrics/clean_32subj_loso_oof_B_15hz.csv
  - results/metrics/clean_32subj_loso_per_subject_B_15hz.csv
  - outputs/training/clean_32subj/mobilefog_training_results/loso_35fold_B_15hz/fold_*.npz

Outputs written to outputs/slides/:
  - chart_cm_loso_top2.png
  - chart_roc_pr_loso32.png
  - chart_persubject_loso.png
  - slides_08_09_metrik_evaluasi.pptx
  - slide_08_metrik_evaluasi_bag1.png
  - slide_09_metrik_evaluasi_bag2.png
  - slides_08_09_metrik_evaluasi.pdf
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt
from scipy import stats

from mobilefog.metrics import (
    apply_min_duration_filter,
    compute_binary_classification_metrics,
    compute_roc_pr_curves,
)
from mobilefog.paths import RESULTS_METRICS_DIR, SLIDES_OUTPUT_DIR, TRAINING_CLEAN_DIR

OUT_DIR = SLIDES_OUTPUT_DIR

CSV_RAW = RESULTS_METRICS_DIR / "baseline_35subj_results_summary.csv"
CSV_BBOX = RESULTS_METRICS_DIR / "bbox_35subj_results_summary.csv"
CSV_CLEAN = RESULTS_METRICS_DIR / "clean_32subj_results_summary.csv"
CSV_OOF = RESULTS_METRICS_DIR / "clean_32subj_loso_oof_B_15hz.csv"
CSV_SUBJ = RESULTS_METRICS_DIR / "clean_32subj_loso_per_subject_B_15hz.csv"
LOSO_NPZ_GLOB = str(TRAINING_CLEAN_DIR / "mobilefog_training_results/loso_35fold_B_15hz/fold_*.npz")

CHART_CM_PATH = OUT_DIR / "chart_cm_loso_top2.png"
CHART_ROC_PR_PATH = OUT_DIR / "chart_roc_pr_loso32.png"
CHART_SUBJ_PATH = OUT_DIR / "chart_persubject_loso.png"

PPTX_PATH = OUT_DIR / "slides_08_09_metrik_evaluasi.pptx"
PNG_8_PATH = OUT_DIR / "slide_08_metrik_evaluasi_bag1.png"
PNG_9_PATH = OUT_DIR / "slide_09_metrik_evaluasi_bag2.png"
PDF_PATH = OUT_DIR / "slides_08_09_metrik_evaluasi.pdf"

SLIDE_W = 13.333
SLIDE_H = 7.500

HEX = {
    "bg": "#F8FAFC",
    "white": "#FFFFFF",
    "border": "#CBD5E1",
    "row_alt": "#F8FAFC",
    "slate_100": "#F1F5F9",
    "navy": "#0F172A",
    "slate_dark": "#1E293B",
    "slate_mid": "#475569",
    "muted": "#64748B",
    "blue": "#2563EB",
    "blue_bg": "#EFF6FF",
    "blue_border": "#93C5FD",
    "blue_dark": "#1E3A8A",
    "green": "#16A34A",
    "green_bg": "#F0FDF4",
    "green_border": "#86EFAC",
    "green_dark": "#14532D",
    "purple": "#7C3AED",
    "purple_bg": "#F5F3FF",
    "purple_border": "#C4B5FD",
    "purple_dark": "#5B21B6",
    "amber": "#D97706",
    "amber_bg": "#FFFBEB",
    "amber_border": "#FCD34D",
    "amber_dark": "#92400E",
    "red": "#DC2626",
    "red_bg": "#FEF2F2",
    "grid": "#E2E8F0",
}

MODEL_META = [
    ("lgbm", "LightGBM+FFT", "#2563EB"),
    ("stgcn", "Light_ST-GCN", "#16A34A"),
    ("mamba", "Pure_Bi-Mamba", "#D97706"),
    ("hybrid", "Hybrid_GCN+BiMamba", "#DC2626"),
]


def hex_to_rgb(h: str) -> RGBColor:
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def configure_fonts() -> str:
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


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "< 0.0001"
    if p < 1e-3:
        return f"{p:.4f}"
    return f"{p:.3f}"


def fmt_p_eq(p: float) -> str:
    if p < 1e-4:
        return "p < 0.0001"
    if p < 1e-3:
        return f"p = {p:.4f}"
    return f"p = {p:.3f}"


# ---------------------------------------------------------------------------
# Data loading & exact metric + scipy.stats computations
# ---------------------------------------------------------------------------
def load_all_metrics():
    # 1. Load 16-run fixed split CSVs
    def load_runs(path: Path):
        with open(path) as f:
            return {(r["scenario"], r["model"]): {k: (float(v) if k not in ("scenario", "model") else v) for k, v in r.items()} for r in csv.DictReader(f)}

    raw_runs = load_runs(CSV_RAW)
    bbox_runs = load_runs(CSV_BBOX)
    clean_runs = load_runs(CSV_CLEAN)
    run_keys = sorted(clean_runs.keys())

    # 2. Load 32-fold LOSO OOF predictions (.npz)
    npz_files = sorted(glob.glob(LOSO_NPZ_GLOB))
    assert len(npz_files) == 32, f"Expected 32 fold npz files, found {len(npz_files)}"

    y_all_list, tid_list = [], []
    probs = {k: [] for k, _, _ in MODEL_META}
    preds_pp = {k: [] for k, _, _ in MODEL_META}

    for p in npz_files:
        d = np.load(p, allow_pickle=True)
        y = d["y_te"].astype(np.int64)
        tids = d["trial_id"]
        y_all_list.append(y)
        tid_list.append(tids)
        for key, _, _ in MODEL_META:
            pr = d[f"p_{key}"].astype(np.float64)
            thr = float(d[f"thr_{key}"])
            pb_raw = (pr >= thr).astype(np.int64)
            pb_pp = apply_min_duration_filter(pb_raw, tids, min_run=2)
            probs[key].append(pr)
            preds_pp[key].append(pb_pp)

    y_oof = np.concatenate(y_all_list)
    oof_csv = {r["model"]: r for r in csv.DictReader(open(CSV_OOF))}

    loso_models = {}
    for key, name, color in MODEL_META:
        p_oof = np.concatenate(probs[key])
        yp_oof = np.concatenate(preds_pp[key])
        fpr, tpr, roc_auc, rec_p, prec_p, pr_auc = compute_roc_pr_curves(y_oof, p_oof)

        # Assert exact match with loso_oof_summary_B_15hz.csv
        csv_roc = float(oof_csv[name]["oof_roc_auc"])
        csv_pr = float(oof_csv[name]["oof_pr_auc"])
        assert abs(roc_auc - csv_roc) < 5e-4, f"ROC mismatch {name}: {roc_auc} vs {csv_roc}"
        assert abs(pr_auc - csv_pr) < 5e-4, f"PR mismatch {name}: {pr_auc} vs {csv_pr}"

        cls_metrics = compute_binary_classification_metrics(y_oof, yp_oof)

        loso_models[key] = {
            "key": key,
            "name": name,
            "color": color,
            "p_oof": p_oof,
            **cls_metrics,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fpr": fpr, "tpr": tpr,
            "rec_curve": rec_p, "prec_curve": prec_p,
        }

    # 3. Per-subject LOSO stratification (32 subjects = 9 Non-FoG + 23 FoG-active)
    subj_rows = list(csv.DictReader(open(CSV_SUBJ)))
    per_subj = {}
    for _, name, _ in MODEL_META:
        mrows = [r for r in subj_rows if r["model"] == name]
        nonfog = [r for r in mrows if r["fog_active"] == "0"]
        fog = [r for r in mrows if r["fog_active"] == "1"]
        fog_hi = [r for r in fog if float(r["fog_pct"]) >= 10.0]
        fog_lo = [r for r in fog if float(r["fog_pct"]) < 10.0]
        per_subj[name] = {
            "nonfog": nonfog,
            "fog": fog,
            "fog_hi": fog_hi,
            "fog_lo": fog_lo,
            "nonfog_spec_mean": float(np.mean([float(r["spec_pp"]) for r in nonfog])),
            "nonfog_perfect_cnt": sum(float(r["spec_pp"]) == 1.0 for r in nonfog),
            "fog_hi_pr_mean": float(np.mean([float(r["pr_auc"]) for r in fog_hi])),
            "fog_lo_pr_mean": float(np.mean([float(r["pr_auc"]) for r in fog_lo])),
            "fog_lo_roc_mean": float(np.mean([float(r["roc_auc"]) for r in fog_lo])),
            "fog_pct": np.array([float(r["fog_pct"]) for r in fog]),
            "fog_pr": np.array([float(r["pr_auc"]) for r in fog]),
            "fog_f1": np.array([float(r["f1_pp"]) for r in fog]),
            "fog_roc": np.array([float(r["roc_auc"]) for r in fog]),
        }

    # 4. Exact scipy.stats Hypothesis Tests & Spearman Correlations
    pr_raw = np.array([raw_runs[k]["test_pr_auc"] for k in run_keys])
    pr_bbox = np.array([bbox_runs[k]["test_pr_auc"] for k in run_keys])
    pr_clean = np.array([clean_runs[k]["test_pr_auc"] for k in run_keys])
    f1_bbox = np.array([bbox_runs[k]["test_f1_pp"] for k in run_keys])
    f1_clean = np.array([clean_runs[k]["test_f1_pp"] for k in run_keys])

    sw_stat, sw_p = stats.shapiro(pr_clean - pr_raw)
    kw_stat, kw_p = stats.kruskal(pr_raw, pr_bbox, pr_clean)
    wil_bb_stat, wil_bb_p = stats.wilcoxon(pr_bbox, pr_raw, alternative="greater")
    wil_cl_stat, wil_cl_p = stats.wilcoxon(pr_clean, pr_bbox, alternative="greater")
    wil_f1_stat, wil_f1_p = stats.wilcoxon(f1_clean, f1_bbox, alternative="greater")

    st_hi = np.array([float(r["pr_auc"]) for r in per_subj["Light_ST-GCN"]["fog_hi"]])
    st_lo = np.array([float(r["pr_auc"]) for r in per_subj["Light_ST-GCN"]["fog_lo"]])
    mw_st_stat, mw_st_p = stats.mannwhitneyu(st_hi, st_lo, alternative="greater")

    lg_hi = np.array([float(r["pr_auc"]) for r in per_subj["LightGBM+FFT"]["fog_hi"]])
    lg_lo = np.array([float(r["pr_auc"]) for r in per_subj["LightGBM+FFT"]["fog_lo"]])
    mw_lg_stat, mw_lg_p = stats.mannwhitneyu(lg_hi, lg_lo, alternative="greater")

    # Friedman test across 4 models on n=23 FoG subjects (ROC-AUC & F1-PP)
    fr_f1_stat, fr_f1_p = stats.friedmanchisquare(
        per_subj["LightGBM+FFT"]["fog_f1"],
        per_subj["Light_ST-GCN"]["fog_f1"],
        per_subj["Pure_Bi-Mamba"]["fog_f1"],
        per_subj["Hybrid_GCN+BiMamba"]["fog_f1"],
    )

    # Spearman correlations
    rho_st_lg, p_st_lg = stats.spearmanr(loso_models["stgcn"]["p_oof"], loso_models["lgbm"]["p_oof"])
    rho_st_mb, p_st_mb = stats.spearmanr(loso_models["stgcn"]["p_oof"], loso_models["mamba"]["p_oof"])
    rho_lg_mb, p_lg_mb = stats.spearmanr(loso_models["lgbm"]["p_oof"], loso_models["mamba"]["p_oof"])

    rho_prev_st, p_prev_st = stats.spearmanr(per_subj["Light_ST-GCN"]["fog_pct"], per_subj["Light_ST-GCN"]["fog_pr"])
    rho_prev_lg, p_prev_lg = stats.spearmanr(per_subj["LightGBM+FFT"]["fog_pct"], per_subj["LightGBM+FFT"]["fog_pr"])

    stat_results = {
        "shapiro": (float(sw_stat), float(sw_p)),
        "kruskal": (float(kw_stat), float(kw_p)),
        "wilcoxon_bbox_raw": (float(wil_bb_stat), float(wil_bb_p)),
        "wilcoxon_clean_bbox_pr": (float(wil_cl_stat), float(wil_cl_p)),
        "wilcoxon_clean_bbox_f1": (float(wil_f1_stat), float(wil_f1_p)),
        "mannwhitney_stgcn": (float(mw_st_stat), float(mw_st_p)),
        "mannwhitney_lgbm": (float(mw_lg_stat), float(mw_lg_p)),
        "friedman_f1": (float(fr_f1_stat), float(fr_f1_p)),
        "spearman_st_lg": (float(rho_st_lg), float(p_st_lg)),
        "spearman_st_mb": (float(rho_st_mb), float(p_st_mb)),
        "spearman_lg_mb": (float(rho_lg_mb), float(p_lg_mb)),
        "spearman_prev_st": (float(rho_prev_st), float(p_prev_st)),
        "spearman_prev_lg": (float(rho_prev_lg), float(p_prev_lg)),
    }

    print("=== VERIFIED LOSO 32-FOLD OOF METRICS ===")
    for k, _, _ in MODEL_META:
        m = loso_models[k]
        print(
            f"{m['name']:18s} | TP={m['tp']:4d} FP={m['fp']:3d} FN={m['fn']:3d} TN={m['tn']:4d} | "
            f"MacroP={m['macro_prec']:.4f} MacroR={m['macro_rec']:.4f} Spec={m['spec']:.4f} MacroF1={m['macro_f1']:.4f} "
            f"F1_FoG={m['f1_fog']:.4f} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} Err={100*m['err_rate']:.2f}%"
        )
    print("=== VERIFIED SCIPY.STATS RESULTS ===")
    for k, v in stat_results.items():
        print(f"  {k:24s}: stat={v[0]:.4f}, p={v[1]:.6g}")

    return y_oof, loso_models, per_subj, stat_results, clean_runs


# ---------------------------------------------------------------------------
# Charts (300 DPI)
# ---------------------------------------------------------------------------
def build_chart_cm(font_family: str, loso_models: dict):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.45), dpi=300)
    fig.patch.set_facecolor(HEX["white"])

    configs = [
        (axes[0], loso_models["lgbm"], "Blues", f"LightGBM+FFT (Err {100*loso_models['lgbm']['err_rate']:.2f}%)"),
        (axes[1], loso_models["stgcn"], "Greens", f"Light_ST-GCN (Err {100*loso_models['stgcn']['err_rate']:.2f}%)"),
    ]
    labels = ["Non-FoG\n(0)", "FoG\n(1)"]

    for ax, m, cmap, title in configs:
        cm = m["cm"]
        row_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
        ax.imshow(row_pct, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)
        ax.set_title(title, fontsize=8.8, fontweight="bold", color=HEX["navy"], fontfamily=font_family, pad=6)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=7.6, color=HEX["slate_dark"], fontfamily=font_family)
        ax.set_yticklabels(labels, fontsize=7.6, color=HEX["slate_dark"], fontfamily=font_family)
        ax.set_xlabel("Prediksi Model", fontsize=7.8, fontweight="bold", color=HEX["slate_mid"], fontfamily=font_family, labelpad=4)
        if ax is axes[0]:
            ax.set_ylabel("Aktual (Ground Truth)", fontsize=7.8, fontweight="bold", color=HEX["slate_mid"], fontfamily=font_family, labelpad=4)

        cell_tags = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = row_pct[i, j]
                txt_col = "#FFFFFF" if pct >= 55 else HEX["navy"]
                ax.text(
                    j, i,
                    f"{cell_tags[i][j]}: {val:,}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=8.2, fontweight="bold", color=txt_col,
                    fontfamily=font_family,
                )

    plt.tight_layout(pad=0.45, w_pad=1.2)
    fig.savefig(CHART_CM_PATH, dpi=300, facecolor=HEX["white"])
    plt.close(fig)


def build_chart_roc_pr(font_family: str, y_oof: np.ndarray, loso_models: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.75, 2.45), dpi=300)
    fig.patch.set_facecolor(HEX["white"])

    # Plot ROC Curve (Left)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.9, alpha=0.6)
    for key, _, _ in MODEL_META:
        m = loso_models[key]
        short = m["name"].replace("Hybrid_GCN+BiMamba", "Hybrid")
        ax1.plot(m["fpr"], m["tpr"], color=m["color"], linewidth=1.6, label=f"{short} ({m['roc_auc']:.3f})")
    ax1.set_title("32-Fold LOSO OOF ROC", fontsize=8.8, fontweight="bold", color=HEX["navy"], fontfamily=font_family, pad=5)
    ax1.set_xlabel("False Positive Rate", fontsize=7.6, color=HEX["slate_mid"], fontfamily=font_family)
    ax1.set_ylabel("Sensitivity (TPR)", fontsize=7.6, color=HEX["slate_mid"], fontfamily=font_family)
    ax1.tick_params(labelsize=7.0)
    ax1.grid(True, linestyle=":", color=HEX["grid"], linewidth=0.7)
    ax1.legend(loc="lower right", fontsize=6.4, frameon=True, facecolor=HEX["white"], edgecolor=HEX["border"], borderpad=0.3)

    # Plot PR Curve (Right)
    pos_rate = float(y_oof.mean())
    ax2.axhline(pos_rate, color="#64748B", linestyle=":", linewidth=0.9, label=f"Chance ({pos_rate:.2f})")
    for key, _, _ in MODEL_META:
        m = loso_models[key]
        short = m["name"].replace("Hybrid_GCN+BiMamba", "Hybrid")
        ax2.plot(m["rec_curve"], m["prec_curve"], color=m["color"], linewidth=1.6, label=f"{short} ({m['pr_auc']:.3f})")
    ax2.set_title("32-Fold LOSO OOF PR", fontsize=8.8, fontweight="bold", color=HEX["navy"], fontfamily=font_family, pad=5)
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=7.6, color=HEX["slate_mid"], fontfamily=font_family)
    ax2.set_ylabel("Precision", fontsize=7.6, color=HEX["slate_mid"], fontfamily=font_family)
    ax2.set_ylim(0.0, 1.03)
    ax2.tick_params(labelsize=7.0)
    ax2.grid(True, linestyle=":", color=HEX["grid"], linewidth=0.7)
    ax2.legend(loc="upper right", fontsize=6.2, frameon=True, facecolor=HEX["white"], edgecolor=HEX["border"], borderpad=0.3)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(HEX["border"])
        ax.spines["bottom"].set_color(HEX["border"])

    plt.tight_layout(pad=0.40, w_pad=0.9)
    fig.savefig(CHART_ROC_PR_PATH, dpi=300, facecolor=HEX["white"])
    plt.close(fig)


def build_chart_persubject(font_family: str, per_subj: dict, stat_results: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.05, 4.35), dpi=300, gridspec_kw={"height_ratios": [1.0, 1.05]})
    fig.patch.set_facecolor(HEX["white"])

    # Panel A (Top): Stratified Per-Subject Performance across 4 models
    groups = ["Non-FoG Spec\n(n=9 pasien)", "FoG ≥10% PR-AUC\n(n=13 pasien)", "FoG <10% ROC-AUC\n(n=10 pasien)"]
    x = np.arange(len(groups))
    w = 0.19
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    model_order = [
        ("LightGBM+FFT", "LightGBM+FFT", "#2563EB"),
        ("Light_ST-GCN", "Light_ST-GCN", "#16A34A"),
        ("Pure_Bi-Mamba", "Pure_Bi-Mamba", "#D97706"),
        ("Hybrid_GCN+BiMamba", "Hybrid", "#DC2626"),
    ]
    for (mname, short, col), off in zip(model_order, offsets):
        ps = per_subj[mname]
        vals = [ps["nonfog_spec_mean"], ps["fog_hi_pr_mean"], ps["fog_lo_roc_mean"]]
        bars = ax1.bar(x + off, vals, width=w * 0.92, color=col, edgecolor="#1E293B", linewidth=0.5, label=short, zorder=3)
        for rect, val in zip(bars, vals):
            ax1.text(
                rect.get_x() + rect.get_width() / 2.0,
                val + 0.015,
                f".{int(round(val * 1000)):03d}" if val < 0.9995 else "1.00",
                ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=HEX["slate_dark"], fontfamily=font_family,
            )

    ax1.set_title("A. Stratifikasi 32 Subjek LOSO per Kelompok Klinis", fontsize=8.9, fontweight="bold", color=HEX["navy"], fontfamily=font_family, loc="left", pad=22)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=7.8, fontweight="bold", color=HEX["slate_dark"], fontfamily=font_family)
    ax1.set_ylim(0.0, 1.14)
    ax1.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax1.tick_params(axis="y", labelsize=7.0)
    ax1.tick_params(axis="x", length=0, pad=4)
    ax1.yaxis.grid(True, linestyle=":", color=HEX["grid"], linewidth=0.8, zorder=1)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=6.6, frameon=True, facecolor=HEX["bg"], edgecolor=HEX["border"], handlelength=1.1, columnspacing=0.8)

    # Panel B (Bottom): Scatter FoG Prevalence (%) vs Per-Subject PR-AUC (n=23 FoG subjects)
    ps_st = per_subj["Light_ST-GCN"]
    ps_lg = per_subj["LightGBM+FFT"]
    rho_st, p_st = stat_results["spearman_prev_st"]
    rho_lg, p_lg = stat_results["spearman_prev_lg"]

    ax2.axvline(10.0, color="#94A3B8", linestyle="--", linewidth=1.0, zorder=1)
    ax2.text(11.5, 0.08, "Batas FoG ≥ 10%\n(episode stabil)", fontsize=6.8, color=HEX["muted"], fontfamily=font_family)

    ax2.scatter(ps_st["fog_pct"], ps_st["fog_pr"], color="#16A34A", edgecolor="#14532D", s=34, alpha=0.88, label=f"Light_ST-GCN (ρ = {rho_st:.3f}, {fmt_p_eq(p_st)})", zorder=4)
    ax2.scatter(ps_lg["fog_pct"], ps_lg["fog_pr"], color="#2563EB", edgecolor="#1E3A8A", marker="^", s=34, alpha=0.85, label=f"LightGBM+FFT (ρ = {rho_lg:.3f}, {fmt_p_eq(p_lg)})", zorder=4)

    ax2.set_title("B. Korelasi Prevalensi FoG (%) vs PR-AUC (n=23 Subjek FoG-Aktif)", fontsize=8.9, fontweight="bold", color=HEX["navy"], fontfamily=font_family, loc="left", pad=6)
    ax2.set_xlabel("Persentase Window FoG per Subjek (%)", fontsize=7.8, fontweight="bold", color=HEX["slate_mid"], fontfamily=font_family)
    ax2.set_ylabel("Per-Subject PR-AUC", fontsize=7.8, fontweight="bold", color=HEX["slate_mid"], fontfamily=font_family)
    ax2.set_xlim(-3, 104)
    ax2.set_ylim(-0.04, 1.07)
    ax2.tick_params(labelsize=7.0)
    ax2.grid(True, linestyle=":", color=HEX["grid"], linewidth=0.8, zorder=1)
    ax2.legend(loc="lower right", fontsize=6.9, frameon=True, facecolor=HEX["white"], edgecolor=HEX["border"])

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(HEX["border"])
        ax.spines["bottom"].set_color(HEX["border"])

    plt.tight_layout(pad=0.45, h_pad=1.0)
    fig.savefig(CHART_SUBJ_PATH, dpi=300, facecolor=HEX["white"])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dual-renderer canvas (PPTX native + 1:1 Matplotlib PNG/PDF)
# ---------------------------------------------------------------------------
def set_cell_margins(cell, left=0.05, right=0.05, top=0.02, bottom=0.02):
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


class DualSlide:
    def __init__(self, pptx_slide, font_family: str):
        self.ps = pptx_slide
        self.font = font_family
        self.fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=200)
        self.ax = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
        self.fig.patch.set_facecolor(HEX["bg"])
        self.ax.set_facecolor(HEX["bg"])
        self.ax.set_xlim(0.0, SLIDE_W)
        self.ax.set_ylim(0.0, SLIDE_H)
        self.ax.axis("off")

        fill = self.ps.background.fill
        fill.solid()
        fill.fore_color.rgb = hex_to_rgb(HEX["bg"])

    def rect(self, x, y, w, h, fill_hex, edge_hex=None, lw=1.2, rounding=0.0, radius_adj=0.06, zorder=2):
        stype = MSO_SHAPE.ROUNDED_RECTANGLE if rounding > 0 else MSO_SHAPE.RECTANGLE
        shp = self.ps.shapes.add_shape(stype, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb = hex_to_rgb(fill_hex)
        if edge_hex:
            shp.line.color.rgb = hex_to_rgb(edge_hex)
            shp.line.width = Pt(lw)
        else:
            shp.line.fill.background()
        if rounding > 0 and shp.adjustments:
            shp.adjustments[0] = radius_adj

        y_bot = SLIDE_H - y - h
        if rounding <= 0:
            patch = Rectangle((x, y_bot), w, h, facecolor=fill_hex, edgecolor=edge_hex if edge_hex else "none", linewidth=lw if edge_hex else 0, zorder=zorder)
        else:
            patch = FancyBboxPatch((x, y_bot), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fill_hex, edgecolor=edge_hex if edge_hex else "none", linewidth=lw if edge_hex else 0, zorder=zorder)
        self.ax.add_patch(patch)

    def picture_fit(self, img_path: Path, box_l: float, box_t: float, box_w: float, box_h: float, zorder: int = 5):
        with Image.open(img_path) as im:
            arr = np.asarray(im)
        ratio = arr.shape[1] / arr.shape[0]
        if ratio >= box_w / box_h:
            cw, ch = box_w, box_w / ratio
        else:
            ch, cw = box_h, box_h * ratio
        cx = box_l + (box_w - cw) / 2.0
        cy = box_t + (box_h - ch) / 2.0
        self.ps.shapes.add_picture(str(img_path), Inches(cx), Inches(cy), width=Inches(cw), height=Inches(ch))
        self.ax.imshow(arr, extent=(cx, cx + cw, SLIDE_H - cy - ch, SLIDE_H - cy), aspect="auto", zorder=zorder, interpolation="lanczos")

    def pill(self, x, y, w, h, text, fill_hex, edge_hex, text_hex, fsize=8.4, bold=True, lw=1.0, rounding=0.08, zorder=6):
        self.rect(x, y, w, h, fill_hex, edge_hex, lw=lw, rounding=rounding, radius_adj=0.25, zorder=zorder)
        self.text_box(x, y, w, h, [(text, fsize, bold, text_hex)], align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, mx=0.02, my=0.01, zorder=zorder + 1)

    def table(self, x: float, y: float, w: float, h: float, col_ws: list[float], headers: list[str], rows: list[list[tuple[str, str, bool]]], row_bgs: list[str], fsize: float = 8.2, hdr_fsize: float = 8.3):
        n_rows = len(rows) + 1
        n_cols = len(headers)
        row_h = h / n_rows

        # PPTX Native Table
        tbl_shape = self.ps.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w), Inches(h))
        tbl = tbl_shape.table
        for ci, cw in enumerate(col_ws):
            tbl.columns[ci].width = Inches(cw)

        for ci, htxt in enumerate(headers):
            cell = tbl.cell(0, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = hex_to_rgb(HEX["navy"])
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            set_cell_margins(cell, 0.04, 0.04, 0.02, 0.02)
            set_cell_border(cell, HEX["navy"], 0.5)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = htxt
            r.font.name = self.font
            r.font.size = Pt(hdr_fsize)
            r.font.bold = True
            r.font.color.rgb = hex_to_rgb(HEX["white"])

        for ri, (row, rbg) in enumerate(zip(rows, row_bgs), start=1):
            for ci, (val, tcol, is_b) in enumerate(row):
                cell = tbl.cell(ri, ci)
                cell.fill.solid()
                cell.fill.fore_color.rgb = hex_to_rgb(rbg)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                set_cell_margins(cell, 0.04, 0.04, 0.02, 0.02)
                set_cell_border(cell, HEX["border"], 0.6)
                p = cell.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
                r = p.add_run()
                r.text = val
                r.font.name = self.font
                r.font.size = Pt(fsize)
                r.font.bold = is_b
                r.font.color.rgb = hex_to_rgb(tcol)

        # Matplotlib 1:1 Table
        cx = x
        for ci, (htxt, cw) in enumerate(zip(headers, col_ws)):
            y_bot = SLIDE_H - y - row_h
            self.ax.add_patch(Rectangle((cx, y_bot), cw, row_h, facecolor=HEX["navy"], edgecolor=HEX["navy"], linewidth=0.6, zorder=5))
            self.ax.text(
                cx + (0.06 if ci == 0 else cw / 2.0),
                SLIDE_H - (y + row_h / 2.0),
                htxt, fontsize=hdr_fsize, fontweight="bold", color=HEX["white"],
                ha="left" if ci == 0 else "center", va="center", fontfamily=self.font, zorder=7,
            )
            cx += cw

        for ri, (row, rbg) in enumerate(zip(rows, row_bgs), start=1):
            ry = y + ri * row_h
            y_bot = SLIDE_H - ry - row_h
            cx = x
            for ci, ((val, tcol, is_b), cw) in enumerate(zip(row, col_ws)):
                self.ax.add_patch(Rectangle((cx, y_bot), cw, row_h, facecolor=rbg, edgecolor=HEX["border"], linewidth=0.6, zorder=5))
                self.ax.text(
                    cx + (0.06 if ci == 0 else cw / 2.0),
                    SLIDE_H - (ry + row_h / 2.0),
                    val, fontsize=fsize, fontweight="bold" if is_b else "normal", color=tcol,
                    ha="left" if ci == 0 else "center", va="center", fontfamily=self.font, zorder=7,
                )
                cx += cw

    def text_box(self, x, y, w, h, runs, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, mx=0.0, my=0.0, zorder=7, line_spacing=1.25):
        tb = self.ps.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = valign
        tf.margin_left = Inches(mx)
        tf.margin_right = Inches(mx)
        tf.margin_top = Inches(my)
        tf.margin_bottom = Inches(my)
        p = tf.paragraphs[0]
        p.alignment = align
        p.space_after = Pt(1.5)
        for txt, fsize, is_bold, col_hex in runs:
            if txt == "\n":
                p = tf.add_paragraph()
                p.alignment = align
                p.space_after = Pt(1.5)
                continue
            r = p.add_run()
            r.text = txt
            r.font.name = self.font
            r.font.size = Pt(fsize)
            r.font.bold = is_bold
            r.font.color.rgb = hex_to_rgb(col_hex)

        paragraphs = [[]]
        for item in runs:
            if item[0] == "\n":
                paragraphs.append([])
            else:
                paragraphs[-1].append(item)

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        avail_w = max(0.2, w - 2 * mx)

        rendered_lines = []
        line_heights = []
        for para in paragraphs:
            tokens = []
            for txt, fsize, is_bold, col_hex in para:
                parts = txt.split(" ")
                for idx, part in enumerate(parts):
                    if part == "" and idx < len(parts) - 1:
                        tokens.append((" ", fsize, is_bold, col_hex))
                    elif part != "":
                        piece = part + (" " if idx < len(parts) - 1 else "")
                        tokens.append((piece, fsize, is_bold, col_hex))
                    elif idx == len(parts) - 1 and txt.endswith(" "):
                        tokens.append((" ", fsize, is_bold, col_hex))

            if not tokens:
                rendered_lines.append([])
                line_heights.append(0.13)
                continue

            cur_line = []
            cur_w = 0.0
            max_fs = max(t[1] for t in tokens)
            for tok, fsize, is_bold, col_hex in tokens:
                probe = self.ax.text(0, 0, tok, fontsize=fsize, fontweight="bold" if is_bold else "normal", fontfamily=self.font, alpha=0.0)
                bb = probe.get_window_extent(renderer=renderer).transformed(self.ax.transData.inverted())
                probe.remove()
                tw = bb.width
                if cur_line and (cur_w + tw > avail_w) and tok.strip():
                    rendered_lines.append(cur_line)
                    line_heights.append((max_fs / 72.0) * line_spacing)
                    cur_line = [(tok, fsize, is_bold, col_hex)]
                    cur_w = tw
                else:
                    cur_line.append((tok, fsize, is_bold, col_hex))
                    cur_w += tw
            if cur_line:
                rendered_lines.append(cur_line)
                line_heights.append((max_fs / 72.0) * line_spacing)

        total_h = sum(line_heights)
        if valign == MSO_ANCHOR.MIDDLE:
            cur_y = y + max(my, (h - total_h) / 2.0)
        elif valign == MSO_ANCHOR.BOTTOM:
            cur_y = y + max(my, h - my - total_h)
        else:
            cur_y = y + my

        for line_tokens, lh in zip(rendered_lines, line_heights):
            if not line_tokens:
                cur_y += lh
                continue
            last_txt, last_fs, last_b, last_c = line_tokens[-1]
            line_tokens[-1] = (last_txt.rstrip(" "), last_fs, last_b, last_c)
            t_widths = []
            for tok, fsize, is_bold, _ in line_tokens:
                probe = self.ax.text(0, 0, tok, fontsize=fsize, fontweight="bold" if is_bold else "normal", fontfamily=self.font, alpha=0.0)
                bb = probe.get_window_extent(renderer=renderer).transformed(self.ax.transData.inverted())
                probe.remove()
                t_widths.append(bb.width)
            full_w = sum(t_widths)
            if align == PP_ALIGN.CENTER:
                cx = x + mx + (avail_w - full_w) / 2.0
            elif align == PP_ALIGN.RIGHT:
                cx = x + w - mx - full_w
            else:
                cx = x + mx
            cy_mid = cur_y + lh / 2.0
            for (tok, fsize, is_bold, col_hex), tw in zip(line_tokens, t_widths):
                if tok:
                    self.ax.text(cx, SLIDE_H - cy_mid, tok, fontsize=fsize, fontweight="bold" if is_bold else "normal", color=col_hex, ha="left", va="center", fontfamily=self.font, zorder=zorder)
                cx += tw
            cur_y += lh


def draw_chrome(ds: DualSlide, bar_l: str, bar_r: str, pill_txt: str, pill_bg: str, pill_edge: str, pill_fg: str, title: str, sub: str, r_title: str, r_sub: str, slide_lbl: str):
    ds.rect(0.0, 0.0, SLIDE_W / 2.0, 0.08, bar_l, rounding=0.0, zorder=10)
    ds.rect(SLIDE_W / 2.0, 0.0, SLIDE_W / 2.0, 0.08, bar_r, rounding=0.0, zorder=10)

    ds.pill(0.42, 0.18, 3.25, 0.28, pill_txt, pill_bg, pill_edge, pill_fg, fsize=8.3, lw=1.0, rounding=0.10)
    ds.text_box(0.42, 0.48, 9.25, 0.40, [(title, 19.8, True, HEX["navy"])], valign=MSO_ANCHOR.MIDDLE)
    ds.text_box(0.42, 0.87, 9.25, 0.24, [(sub, 9.2, False, HEX["slate_mid"])], valign=MSO_ANCHOR.MIDDLE)

    ds.rect(9.85, 0.22, 3.06, 0.84, pill_bg, pill_edge, lw=1.3, rounding=0.12, radius_adj=0.14, zorder=4)
    ds.text_box(
        9.95, 0.27, 2.86, 0.74,
        [(r_title, 10.0, True, pill_fg), ("\n", 10.0, False, pill_fg), (r_sub, 8.6, False, HEX["slate_mid"])],
        align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, mx=0.04, my=0.02,
    )

    ds.rect(0.42, 7.06, 12.49, 0.015, HEX["border"], rounding=0.0, zorder=4)
    ds.text_box(0.42, 7.11, 8.80, 0.28, [("Mobile-FOG  •  Deteksi Freezing of Gait Pasien Parkinson via Video Smartphone  •  Kelompok 4 ITS", 8.8, False, HEX["muted"])], valign=MSO_ANCHOR.MIDDLE)
    ds.text_box(10.50, 7.11, 2.41, 0.28, [(slide_lbl, 8.8, True, HEX["muted"])], align=PP_ALIGN.RIGHT, valign=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------------------
# Build Slide 8: Real Classification Metrics, Confusion Matrices, ROC/PR
# ---------------------------------------------------------------------------
def build_slide_8(prs: Presentation, font_family: str, y_oof: np.ndarray, loso_models: dict, clean_runs: dict) -> DualSlide:
    ps = prs.slides.add_slide(prs.slide_layouts[6])
    ds = DualSlide(ps, font_family)

    n_win = len(y_oof)
    n_fog = int(y_oof.sum())
    n_non = n_win - n_fog
    lg = loso_models["lgbm"]
    st = loso_models["stgcn"]

    st_fix_pr = clean_runs[("B_15hz", "Light_ST-GCN")]["test_pr_auc"]
    lg_fix_f1 = clean_runs[("B_15hz", "LightGBM+FFT")]["test_f1_pp"]

    draw_chrome(
        ds,
        bar_l=HEX["blue"], bar_r=HEX["green"],
        pill_txt="HASIL METRIK EVALUASI  •  SLIDE 8",
        pill_bg=HEX["blue_bg"], pill_edge=HEX["blue"], pill_fg=HEX["blue_dark"],
        title="Hasil Klasifikasi, Error Rate & Confusion Matrix (LOSO 32-Fold)",
        sub=f"Evaluasi penuh Out-Of-Fold pada 32 subjek bersih ({n_win:,} window: {n_fog:,} FoG / {n_non:,} Non-FoG) — Skenario B_15hz",
        r_title=f"{n_win:,} Window OOF (B_15hz)",
        r_sub="32 Subjek (23 FoG · 9 Non-FoG)",
        slide_lbl="Slide 8",
    )

    # Row 1: Table Card
    ds.rect(0.42, 1.22, 12.49, 1.86, HEX["white"], HEX["border"], lw=1.2, rounding=0.10, radius_adj=0.05, zorder=2)
    ds.text_box(0.58, 1.28, 7.60, 0.24, [("Rekapitulasi Metrik Klasifikasi LOSO 32-Fold (Post-Processed Min-Run 2, B_15hz)", 10.2, True, HEX["navy"])])
    ds.text_box(8.10, 1.29, 4.65, 0.22, [(f"Fixed Split B_15hz Best: ST-GCN PR {st_fix_pr:.4f} · LGBM F1 {lg_fix_f1:.4f}", 8.2, True, HEX["green_dark"])], align=PP_ALIGN.RIGHT)

    headers = ["Model", "Macro Prec", "Macro Rec", "Specificity", "Macro F1", "F1-FoG (PP)", "ROC-AUC", "PR-AUC", "Error Rate"]
    col_ws = [2.05, 1.24, 1.24, 1.24, 1.24, 1.28, 1.26, 1.26, 1.36]

    order_keys = ["lgbm", "stgcn", "mamba", "hybrid"]
    best_max = {
        "macro_prec": max(loso_models[k]["macro_prec"] for k in order_keys),
        "macro_rec": max(loso_models[k]["macro_rec"] for k in order_keys),
        "spec": max(loso_models[k]["spec"] for k in order_keys),
        "macro_f1": max(loso_models[k]["macro_f1"] for k in order_keys),
        "f1_fog": max(loso_models[k]["f1_fog"] for k in order_keys),
        "roc_auc": max(loso_models[k]["roc_auc"] for k in order_keys),
        "pr_auc": max(loso_models[k]["pr_auc"] for k in order_keys),
    }
    best_err = min(loso_models[k]["err_rate"] for k in order_keys)

    row_bgs = [HEX["blue_bg"], HEX["green_bg"], HEX["white"], HEX["row_alt"]]
    table_rows = []
    for k in order_keys:
        m = loso_models[k]
        hi_col = HEX["blue_dark"] if k == "lgbm" else (HEX["green_dark"] if k == "stgcn" else HEX["slate_dark"])
        row = [(m["name"], HEX["navy"], k in ("lgbm", "stgcn"))]
        for metric_k in ["macro_prec", "macro_rec", "spec", "macro_f1", "f1_fog", "roc_auc", "pr_auc"]:
            is_best = abs(m[metric_k] - best_max[metric_k]) < 1e-6
            row.append((f"{m[metric_k]:.4f}", hi_col if is_best else HEX["slate_dark"], is_best))
        is_best_err = abs(m["err_rate"] - best_err) < 1e-6
        row.append((f"{100*m['err_rate']:.2f}%", hi_col if is_best_err else HEX["slate_dark"], is_best_err))
        table_rows.append(row)

    ds.table(0.58, 1.56, 12.17, 1.40, col_ws, headers, table_rows, row_bgs, fsize=8.4, hdr_fsize=8.4)

    # Row 2 Left: Confusion Matrices
    ds.rect(0.42, 3.20, 6.30, 3.06, HEX["white"], HEX["border"], lw=1.2, rounding=0.10, radius_adj=0.04, zorder=2)
    ds.text_box(0.58, 3.27, 5.98, 0.24, [("Confusion Matrix OOF — 2 Model Terbaik (7.766 Window)", 10.4, True, HEX["navy"])])
    ds.text_box(0.58, 3.49, 5.98, 0.19, [("Jumlah window & persentase normalisasi baris aktual (0: Non-FoG, 1: FoG)", 8.0, False, HEX["muted"])])
    ds.picture_fit(CHART_CM_PATH, 0.56, 3.70, 6.02, 2.48, zorder=5)

    # Row 2 Right: ROC & PR Curves
    ds.rect(6.92, 3.20, 5.99, 3.06, HEX["white"], HEX["border"], lw=1.2, rounding=0.10, radius_adj=0.04, zorder=2)
    ds.text_box(7.08, 3.27, 5.67, 0.24, [("Kurva ROC & Precision-Recall Global (32-Fold LOSO)", 10.4, True, HEX["navy"])])
    ds.text_box(7.08, 3.49, 5.67, 0.19, [("Kurva empiris penuh lintas 32 pasien unseen (Skenario B_15hz)", 8.0, False, HEX["muted"])])
    ds.picture_fit(CHART_ROC_PR_PATH, 7.04, 3.70, 5.75, 2.48, zorder=5)

    # Bottom Summary
    ds.rect(0.42, 6.38, 12.49, 0.56, HEX["green_bg"], HEX["green"], lw=1.3, rounding=0.10, radius_adj=0.14, zorder=3)
    ds.text_box(
        0.58, 6.41, 12.17, 0.50,
        [
            ("Temuan Evaluasi Klasifikasi: ", 8.5, True, HEX["green_dark"]),
            (
                f"LightGBM+FFT meraih Error Rate terendah ({100*lg['err_rate']:.2f}% | TN={lg['tn']:,}, FP={lg['fp']:,}) serta Macro F1 terbaik ({lg['macro_f1']:.4f}); "
                f"Light_ST-GCN unggul kalibrasi probabilitas lintas subjek (ROC-AUC {st['roc_auc']:.4f}, PR-AUC {st['pr_auc']:.4f}).",
                8.5, False, HEX["slate_dark"],
            ),
        ],
        valign=MSO_ANCHOR.MIDDLE,
    )
    return ds


# ---------------------------------------------------------------------------
# Build Slide 9: Real Statistical Tests, Spearman rho, Per-Subject LOSO
# ---------------------------------------------------------------------------
def build_slide_9(prs: Presentation, font_family: str, loso_models: dict, per_subj: dict, stat_results: dict) -> DualSlide:
    ps = prs.slides.add_slide(prs.slide_layouts[6])
    ds = DualSlide(ps, font_family)

    sw_s, sw_p = stat_results["shapiro"]
    kw_s, kw_p = stat_results["kruskal"]
    wb_s, wb_p = stat_results["wilcoxon_bbox_raw"]
    wc_s, wc_p = stat_results["wilcoxon_clean_bbox_pr"]
    mw_s, mw_p = stat_results["mannwhitney_stgcn"]
    fr_s, fr_p = stat_results["friedman_f1"]

    r_sl, p_sl = stat_results["spearman_st_lg"]
    r_sm, p_sm = stat_results["spearman_st_mb"]
    r_lm, p_lm = stat_results["spearman_lg_mb"]
    r_pst, p_pst = stat_results["spearman_prev_st"]

    lg_ps = per_subj["LightGBM+FFT"]
    st_ps = per_subj["Light_ST-GCN"]

    draw_chrome(
        ds,
        bar_l=HEX["purple"], bar_r=HEX["amber"],
        pill_txt="UJI STATISTIK & KORELASI NYATA  •  SLIDE 9",
        pill_bg=HEX["purple_bg"], pill_edge=HEX["purple"], pill_fg=HEX["purple_dark"],
        title="Hasil Uji Signifikansi Statistik, Korelasi ρ & Performa Per-Subjek",
        sub="Pengujian statistik nyata pada 16 eksperimen (3 tahap), 32 subjek LOSO (B_15hz), dan konsistensi antar-model",
        r_title=f"p {fmt_p(kw_p)} (Signifikan)",
        r_sub="Shapiro-Wilk · Kruskal-Wallis · Spearman ρ",
        slide_lbl="Slide 9",
    )

    # Left Column: Per-Subject Chart
    ds.rect(0.42, 1.24, 6.35, 4.98, HEX["white"], HEX["border"], lw=1.2, rounding=0.10, radius_adj=0.04, zorder=2)
    ds.text_box(0.58, 1.31, 6.03, 0.24, [("Performa Per-Subjek LOSO (n=32) & Korelasi Prevalensi FoG", 10.4, True, HEX["navy"])])
    ds.text_box(0.58, 1.53, 6.03, 0.19, [("Stratifikasi 9 subjek Non-FoG vs 23 subjek FoG-aktif (≥10% vs <10% window FoG)", 8.0, False, HEX["muted"])])
    ds.picture_fit(CHART_SUBJ_PATH, 0.57, 1.76, 6.05, 4.36, zorder=5)

    # Right Top: Real Statistical Hypothesis Test Table
    ds.rect(6.97, 1.24, 5.94, 2.84, HEX["white"], HEX["border"], lw=1.2, rounding=0.10, radius_adj=0.04, zorder=2)
    ds.text_box(7.13, 1.31, 5.62, 0.24, [("Hasil Uji Signifikansi Statistik Nyata (Data Project)", 10.4, True, HEX["navy"])])
    ds.text_box(7.13, 1.53, 5.62, 0.19, [("Dihitung dengan scipy.stats langsung dari 16 eksperimen & 32 subjek LOSO", 7.9, False, HEX["muted"])])

    st_headers = ["Pengujian", "Uji Statistik", "Stat", "p-value", "Simpulan"]
    st_col_ws = [1.92, 1.16, 0.80, 0.84, 0.90]
    st_rows = [
        [
            ("Normalitas Δ PR (Clean−Raw)", HEX["navy"], False),
            ("Shapiro-Wilk", HEX["slate_dark"], False),
            (f"W={sw_s:.3f}", HEX["slate_dark"], False),
            (fmt_p(sw_p), HEX["slate_dark"], False),
            ("Normal" if sw_p >= 0.05 else "Non-Param", HEX["blue_dark"], True),
        ],
        [
            ("3 Tahap (Raw / BBox / Clean)", HEX["navy"], True),
            ("Kruskal-Wallis", HEX["slate_dark"], False),
            (f"H={kw_s:.2f}", HEX["slate_dark"], False),
            (fmt_p(kw_p), HEX["green_dark"], True),
            ("Signifikan**" if kw_p < 0.01 else "Signifikan*", HEX["green_dark"], True),
        ],
        [
            ("Dampak BBox (vs Raw, n=16)", HEX["navy"], False),
            ("Wilcoxon", HEX["slate_dark"], False),
            (f"W={wb_s:.1f}", HEX["slate_dark"], False),
            (fmt_p(wb_p), HEX["green_dark"], True),
            ("Signifikan**" if wb_p < 0.01 else "Signifikan*", HEX["green_dark"], True),
        ],
        [
            ("Dampak Clean (vs BBox, n=16)", HEX["navy"], False),
            ("Wilcoxon", HEX["slate_dark"], False),
            (f"W={wc_s:.1f}", HEX["slate_dark"], False),
            (fmt_p(wc_p), HEX["green_dark"], True),
            ("Signifikan**" if wc_p < 0.01 else "Signifikan*", HEX["green_dark"], True),
        ],
        [
            ("FoG ≥10% vs <10% (ST-GCN PR)", HEX["navy"], False),
            ("Mann-Whitney U", HEX["slate_dark"], False),
            (f"U={mw_s:.1f}", HEX["slate_dark"], False),
            (fmt_p(mw_p), HEX["purple_dark"], True),
            ("Signifikan**" if mw_p < 0.01 else "Signifikan*", HEX["purple_dark"], True),
        ],
    ]
    st_bgs = [HEX["white"], HEX["green_bg"], HEX["row_alt"], HEX["green_bg"], HEX["purple_bg"]]
    ds.table(7.13, 1.78, 5.62, 2.16, st_col_ws, st_headers, st_rows, st_bgs, fsize=7.7, hdr_fsize=7.9)

    # Right Bottom: Spearman Correlation & Non-FoG Specificity Check
    ds.rect(6.97, 4.22, 5.94, 2.00, HEX["white"], HEX["border"], lw=1.2, rounding=0.10, radius_adj=0.05, zorder=2)
    ds.text_box(7.13, 4.29, 5.62, 0.24, [("Korelasi Spearman (ρ) Antar-Model & Kontrol False Alarm", 10.4, True, HEX["navy"])])

    # Box 1: Spearman & Friedman
    ds.rect(7.13, 4.58, 5.62, 0.80, HEX["purple_bg"], HEX["purple_border"], lw=1.0, rounding=0.08, zorder=3)
    ds.text_box(
        7.25, 4.63, 5.38, 0.70,
        [
            ("Spearman ρ Probabilitas OOF (n=7.766 window, semua p < 0.0001):", 8.2, True, HEX["purple_dark"]),
            ("\n", 8.0, False, HEX["purple_dark"]),
            (f"ST-GCN ↔ LGBM: ρ = {r_sl:.3f}   |   ST-GCN ↔ Bi-Mamba: ρ = {r_sm:.3f}   |   LGBM ↔ Bi-Mamba: ρ = {r_lm:.3f}", 8.1, True, HEX["slate_dark"]),
            ("\n", 8.0, False, HEX["slate_dark"]),
            (f"Uji Friedman antar-4 model (F1-PP pada n=23 subjek FoG): χ² = {fr_s:.2f}, p = {fmt_p(fr_p)}", 8.0, False, HEX["slate_mid"]),
        ],
        valign=MSO_ANCHOR.MIDDLE,
    )

    # Box 2: 9 Non-FoG Patients Specificity
    ds.rect(7.13, 5.46, 5.62, 0.64, HEX["green_bg"], HEX["green_border"], lw=1.0, rounding=0.08, zorder=3)
    ds.text_box(
        7.25, 5.50, 5.38, 0.56,
        [
            ("Kontrol False Alarm pada 9 Subjek Non-FoG (0% FoG): ", 8.2, True, HEX["green_dark"]),
            (
                f"LightGBM+FFT Spec = {100*lg_ps['nonfog_spec_mean']:.2f}% ({lg_ps['nonfog_perfect_cnt']}/9 sempurna 100%); "
                f"Light_ST-GCN Spec = {100*st_ps['nonfog_spec_mean']:.2f}% ({st_ps['nonfog_perfect_cnt']}/9 sempurna 100%).",
                8.1, False, HEX["slate_dark"],
            ),
        ],
        valign=MSO_ANCHOR.MIDDLE,
    )

    # Bottom Summary
    ds.rect(0.42, 6.38, 12.49, 0.56, HEX["purple_bg"], HEX["purple"], lw=1.3, rounding=0.10, radius_adj=0.14, zorder=3)
    ds.text_box(
        0.58, 6.41, 12.17, 0.50,
        [
            ("Simpulan Statistik Nyata: ", 8.5, True, HEX["purple_dark"]),
            (
                f"Kenaikan performa antar 3 tahap terbukti signifikan secara statistik (Kruskal-Wallis H={kw_s:.2f}, {fmt_p_eq(kw_p)}; Wilcoxon BBox & Clean masing-masing {fmt_p_eq(wb_p)} & {fmt_p_eq(wc_p)}). "
                f"PR-AUC per-subjek berkorelasi kuat dengan prevalensi FoG (ρ = {r_pst:.3f}, {fmt_p_eq(p_pst)}).",
                8.4, False, HEX["slate_dark"],
            ),
        ],
        valign=MSO_ANCHOR.MIDDLE,
    )
    return ds


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font_family = configure_fonts()
    print(f"Using primary font: {font_family}")

    y_oof, loso_models, per_subj, stat_results, clean_runs = load_all_metrics()

    build_chart_cm(font_family, loso_models)
    build_chart_roc_pr(font_family, y_oof, loso_models)
    build_chart_persubject(font_family, per_subj, stat_results)

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    ds8 = build_slide_8(prs, font_family, y_oof, loso_models, clean_runs)
    ds9 = build_slide_9(prs, font_family, loso_models, per_subj, stat_results)

    prs.save(PPTX_PATH)
    ds8.fig.savefig(PNG_8_PATH, dpi=200, facecolor=HEX["bg"])
    ds9.fig.savefig(PNG_9_PATH, dpi=200, facecolor=HEX["bg"])
    with PdfPages(PDF_PATH) as pdf:
        pdf.savefig(ds8.fig, facecolor=HEX["bg"])
        pdf.savefig(ds9.fig, facecolor=HEX["bg"])

    plt.close(ds8.fig)
    plt.close(ds9.fig)

    for p in (CHART_CM_PATH, CHART_ROC_PR_PATH, CHART_SUBJ_PATH, PPTX_PATH, PNG_8_PATH, PNG_9_PATH, PDF_PATH):
        print(f"Generated: {p} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
