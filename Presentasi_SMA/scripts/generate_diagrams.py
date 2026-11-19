"""Generate all 8 high-resolution architecture & analysis diagrams for Mobile-FOG (SMA level)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch

from diagram_utils import (
    FONT_MONO,
    FONT_SANS,
    PALETTE,
    draw_arrow,
    draw_box,
    draw_code_tag,
    draw_header,
    draw_op_circle,
    draw_outer_frame,
    draw_poly_arrow,
    draw_projection_cone,
    draw_shape_tag,
    draw_sma_pill,
    draw_tensor_bar,
    new_canvas,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "diagrams"


def _save(fig, filename: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / filename
    fig.savefig(out, dpi=260, facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# 1. Keseluruhan Arsitektur Mobile-FOG (End-to-End)
# ---------------------------------------------------------------------------
def build_01_overall():
    W, H = 16.0, 8.6
    fig, ax = new_canvas(W, H)
    draw_header(
        ax,
        W,
        top_y=8.32,
        title="Arsitektur Keseluruhan Mobile-FOG (End-to-End)",
        subtitle="Dari rekaman video HP biasa (30 fps) menjadi deteksi otomatis 'kaki membeku' (Freezing of Gait)",
        badge="Stack: YOLO26-Pose + 3+1 Classifier",
    )

    draw_outer_frame(
        ax,
        0.25,
        0.95,
        15.50,
        6.50,
        title="Pipeline Deteksi Mobile-FOG",
        subtitle="Alur lengkap dari input video hingga alarm klinis",
        lw=2.0,
    )

    # Col 1: Inputs (x=0.50, w=2.30)
    draw_box(
        ax,
        0.50,
        4.45,
        2.30,
        1.95,
        "Video Kamera HP",
        facecolor=PALETTE["mint_fill"],
        edgecolor=PALETTE["mint_edge"],
        bold=True,
        fontsize=10.5,
        subtext="720p @ 29.98 fps\nPutar 360° (2 menit)\n71 video • 35 pasien",
        subsize=8.3,
    )
    draw_shape_tag(ax, 1.65, 4.22, "[T, 720, 1280, 3]", fontsize=8.4)

    draw_box(
        ax,
        0.50,
        1.65,
        2.30,
        1.95,
        "Sensor IMU (Latih)",
        facecolor=PALETTE["yellow_fill"],
        edgecolor=PALETTE["yellow_edge"],
        linestyle="--",
        bold=True,
        fontsize=10.2,
        subtext="128 Hz • Flag 0 / 1\nKunci jawaban dokter\n(Hanya saat latih & uji)",
        subsize=8.2,
    )
    draw_code_tag(ax, 1.65, 1.42, "SUBxx_y.txt", fontsize=8.5)

    # Col 2: Stage 1 YOLO26-Pose (x=3.30, w=2.45)
    draw_outer_frame(
        ax,
        3.30,
        3.50,
        2.45,
        3.10,
        title="1. Mata AI",
        subtitle="YOLO26n/s-Pose",
        facecolor=PALETTE["blue_fill"],
        edgecolor=PALETTE["blue_edge"],
        lw=1.7,
        title_size=10.5,
    )
    draw_box(
        ax,
        3.52,
        5.12,
        2.00,
        0.72,
        "Backbone + Neck",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["white_edge"],
        fontsize=9.2,
        bold=True,
        subtext="Ekstrak fitur visual",
        subsize=7.8,
    )
    draw_box(
        ax,
        3.52,
        4.22,
        2.00,
        0.74,
        "NMS-Free + RLE Head",
        facecolor=PALETTE["charcoal_fill"],
        edgecolor=PALETTE["charcoal_edge"],
        textcolor="#FFFFFF",
        fontsize=8.9,
        bold=True,
        subtext="Tanpa antrean NMS + σ",
        subsize=7.6,
    )
    draw_tensor_bar(
        ax,
        4.12,
        3.82,
        n_blocks=5,
        cell=0.19,
        depth=2,
        palette="mint",
        label_shape="[T, 17, 3]",
        shape_pos="right",
    )
    draw_code_tag(ax, 4.52, 3.22, "2.9M | 40.3 ms CPU", fontsize=8.3)

    draw_arrow(ax, (2.80, 5.45), (3.30, 5.45), lw=1.8)

    # Col 3: Stage 2 Sync & Preprocess (x=6.25, w=2.55)
    draw_outer_frame(
        ax,
        6.25,
        1.55,
        2.55,
        5.05,
        title="2. Pembersih Data",
        subtitle="Sync & Preprocessing",
        facecolor=PALETTE["peach_fill"],
        edgecolor=PALETTE["peach_edge"],
        lw=1.7,
        title_size=10.5,
    )
    steps_s2 = [
        (5.08, "Align Jam IMU → Video", "128 Hz → 29.98 fps\n(+ buang ±0.5s train)"),
        (4.14, "Interpolasi & Anti-Swap", "Tambal oklusi & cegah\nkaki kiri-kanan tertukar"),
        (3.20, "Norm Skala & v, a", "Bagi tinggi badan;\ntambah (vx, vy, ax, ay)"),
        (2.26, "Sliding Window 2 dtk", "60 frame (overlap 50%)"),
    ]
    for y_box, t_main, t_sub in steps_s2:
        draw_box(
            ax,
            6.47,
            y_box,
            2.12,
            0.76,
            t_main,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["peach_edge"],
            fontsize=8.7,
            bold=True,
            subtext=t_sub,
            subsize=7.4,
            lw=1.3,
        )
    draw_tensor_bar(
        ax,
        7.05,
        1.84,
        n_blocks=5,
        cell=0.18,
        depth=2,
        palette="blue",
        label_shape="[6, 60, 17]",
        shape_pos="right",
    )

    draw_arrow(ax, (5.75, 5.45), (6.25, 5.45), lw=1.8)
    draw_poly_arrow(
        ax, [(2.80, 2.62), (5.98, 2.62), (5.98, 5.25), (6.25, 5.25)], lw=1.6, ls="--", color=PALETTE["yellow_edge"]
    )
    ax.text(4.45, 2.75, "label sinkronisasi (latih)", fontsize=8.0, color="#92400E", ha="center", fontfamily=FONT_SANS)

    # Col 4: Stage 3 Adu 3+1 Otak AI (x=9.35, w=3.35)
    draw_outer_frame(
        ax,
        9.35,
        1.25,
        3.35,
        5.55,
        title="3. Adu 3+1 Otak AI",
        subtitle="Pilih / bandingkan model terbaik",
        facecolor="#F8FAFC",
        edgecolor=PALETTE["gray_edge"],
        ls="--",
        lw=1.7,
        title_size=10.5,
    )
    branches = [
        (5.05, "3A. Light ST-GCN-cls", "Otak Peta Tulang (36K–108K | ~0.60 GB)", PALETTE["mint_fill"], PALETTE["mint_edge"], 1.5),
        (3.90, "3B. LightGBM + FFT", "Equalizer 3–8 Hz + Tree (<1 MB | 0 GB)", PALETTE["yellow_fill"], PALETTE["yellow_edge"], 1.5),
        (2.75, "3C. Pure Bi-Mamba", "Ingatan Selektif Linear (138K | 0.47 GB)", PALETTE["blue_fill"], PALETTE["blue_edge"], 1.5),
        (1.48, "3D. Hybrid GCN + Bi-Mamba", "Peta Tulang + Ingatan (147K | 0.53 GB)", PALETTE["purple_fill"], PALETTE["purple_edge"], 2.2),
    ]
    for yb, btitle, bsub, bfill, bedge, blw in branches:
        bh = 1.04 if "Hybrid" in btitle else 0.94
        draw_box(
            ax,
            9.57,
            yb,
            2.92,
            bh,
            btitle,
            facecolor=bfill,
            edgecolor=bedge,
            fontsize=9.4,
            bold=True,
            subtext=bsub,
            subsize=7.5,
            lw=blw,
        )
    draw_box(
        ax,
        10.75,
        1.32,
        1.65,
        0.26,
        "Kandidat Unggulan",
        facecolor="#6D28D9",
        edgecolor="#5B21B6",
        textcolor="#FFFFFF",
        fontsize=7.2,
        bold=True,
        rounding=0.08,
        lw=1.0,
    )

    bus_x = 9.08
    ax.plot([8.80, bus_x], [3.65, 3.65], color=PALETTE["arrow"], lw=1.8, zorder=6)
    branch_ys = [5.52, 4.37, 3.22, 2.00]
    ax.plot([bus_x, bus_x], [min(branch_ys), max(branch_ys)], color=PALETTE["arrow"], lw=1.8, zorder=6)
    for by in branch_ys:
        draw_arrow(ax, (bus_x, by), (9.57, by), lw=1.6)

    # Col 5: Stage 4 Wasit & Output
    draw_outer_frame(
        ax,
        13.20,
        4.00,
        2.30,
        2.60,
        title="4. Wasit & Uji",
        subtitle="Validasi & Filter",
        facecolor=PALETTE["gray_fill"],
        edgecolor=PALETTE["gray_edge"],
        lw=1.7,
        title_size=10.3,
    )
    draw_box(
        ax,
        13.40,
        5.08,
        1.90,
        0.74,
        "Filter Durasi 10%",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["gray_edge"],
        fontsize=8.7,
        bold=True,
        subtext="Buang kedipan palsu",
        subsize=7.5,
    )
    draw_box(
        ax,
        13.40,
        4.18,
        1.90,
        0.74,
        "Subject Split 23/5/7",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["gray_edge"],
        fontsize=8.6,
        bold=True,
        subtext="+ 35-Fold LOSO adil",
        subsize=7.5,
    )

    col_x = 12.92
    for by in branch_ys:
        ax.plot([12.49, col_x], [by, by], color=PALETTE["arrow"], lw=1.6, zorder=6)
    ax.plot([col_x, col_x], [min(branch_ys), max(branch_ys)], color=PALETTE["arrow"], lw=1.6, zorder=6)
    draw_arrow(ax, (col_x, 5.30), (13.20, 5.30), lw=1.8)

    draw_outer_frame(
        ax,
        13.20,
        1.45,
        2.30,
        2.15,
        title="Output Deteksi",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["frame_border"],
        lw=1.7,
        title_size=10.2,
    )
    draw_box(
        ax,
        13.40,
        2.44,
        1.90,
        0.56,
        "Normal (0.5–3 Hz)",
        facecolor=PALETTE["mint_fill"],
        edgecolor=PALETTE["mint_edge"],
        fontsize=8.6,
        bold=True,
    )
    draw_box(
        ax,
        13.40,
        1.68,
        1.90,
        0.58,
        "FoG / Beku (3–8 Hz)",
        facecolor=PALETTE["red_fill"],
        edgecolor=PALETTE["red_edge"],
        textcolor="#991B1B",
        fontsize=8.6,
        bold=True,
    )
    draw_arrow(ax, (14.35, 4.00), (14.35, 3.60), lw=1.8)

    draw_sma_pill(
        ax,
        0.25,
        0.16,
        15.50,
        0.60,
        "Analogi SMA:  1) Mata AI menggambar manusia lidi   →   2) Data disamakan & dipotong per klip 2 detik   →   3) 4 otak AI diadu secara adil   →   4) Alarm FoG berbunyi bebas kedipan palsu",
        fontsize=9.3,
    )
    _save(fig, "01_arsitektur_keseluruhan_mobilefog.png")


# ---------------------------------------------------------------------------
# 2. Bagian 1: YOLO26-Pose
# ---------------------------------------------------------------------------
def build_02_yolo26():
    W, H = 14.0, 7.8
    fig, ax = new_canvas(W, H)
    draw_header(
        ax,
        W,
        top_y=7.52,
        title="Bagian 1: YOLO26-Pose — 'Mata AI' Penangkap 17 Titik Sendi",
        subtitle="Mengubah video mentah menjadi kerangka manusia lidi super cepat tanpa antrean NMS",
        badge="YOLO26n: 2.9M params | 57.2 mAP",
    )

    draw_outer_frame(
        ax,
        0.30,
        1.00,
        8.45,
        5.80,
        title="Arsitektur YOLO26-Pose (Ultralytics, 2026)",
        subtitle="Single-stage End-to-End Pose Estimator (tanpa modul DFL & NMS)",
        lw=2.0,
    )

    draw_box(
        ax,
        0.55,
        2.85,
        1.55,
        2.20,
        "Frame Video",
        facecolor=PALETTE["mint_fill"],
        edgecolor=PALETTE["mint_edge"],
        bold=True,
        fontsize=9.8,
        subtext="720p diubah ke\n640×640 RGB",
        subsize=8.0,
    )
    draw_tensor_bar(
        ax,
        1.25,
        2.30,
        n_blocks=3,
        cell=0.22,
        depth=3,
        palette="blue",
        label_shape="[640, 640, 3]",
        shape_pos="bottom",
    )

    draw_outer_frame(
        ax,
        2.60,
        2.30,
        2.25,
        3.40,
        title="Backbone + Neck",
        subtitle="CSPDarknet + PAN-FPN",
        facecolor=PALETTE["blue_fill"],
        edgecolor=PALETTE["blue_edge"],
        lw=1.6,
        title_size=10.0,
    )
    for yb, w_bar, lbl in [
        (4.25, 1.55, "Skala Besar (P3)"),
        (3.45, 1.25, "Skala Sedang (P4)"),
        (2.65, 0.95, "Skala Kecil (P5)"),
    ]:
        draw_box(
            ax,
            2.80 + (1.85 - w_bar) / 2,
            yb,
            w_bar,
            0.55,
            lbl,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["blue_edge"],
            fontsize=8.2,
            bold=True,
            lw=1.3,
        )
    draw_box(
        ax,
        2.60,
        1.30,
        2.25,
        0.65,
        "MuSGD Optimizer",
        facecolor=PALETTE["yellow_fill"],
        edgecolor=PALETTE["yellow_edge"],
        fontsize=8.8,
        bold=True,
        subtext="Konvergensi lebih stabil",
        subsize=7.6,
    )

    draw_arrow(ax, (2.10, 3.95), (2.60, 3.95), lw=1.8)

    draw_box(
        ax,
        5.35,
        4.10,
        3.15,
        1.75,
        "1. DFL-Free & NMS-Free Head",
        facecolor=PALETTE["mint_fill"],
        edgecolor=PALETTE["mint_edge"],
        fontsize=9.8,
        bold=True,
        subtext="One-to-One Matching langsung 1 orang = 1 kerangka!\nTidak perlu antrean NMS hapus kotak ganda.",
        subsize=8.0,
    )
    draw_code_tag(ax, 6.92, 3.86, "40.3 ms CPU ONNX (~25 FPS) | 1.8 ms T4 TRT", fontsize=7.9)

    draw_box(
        ax,
        5.35,
        1.65,
        3.15,
        1.90,
        "2. RLE Uncertainty Pose Head",
        facecolor=PALETTE["charcoal_fill"],
        edgecolor=PALETTE["charcoal_edge"],
        textcolor="#FFFFFF",
        fontsize=9.8,
        bold=True,
        subtext="Prediksi posisi (x, y) + tingkat ragu (σx, σy)\nsaat kaki saling menutupi (oklusi putar 360°).",
        subsize=8.0,
    )
    draw_code_tag(ax, 6.92, 1.38, "Residual Log-Likelihood Estimation (RLE)", fontsize=7.9)

    ax.plot([4.85, 5.10], [3.95, 3.95], color=PALETTE["arrow"], lw=1.8)
    ax.plot([5.10, 5.10], [2.60, 4.95], color=PALETTE["arrow"], lw=1.8)
    draw_arrow(ax, (5.10, 4.95), (5.35, 4.95), lw=1.8)
    draw_arrow(ax, (5.10, 2.60), (5.35, 2.60), lw=1.8)

    draw_outer_frame(
        ax,
        9.05,
        1.00,
        4.65,
        5.80,
        title="Output: Manusia Lidi 17 Titik",
        subtitle="Format COCO [T, 17, 3] = (x, y, confidence)",
        lw=2.0,
    )

    draw_arrow(ax, (8.50, 4.95), (9.05, 4.95), lw=1.8)
    draw_arrow(ax, (8.50, 2.60), (9.05, 2.60), lw=1.8)

    halo = Ellipse((10.65, 3.05), width=2.25, height=1.85, facecolor="#FEE2E2", edgecolor="#EF4444", ls="--", lw=1.3, alpha=0.55, zorder=3)
    ax.add_patch(halo)
    ax.text(12.02, 3.05, "Fokus Utama\nDeteksi FoG\n(Lutut & Engkel)", fontsize=8.3, fontweight="bold", color="#991B1B", ha="left", va="center", fontfamily=FONT_SANS)

    pts = {
        "nose": (10.65, 5.42),
        "l_eye": (10.55, 5.51),
        "r_eye": (10.75, 5.51),
        "l_ear": (10.43, 5.45),
        "r_ear": (10.87, 5.45),
        "l_sho": (10.12, 4.88),
        "r_sho": (11.18, 4.88),
        "l_elb": (9.70, 4.34),
        "r_elb": (11.60, 4.34),
        "l_wri": (9.46, 3.88),
        "r_wri": (11.84, 3.88),
        "l_hip": (10.30, 3.65),
        "r_hip": (11.00, 3.65),
        "l_knee": (10.22, 2.95),
        "r_knee": (11.08, 2.95),
        "l_ank": (10.15, 2.28),
        "r_ank": (11.15, 2.28),
    }
    bones = [
        ("l_ear", "l_eye"), ("l_eye", "nose"), ("nose", "r_eye"), ("r_eye", "r_ear"),
        ("l_sho", "r_sho"),
        ("l_sho", "l_elb"), ("l_elb", "l_wri"),
        ("r_sho", "r_elb"), ("r_elb", "r_wri"),
        ("l_sho", "l_hip"), ("r_sho", "r_hip"), ("l_hip", "r_hip"),
        ("l_hip", "l_knee"), ("l_knee", "l_ank"),
        ("r_hip", "r_knee"), ("r_knee", "r_ank"),
    ]
    for a, b in bones:
        xa, ya = pts[a]
        xb, yb = pts[b]
        is_leg = a in ("l_hip", "r_hip", "l_knee", "r_knee") and b in ("l_knee", "r_knee", "l_ank", "r_ank")
        ax.plot([xa, xb], [ya, yb], color="#DC2626" if is_leg else "#64748B", lw=3.0 if is_leg else 2.2, zorder=5)

    for k, (px, py) in pts.items():
        if k in ("nose", "l_eye", "r_eye", "l_ear", "r_ear"):
            c, r = "#60A5FA", 0.055
        elif k in ("l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri"):
            c, r = "#3B82F6", 0.07
        elif k in ("l_hip", "r_hip"):
            c, r = "#F59E0B", 0.085
        else:
            c, r = "#DC2626", 0.095
        ax.add_patch(Circle((px, py), radius=r, facecolor=c, edgecolor="#FFFFFF", lw=1.2, zorder=7))

    ax.text(11.30, 5.45, "0–4: Wajah", fontsize=8.1, color="#1D4ED8", fontweight="bold", va="center")
    ax.text(11.36, 4.92, "5–6: Bahu", fontsize=8.1, color="#1D4ED8", fontweight="bold", va="center")
    ax.text(11.88, 4.22, "7–10: Tangan", fontsize=8.1, color="#1D4ED8", va="center")
    ax.text(10.65, 3.92, "11–12: Pinggul", fontsize=7.9, color="#B45309", fontweight="bold", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.12", fc="#FFFBEB", ec="none", alpha=0.9))
    ax.text(9.55, 2.95, "13–14:\nLutut", fontsize=8.1, color="#991B1B", fontweight="bold", ha="center", va="center")
    ax.text(9.55, 2.28, "15–16:\nEngkel", fontsize=8.1, color="#991B1B", fontweight="bold", ha="center", va="center")

    draw_box(
        ax,
        9.25,
        1.16,
        4.25,
        0.86,
        "Catatan Jujur: 17 Titik COCO sampai Engkel",
        facecolor=PALETTE["yellow_fill"],
        edgecolor=PALETTE["yellow_edge"],
        fontsize=8.5,
        bold=True,
        subtext="Belum ada titik ujung jari kaki/tumit → ayunan & getar\nlutut-engkel jadi fokus utama pemodelan.",
        subsize=7.4,
    )

    draw_sma_pill(
        ax,
        0.30,
        0.16,
        13.40,
        0.60,
        "Analogi SMA:  Seperti filter AR Instagram/TikTok yang langsung menempelkan 17 titik sendi ke tubuh — versi YOLO26 menghapus 'antrean NMS' agar jauh lebih cepat di HP!",
        fontsize=9.3,
    )
    _save(fig, "02_bagian1_yolo26_pose.png")


# ---------------------------------------------------------------------------
# 3. Bagian 2: Penyelarasan Label & Preprocessing
# ---------------------------------------------------------------------------
def build_03_preprocessing():
    W, H = 14.0, 7.8
    fig, ax = new_canvas(W, H)
    draw_header(
        ax,
        W,
        top_y=7.52,
        title="Bagian 2: Penyelarasan Jam & Preprocessing — 'Pembersih Data'",
        subtitle="Menyamakan jam sensor IMU (128 Hz) ke video (30 fps) dan membersihkan gerakan memutar 360°",
        badge="Output Tensor: [B, 6, 60, 17]",
    )

    draw_outer_frame(
        ax,
        0.30,
        4.35,
        9.85,
        2.45,
        title="Jalur Atas: Penyelarasan Label IMU (Hanya Tahap Latih & Evaluasi)",
        facecolor="#FFFDF5",
        edgecolor=PALETTE["yellow_edge"],
        lw=1.7,
        title_size=10.2,
    )
    draw_box(
        ax,
        0.55,
        4.65,
        2.65,
        1.45,
        "File Sensor SUBxx_y.txt",
        facecolor=PALETTE["yellow_fill"],
        edgecolor=PALETTE["yellow_edge"],
        bold=True,
        fontsize=9.3,
        subtext="128 Hz • 15.360 baris (120 dtk)\nKolom: Time [s], Freezing [0/1]\n(4× lebih cepat dari kamera)",
        subsize=7.6,
    )
    draw_box(
        ax,
        3.65,
        4.65,
        2.90,
        1.45,
        "Samakan via Time [s]",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["yellow_edge"],
        bold=True,
        fontsize=9.3,
        subtext="128 Hz → 29.98 fps (~3.600 frame)\nTiap frame video mendapat label\n0 (Normal) atau 1 (FoG)",
        subsize=7.6,
    )
    draw_box(
        ax,
        7.00,
        4.65,
        2.90,
        1.45,
        "Buang Transisi ±0.5 dtk",
        facecolor=PALETTE["red_fill"],
        edgecolor=PALETTE["red_edge"],
        textcolor="#991B1B",
        bold=True,
        fontsize=9.2,
        subtext="HANYA di Data Latih (Train)!\nData Val & Test dibiarkan utuh\nagar ujian realistis.",
        subsize=7.6,
    )
    draw_arrow(ax, (3.20, 5.38), (3.65, 5.38), lw=1.7)
    draw_arrow(ax, (6.55, 5.38), (7.00, 5.38), lw=1.7)

    draw_outer_frame(
        ax,
        0.30,
        1.05,
        9.85,
        2.95,
        title="Jalur Bawah: Pembersihan Koordinat 17 Sendi [T, 17, 3]",
        facecolor="#F6FAFF",
        edgecolor=PALETTE["blue_edge"],
        lw=1.7,
        title_size=10.2,
    )
    pose_steps = [
        (0.52, "1. Tambal Oklusi", "Interpolasi titik yang\nhilang saat badan\nmembelakangi HP", "conf < thr"),
        (2.90, "2. Anti-Swap Kaki", "Cegah engkel kiri &\nkanan tertukar saat\nberputar 360°", "cek sumbu bahu"),
        (5.28, "3. Normalisasi Skala", "Pusatkan ke pinggul;\nbagi tinggi torso\n(adil tinggi badan)", "(p - hip)/H"),
        (7.66, "4. 6 Kanal Gerak", "Posisi (x, y)\n+ Kecepatan (vx, vy)\n+ Percepatan (ax, ay)", "C = 6 kanal"),
    ]
    for i, (xp, pt, ps, pc) in enumerate(pose_steps):
        draw_box(
            ax,
            xp,
            1.48,
            2.08,
            1.78,
            pt,
            facecolor="#FFFFFF",
            edgecolor=PALETTE["blue_edge"],
            bold=True,
            fontsize=8.9,
            subtext=ps,
            subsize=7.5,
        )
        draw_code_tag(ax, xp + 1.04, 1.24, pc, fontsize=7.9)
        if i < len(pose_steps) - 1:
            draw_arrow(ax, (xp + 2.08, 2.37), (xp + 2.38, 2.37), lw=1.7)

    draw_outer_frame(
        ax,
        10.50,
        1.05,
        3.20,
        5.75,
        title="Sliding Window",
        subtitle="Potong klip video berurutan",
        facecolor=PALETTE["mint_fill"],
        edgecolor=PALETTE["mint_edge"],
        lw=2.0,
        title_size=11.0,
    )
    draw_box(
        ax,
        10.75,
        4.75,
        2.70,
        1.20,
        "Jendela 2 Detik",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["mint_edge"],
        bold=True,
        fontsize=10.0,
        subtext="T = 60 frame (@ 30 fps)\nStride 30 frame (overlap 50%)",
        subsize=8.1,
    )
    draw_tensor_bar(ax, 11.95, 3.65, n_blocks=6, cell=0.23, depth=3, palette="blue")
    draw_tensor_bar(ax, 11.95, 3.20, n_blocks=6, cell=0.23, depth=3, palette="mint")
    draw_shape_tag(ax, 12.10, 2.72, "Tensor: [B, 6, 60, 17]", fontsize=9.0, color="#0F172A")
    ax.text(
        12.10,
        2.25,
        "• 6 = (x, y, vx, vy, ax, ay)\n• 60 = 2 detik gerakan\n• 17 = jumlah sendi tubuh",
        fontsize=8.0,
        color="#1E293B",
        ha="center",
        va="center",
        fontfamily=FONT_SANS,
    )
    draw_box(
        ax,
        10.85,
        1.24,
        2.50,
        0.56,
        "Label Klip: y = 0 / 1",
        facecolor=PALETTE["charcoal_fill"],
        edgecolor=PALETTE["charcoal_edge"],
        textcolor="#FFFFFF",
        bold=True,
        fontsize=9.0,
    )

    draw_arrow(ax, (10.15, 5.38), (10.50, 5.38), lw=1.8)
    draw_arrow(ax, (10.15, 2.40), (10.50, 2.40), lw=1.8)

    draw_sma_pill(
        ax,
        0.30,
        0.16,
        13.40,
        0.60,
        "Analogi SMA:  Seperti menyamakan dua stopwatch beda kecepatan (128 Hz vs 30 fps), memastikan kaki kiri-kanan tidak tertukar saat berputar, lalu memotong video jadi klip-klip pendek 2 detik.",
        fontsize=9.1,
    )
    _save(fig, "03_bagian2_preprocessing_sync.png")


# ---------------------------------------------------------------------------
# 4. Bagian 3A: Lightweight ST-GCN-cls
# ---------------------------------------------------------------------------
def build_04_stgcn():
    W, H = 14.0, 7.8
    fig, ax = new_canvas(W, H)
    draw_header(
        ax,
        W,
        top_y=7.52,
        title="Bagian 3A: Lightweight ST-GCN-cls — 'Otak Peta Tulang'",
        subtitle="Membaca hubungan spasial antar sendi (graf tulang) sekaligus irama gerakannya (Yan et al., 2018)",
        badge="36K–108K params | ~0.60 GB VRAM",
    )

    draw_tensor_bar(
        ax,
        1.10,
        4.35,
        n_blocks=4,
        cell=0.24,
        depth=3,
        palette="blue",
        label_code="input_skeleton",
        code_pos="top",
        label_shape="[6, 60, 17]",
        shape_pos="bottom",
    )
    draw_box(
        ax,
        0.45,
        2.35,
        1.55,
        0.82,
        "BatchNorm",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["white_edge"],
        bold=True,
        fontsize=9.5,
        subtext="Ratakan skala data",
        subsize=7.6,
    )
    draw_arrow(ax, (1.18, 3.85), (1.18, 3.17), lw=1.7)

    draw_outer_frame(
        ax,
        2.45,
        1.05,
        7.95,
        5.75,
        title="ST-GCN Block × 3  (Kanal: 32 → 64 → 64)",
        subtitle="Setiap blok membaca posisi tulang (Spasial) lalu alur waktu 9 frame (Temporal)",
        lw=2.0,
    )
    draw_poly_arrow(ax, [(2.00, 2.76), (2.22, 2.76), (2.22, 3.90), (2.45, 3.90)], lw=1.8)
    draw_arrow(ax, (2.45, 3.90), (2.75, 3.90), lw=1.8)

    ax.plot([2.58, 2.58, 9.80, 9.80], [3.90, 5.75, 5.75, 4.18], color="#D97706", lw=1.8, zorder=4)
    draw_box(
        ax,
        5.15,
        5.50,
        2.20,
        0.48,
        "Residual (Conv 1×1)",
        facecolor=PALETTE["yellow_fill"],
        edgecolor=PALETTE["yellow_edge"],
        fontsize=8.6,
        bold=True,
    )
    draw_arrow(ax, (9.80, 4.35), (9.80, 4.12), color="#D97706", lw=1.8)

    draw_outer_frame(
        ax,
        2.75,
        1.45,
        4.15,
        3.80,
        title="1. Spatial Graph Conv",
        subtitle="Membaca peta tetangga sendi (K = 3 partisi)",
        facecolor=PALETTE["blue_fill"],
        edgecolor=PALETTE["blue_edge"],
        lw=1.6,
        title_size=9.8,
    )

    g_x = 3.65
    hip = (g_x, 4.02)
    knee = (g_x, 3.05)
    ank = (g_x, 2.08)
    ax.plot([hip[0], knee[0]], [hip[1], knee[1]], color="#1D4ED8", lw=3.0, zorder=6)
    ax.plot([knee[0], ank[0]], [knee[1], ank[1]], color="#C95B2B", lw=3.0, zorder=6)
    ax.add_patch(Circle(hip, 0.25, facecolor="#93C5FD", edgecolor="#1D4ED8", lw=1.6, zorder=8))
    ax.add_patch(Circle(knee, 0.28, facecolor="#86EFAC", edgecolor="#15803D", lw=2.2, zorder=8))
    ax.add_patch(Circle(ank, 0.25, facecolor="#FDBA74", edgecolor="#C2410C", lw=1.6, zorder=8))
    ax.text(hip[0], hip[1], "Pinggul", fontsize=6.7, fontweight="bold", ha="center", va="center", zorder=9)
    ax.text(knee[0], knee[1], "Lutut\n(Root)", fontsize=6.7, fontweight="bold", ha="center", va="center", zorder=9)
    ax.text(ank[0], ank[1], "Engkel", fontsize=6.7, fontweight="bold", ha="center", va="center", zorder=9)

    draw_box(ax, 4.35, 3.72, 2.35, 0.58, "Partisi 1: Root", facecolor="#DCFCE7", edgecolor="#16A34A", fontsize=8.2, bold=True, subtext="Diri sendi itu sendiri", subsize=7.2, lw=1.2)
    draw_box(ax, 4.35, 2.96, 2.35, 0.58, "Partisi 2: Centripetal", facecolor="#DBEAFE", edgecolor="#2563EB", fontsize=8.2, bold=True, subtext="Menuju pusat (pinggul)", subsize=7.2, lw=1.2)
    draw_box(ax, 4.35, 2.20, 2.35, 0.58, "Partisi 3: Centrifugal", facecolor="#FFEDD5", edgecolor="#EA580C", fontsize=8.2, bold=True, subtext="Menjauh (ke engkel)", subsize=7.2, lw=1.2)
    draw_code_tag(ax, 5.52, 1.72, "A_k ⊗ edge_mask M (bobot tulang)", fontsize=7.9)

    draw_outer_frame(
        ax,
        7.20,
        2.65,
        2.05,
        2.40,
        title="2. Temporal Conv",
        facecolor=PALETTE["peach_fill"],
        edgecolor=PALETTE["peach_edge"],
        lw=1.6,
        title_size=9.5,
    )
    draw_box(
        ax,
        7.38,
        3.62,
        1.68,
        0.82,
        "Depthwise Sep.\nConv2D (1×9)",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["peach_edge"],
        fontsize=8.4,
        bold=True,
    )
    ax.text(8.22, 3.12, "Baca 9 frame berurutan\n(Stride 2× perkecil T)", fontsize=7.6, color=PALETTE["text_dark"], ha="center", va="center")

    draw_arrow(ax, (6.90, 3.90), (7.20, 3.90), lw=1.8)
    draw_op_circle(ax, 9.80, 3.90, symbol="+", r=0.20, edgecolor="#475569")
    draw_arrow(ax, (9.25, 3.90), (9.60, 3.90), lw=1.8)

    draw_box(
        ax,
        7.20,
        1.55,
        2.95,
        0.72,
        "ReLU  +  Dropout (0.25)",
        facecolor=PALETTE["gray_fill"],
        edgecolor=PALETTE["gray_edge"],
        linestyle="--",
        fontsize=8.8,
        bold=True,
        subtext="Hanya aktifkan sinyal penting & cegah hafalan",
        subsize=7.3,
    )
    draw_poly_arrow(ax, [(9.80, 3.70), (9.80, 2.48), (8.68, 2.48), (8.68, 2.27)], lw=1.5)

    draw_poly_arrow(ax, [(10.15, 1.91), (10.65, 1.91), (10.65, 5.25), (10.95, 5.25)], lw=1.8)

    draw_tensor_bar(
        ax,
        11.95,
        5.25,
        n_blocks=5,
        cell=0.22,
        depth=2,
        palette="mint",
        label_shape="[64, 15, 17]",
        shape_pos="bottom",
    )
    draw_box(
        ax,
        10.95,
        3.55,
        2.60,
        0.80,
        "Global Avg Pool (GAP)",
        facecolor=PALETTE["gray_fill"],
        edgecolor=PALETTE["gray_edge"],
        linestyle="--",
        fontsize=9.2,
        bold=True,
        subtext="Rangkum seluruh waktu & sendi → [64]",
        subsize=7.5,
    )
    draw_arrow(ax, (12.25, 4.78), (12.25, 4.35), lw=1.7)

    draw_box(
        ax,
        10.95,
        2.35,
        2.60,
        0.78,
        "Linear Head (64 → 2)",
        facecolor=PALETTE["charcoal_fill"],
        edgecolor=PALETTE["charcoal_edge"],
        textcolor="#FFFFFF",
        fontsize=9.2,
        bold=True,
        subtext="35.991 (DW) / 108.279 (Std) params",
        subsize=7.5,
    )
    draw_arrow(ax, (12.25, 3.55), (12.25, 3.13), lw=1.7)

    draw_box(ax, 10.95, 1.32, 1.22, 0.62, "Normal", facecolor=PALETTE["mint_fill"], edgecolor=PALETTE["mint_edge"], bold=True, fontsize=9.2)
    draw_box(ax, 12.33, 1.32, 1.22, 0.62, "FoG", facecolor=PALETTE["red_fill"], edgecolor=PALETTE["red_edge"], textcolor="#991B1B", bold=True, fontsize=9.2)
    draw_arrow(ax, (11.75, 2.35), (11.56, 1.94), lw=1.6)
    draw_arrow(ax, (12.75, 2.35), (12.94, 1.94), lw=1.6)

    draw_sma_pill(
        ax,
        0.30,
        0.16,
        13.40,
        0.60,
        "Analogi SMA:  Sendi tubuh terhubung seperti peta rute KRL (pinggul–lutut–engkel). ST-GCN membaca bentuk kerangka tiap detik dan ritme gerakannya sekaligus!",
        fontsize=9.3,
    )
    _save(fig, "04_bagian3a_stgcn_cls.png")


# ---------------------------------------------------------------------------
# 5. Bagian 3B: LightGBM + Spektral FFT
# ---------------------------------------------------------------------------
def build_05_lightgbm_fft():
    W, H = 14.0, 7.8
    fig, ax = new_canvas(W, H)
    draw_header(
        ax,
        W,
        top_y=7.52,
        title="Bagian 3B: LightGBM + Spektral FFT — 'Otak Equalizer & Pohon Keputusan'",
        subtitle="Mengubah getaran kaki menjadi spektrum frekuensi (0.5–8 Hz) lalu ditebak oleh pohon keputusan super cepat",
        badge="< 1 MB | 0 GB VRAM (Jalan di CPU!)",
    )

    draw_outer_frame(
        ax,
        0.30,
        1.05,
        3.05,
        5.75,
        title="1. Sinyal Gerak (60 Frame)",
        subtitle="Jejak kecepatan engkel & lutut",
        lw=1.9,
        title_size=10.2,
    )

    ax_n = ax.inset_axes([0.55 / W, 4.05 / H, 2.55 / W, 1.85 / H])
    t = np.linspace(0, 2.0, 120)
    ax_n.plot(t, np.sin(2 * np.pi * 1.4 * t), color="#16A34A", lw=2.0)
    ax_n.set_title("Jalan Normal (Pelan Teratur)", fontsize=8.4, fontweight="bold", color="#166534", pad=3)
    ax_n.set_facecolor("#F0FDF4")
    ax_n.set_xticks([0, 1, 2])
    ax_n.set_xticklabels(["0s", "1s", "2s"], fontsize=7)
    ax_n.set_yticks([])
    for sp in ("top", "right"):
        ax_n.spines[sp].set_visible(False)

    ax_f = ax.inset_axes([0.55 / W, 1.45 / H, 2.55 / W, 1.85 / H])
    ax_f.plot(t, 0.25 * np.sin(2 * np.pi * 1.2 * t) + 0.85 * np.sin(2 * np.pi * 5.8 * t), color="#DC2626", lw=1.8)
    ax_f.set_title("Kaki Beku / FoG (Gemetar Cepat)", fontsize=8.4, fontweight="bold", color="#991B1B", pad=3)
    ax_f.set_facecolor("#FEF2F2")
    ax_f.set_xticks([0, 1, 2])
    ax_f.set_xticklabels(["0s", "1s", "2s"], fontsize=7)
    ax_f.set_yticks([])
    for sp in ("top", "right"):
        ax_f.spines[sp].set_visible(False)

    draw_outer_frame(
        ax,
        3.65,
        1.05,
        5.05,
        5.75,
        title="2. Fitur Equalizer FFT & Kinematika",
        subtitle="Memecah gerak menjadi frekuensi & geometri putaran 360°",
        lw=1.9,
        title_size=10.2,
    )

    ax_fft = ax.inset_axes([4.30 / W, 3.72 / H, 4.10 / W, 2.22 / H])
    freqs = np.linspace(0.1, 9.5, 300)
    normal_p = 0.9 * np.exp(-((freqs - 1.6) / 0.45) ** 2)
    fog_p = 1.05 * np.exp(-((freqs - 5.6) / 0.70) ** 2)
    ax_fft.axvspan(0.5, 3.0, color="#BBF7D0", alpha=0.55)
    ax_fft.axvspan(3.0, 8.0, color="#FECACA", alpha=0.55)
    ax_fft.plot(freqs, normal_p, color="#15803D", lw=2.0)
    ax_fft.plot(freqs, fog_p, color="#DC2626", lw=2.0)
    ax_fft.axvline(3.0, color="#475569", ls="--", lw=1.1)
    ax_fft.text(1.75, 1.18, "0.5–3 Hz\nNormal", fontsize=7.6, fontweight="bold", color="#166534", ha="center", va="top")
    ax_fft.text(5.6, 1.24, "3–8 Hz\nFreeze Band", fontsize=7.6, fontweight="bold", color="#991B1B", ha="center", va="top")
    ax_fft.set_xlim(0, 9.0)
    ax_fft.set_ylim(0, 1.32)
    ax_fft.set_xlabel("Frekuensi (Hz)", fontsize=7.8)
    ax_fft.set_ylabel("Energi", fontsize=7.8, labelpad=3)
    ax_fft.tick_params(labelsize=7.2)
    ax_fft.set_title("Fast Fourier Transform (FFT) — 'Equalizer Kaki'", fontsize=8.6, fontweight="bold", pad=3)
    for sp in ("top", "right"):
        ax_fft.spines[sp].set_visible(False)

    draw_box(
        ax,
        3.90,
        2.74,
        4.55,
        0.50,
        "Freezing Index (FI) = Energi(3–8 Hz) / Energi(0.5–3 Hz)",
        facecolor=PALETTE["yellow_fill"],
        edgecolor=PALETTE["yellow_edge"],
        fontsize=8.1,
        bold=True,
        fontfamily=FONT_MONO,
    )

    kin_cards = [
        (3.90, "Asimetri Kiri–Kanan\n|v_kiri - v_kanan|"),
        (5.45, "Sudut Putar 360°\nKecepatan rotasi bahu"),
        (7.00, "Langkah Menyempit\nJarak antar engkel"),
    ]
    for kx, ktxt in kin_cards:
        draw_box(
            ax,
            kx,
            1.30,
            1.45,
            1.22,
            ktxt,
            facecolor=PALETTE["blue_fill"],
            edgecolor=PALETTE["blue_edge"],
            fontsize=7.8,
            bold=True,
            lw=1.3,
        )

    draw_arrow(ax, (3.35, 4.95), (3.65, 4.95), lw=1.7)
    draw_arrow(ax, (3.35, 2.40), (3.65, 2.40), lw=1.7)

    draw_outer_frame(
        ax,
        9.00,
        1.05,
        4.70,
        5.75,
        title="3. LightGBM (Leaf-Wise GBDT)",
        subtitle="Belajar dari tabel fitur via kuis Ya/Tidak cepat",
        lw=1.9,
        title_size=10.2,
    )
    draw_tensor_bar(ax, 10.40, 5.55, n_blocks=6, cell=0.22, depth=1, palette="blue", label_shape="Vektor Fitur [1, F]", shape_pos="right")
    draw_box(ax, 9.28, 4.82, 2.00, 0.44, "GOSS: Fokus kasus sulit", facecolor=PALETTE["mint_fill"], edgecolor=PALETTE["mint_edge"], fontsize=7.9, bold=True)
    draw_box(ax, 11.42, 4.82, 2.00, 0.44, "EFB: Gabung fitur jarang", facecolor=PALETTE["peach_fill"], edgecolor=PALETTE["peach_edge"], fontsize=7.9, bold=True)

    draw_outer_frame(ax, 9.25, 2.55, 4.20, 2.08, title="Contoh Pohon Keputusan (Leaf-Wise)", facecolor="#FAFAFA", edgecolor="#CBD5E1", lw=1.3, title_size=8.6)
    n_root = (11.35, 4.02)
    n_l = (10.20, 3.42)
    n_r = (12.45, 3.42)
    n_rl = (11.75, 2.86)
    n_rr = (13.05, 2.86)
    for p1, p2, lbl, dx_l in [(n_root, n_l, "Tidak", -0.22), (n_root, n_r, "Ya", 0.22), (n_r, n_rl, "Tidak", -0.22), (n_r, n_rr, "Ya", 0.20)]:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#64748B", lw=1.5, zorder=5)
        ax.text(
            (p1[0] + p2[0]) / 2 + dx_l,
            (p1[1] + p2[1]) / 2 + 0.05,
            lbl,
            fontsize=7.2,
            color="#475569",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.06", fc="#FAFAFA", ec="none", alpha=0.9),
            zorder=6,
        )

    draw_box(ax, n_root[0]-0.75, n_root[1]-0.18, 1.50, 0.36, "FI_engkel > 1.8 ?", facecolor="#FFFFFF", edgecolor="#475569", fontsize=7.7, bold=True, lw=1.2)
    draw_box(ax, n_l[0]-0.60, n_l[1]-0.18, 1.20, 0.36, "Normal (-0.72)", facecolor=PALETTE["mint_fill"], edgecolor=PALETTE["mint_edge"], fontsize=7.4, bold=True, lw=1.2)
    draw_box(ax, n_r[0]-0.70, n_r[1]-0.18, 1.40, 0.36, "Asimetri > 0.4 ?", facecolor="#FFFFFF", edgecolor="#475569", fontsize=7.6, bold=True, lw=1.2)
    draw_box(ax, n_rl[0]-0.52, n_rl[1]-0.17, 1.04, 0.34, "+0.25", facecolor=PALETTE["yellow_fill"], edgecolor=PALETTE["yellow_edge"], fontsize=7.5, bold=True, lw=1.2)
    draw_box(ax, n_rr[0]-0.52, n_rr[1]-0.17, 1.04, 0.34, "FoG! (+0.91)", facecolor=PALETTE["red_fill"], edgecolor=PALETTE["red_edge"], textcolor="#991B1B", fontsize=7.4, bold=True, lw=1.2)

    draw_box(
        ax,
        9.25,
        1.28,
        4.20,
        1.05,
        "Skor Akhir + Penjelasan SHAP untuk Dokter",
        facecolor=PALETTE["charcoal_fill"],
        edgecolor=PALETTE["charcoal_edge"],
        textcolor="#FFFFFF",
        fontsize=8.8,
        bold=True,
        subtext="Contoh laporan: 'Alarm FoG aktif karena getar 5.6 Hz &\nlangkah tiba-tiba menyempit saat berputar'",
        subsize=7.4,
    )

    draw_arrow(ax, (8.70, 3.95), (9.00, 3.95), lw=1.8)

    draw_sma_pill(
        ax,
        0.30,
        0.16,
        13.40,
        0.60,
        "Analogi SMA:  Seperti equalizer musik — jalan normal ada di 'nada bass' lambat (0.5–3 Hz), sedangkan kaki beku bergetar di 'nada tinggi' (3–8 Hz), lalu ditebak kuis pohon keputusan di CPU!",
        fontsize=9.1,
    )
    _save(fig, "05_bagian3b_lightgbm_fft.png")


# ---------------------------------------------------------------------------
# 6. Bagian 3C: Bi-Mamba & Hybrid Spatial-GCN + Bi-Mamba (VisionMamba style)
# ---------------------------------------------------------------------------
def build_06_mamba_hybrid():
    W, H = 15.4, 10.7
    fig, ax = new_canvas(W, H)
    draw_header(
        ax,
        W,
        top_y=10.42,
        title="Bagian 3C: Bi-Mamba & Hybrid Spatial-GCN + Bi-Mamba — 'Otak Ingatan Selektif'",
        subtitle="Terinspirasi blok VisionMamba: membaca sekuens gerakan maju & mundur sambil men-stabilo momen gemetar",
        badge="Pure: 138K (0.47 GB) | Hybrid: 147K (0.53 GB)",
    )

    # LEFT COLUMN: Input skeleton + Hybrid vs Pure branches
    draw_tensor_bar(
        ax,
        1.90,
        9.15,
        n_blocks=5,
        cell=0.22,
        depth=3,
        palette="blue",
        label_code="skeleton_clip",
        code_pos="top",
        label_shape="[6, 60, 17]",
        shape_pos="bottom",
    )

    draw_outer_frame(
        ax,
        0.30,
        5.50,
        3.25,
        2.55,
        title="Opsi A: Hybrid (Disarankan)",
        subtitle="Peta Tulang dulu, baru Mamba",
        facecolor=PALETTE["purple_fill"],
        edgecolor=PALETTE["purple_edge"],
        lw=2.0,
        title_size=9.9,
    )
    draw_box(
        ax,
        0.52,
        6.62,
        2.80,
        0.70,
        "2× Spatial-GCN Ringan",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["purple_edge"],
        bold=True,
        fontsize=8.9,
        subtext="Pahami koneksi sendi per frame",
        subsize=7.6,
    )
    draw_box(
        ax,
        0.52,
        5.70,
        2.80,
        0.70,
        "Joint Pool + Linear",
        facecolor=PALETTE["mint_fill"],
        edgecolor=PALETTE["mint_edge"],
        bold=True,
        fontsize=8.8,
        subtext="[64, 60, 17] → [L=60, d=64]",
        subsize=7.6,
    )
    draw_arrow(ax, (1.90, 8.62), (1.90, 8.05), lw=1.7)
    draw_arrow(ax, (1.92, 6.62), (1.92, 6.40), lw=1.5)

    draw_outer_frame(
        ax,
        0.30,
        3.05,
        3.25,
        2.05,
        title="Opsi B: Pure Bi-Mamba",
        subtitle="Langsung tanpa graf tulang",
        facecolor=PALETTE["blue_fill"],
        edgecolor=PALETTE["blue_edge"],
        ls="--",
        lw=1.7,
        title_size=9.9,
    )
    draw_box(
        ax,
        0.52,
        3.32,
        2.80,
        1.02,
        "Flatten [60, 102] → Linear",
        facecolor="#FFFFFF",
        edgecolor=PALETTE["blue_edge"],
        bold=True,
        fontsize=8.7,
        subtext="17 sendi × 6 kanal digabung langsung\nmenjadi token waktu [60, 64]",
        subsize=7.4,
    )
    draw_poly_arrow(ax, [(0.82, 9.15), (0.14, 9.15), (0.14, 4.08), (0.30, 4.08)], lw=1.5, ls="--")

    draw_box(
        ax,
        0.30,
        1.15,
        3.25,
        1.52,
        "Apa itu Selective SSM?",
        facecolor=PALETTE["charcoal_fill"],
        edgecolor=PALETTE["charcoal_edge"],
        textcolor="#FFFFFF",
        fontsize=9.3,
        bold=True,
        subtext="AI men-stabilo momen getar FoG (3–8 Hz)\ndan melupakan gerakan biasa yang tidak\nrelevan. Kompleksitas linear O(T)!",
        subsize=7.6,
    )

    # Entry arrows to top hidden_states
    draw_poly_arrow(ax, [(3.55, 6.72), (3.86, 6.72), (3.86, 9.34), (7.52, 9.34)], lw=1.8, color=PALETTE["purple_edge"])
    draw_poly_arrow(ax, [(3.55, 4.08), (4.03, 4.08), (4.03, 9.18), (7.52, 9.18)], lw=1.5, ls="--", color=PALETTE["blue_edge"])

    # TOP TENSORS ABOVE OUTER FRAME
    draw_tensor_bar(
        ax,
        8.25,
        9.28,
        n_blocks=3,
        cell=0.25,
        depth=2,
        palette="blue",
        label_code="hidden_states",
        code_pos="top",
        label_shape="[60, 64]",
        shape_pos="right",
    )
    draw_tensor_bar(
        ax,
        13.10,
        9.28,
        n_blocks=3,
        cell=0.25,
        depth=2,
        palette="yellow",
        label_code="residual",
        code_pos="top",
        label_shape="[60, 64]",
        shape_pos="right",
    )

    # MAIN VISIONMAMBA-STYLE BLOCK FRAME (y=1.60 to 8.84 -> h=7.24)
    draw_outer_frame(
        ax,
        4.30,
        1.60,
        10.75,
        7.24,
        title="Bi-Mamba Block (×2)",
        subtitle="L = 60, d_model = 64",
        lw=2.1,
        title_size=12.0,
    )

    # 1. Drop Path (8.16..8.54)
    draw_box(ax, 7.55, 8.16, 1.40, 0.38, "Drop Path", facecolor=PALETTE["mint_fill"], edgecolor=PALETTE["mint_edge"], linestyle="--", fontsize=9.1, bold=True)
    draw_code_tag(ax, 7.42, 8.35, "drop_path", ha="right", fontsize=8.5)
    draw_arrow(ax, (8.25, 9.06), (8.25, 8.54), lw=1.6)

    # 2. Add residual (+) (cy=7.78)
    draw_op_circle(ax, 8.25, 7.78, symbol="+", r=0.16)
    draw_arrow(ax, (8.25, 8.16), (8.25, 7.94), lw=1.6)
    draw_poly_arrow(ax, [(13.10, 9.06), (13.10, 7.78), (8.41, 7.78)], lw=1.5)

    # Residual lane on far right
    res_lane_x = 14.05
    ax.plot([8.25, 8.25, res_lane_x, res_lane_x], [7.62, 7.50, 7.50, 1.42], color="#D97706", lw=1.6, zorder=4)
    draw_arrow(ax, (res_lane_x, 1.56), (res_lane_x, 1.38), color="#D97706", lw=1.6)

    # 3. RMS Norm (7.02..7.40)
    draw_box(ax, 7.45, 7.02, 1.60, 0.38, "RMS Norm", facecolor="#FFFFFF", edgecolor=PALETTE["white_edge"], fontsize=9.3, bold=True)
    draw_arrow(ax, (8.25, 7.62), (8.25, 7.40), lw=1.6)

    # 4. in_proj cone
    b_top = draw_tensor_bar(ax, 8.25, 6.60, n_blocks=3, cell=0.20, depth=1, palette="blue", label_shape="[60, 64]", shape_pos="right")
    draw_arrow(ax, (8.25, 7.02), (8.25, 6.72), lw=1.5)

    b_wide = draw_tensor_bar(ax, 8.25, 5.90, n_blocks=10, cell=0.20, depth=1, palette="blue", label_shape="[60, 128]  (2*d_inner)", shape_pos="right")
    draw_projection_cone(ax, b_top[0], b_top[1], b_top[2], b_wide[0], b_wide[1], b_wide[3], text="Projection", code_label="in_proj")

    ax.text(6.95, 5.64, "[60, 64]", fontsize=8.2, color=PALETTE["text_muted"], ha="center", fontfamily=FONT_MONO)
    ax.text(10.85, 5.64, "[60, 64]  (Gate)", fontsize=8.2, color=PALETTE["text_muted"], ha="center", fontfamily=FONT_MONO)

    # Splits from [60, 128]
    ax.plot([7.65, 7.65], [5.78, 5.48], color=PALETTE["arrow"], lw=1.5)
    draw_poly_arrow(ax, [(7.65, 5.48), (6.24, 5.48), (6.24, 5.26)], lw=1.5)
    draw_poly_arrow(ax, [(7.65, 5.48), (8.82, 5.48), (8.82, 5.16)], lw=1.5)
    draw_poly_arrow(ax, [(8.85, 5.78), (8.85, 5.48), (11.75, 5.48), (11.75, 4.72)], lw=1.5)

    # 5A. Backward Box (y=3.68..5.26)
    draw_outer_frame(ax, 5.24, 3.68, 2.00, 1.58, title="", facecolor=PALETTE["blue_fill"], edgecolor=PALETTE["blue_edge"], lw=1.5, rounding=0.16)
    ax.text(5.12, 4.68, "Backward", fontsize=9.0, fontweight="bold", color="#2563EB", ha="right", va="center")
    draw_box(ax, 5.80, 4.94, 0.88, 0.24, "flip", facecolor="#DBEAFE", edgecolor=PALETTE["blue_edge"], linestyle="--", fontsize=8.0, bold=True)
    draw_box(ax, 5.52, 4.54, 1.44, 0.30, "Convolution", facecolor="#FFFFFF", edgecolor=PALETTE["white_edge"], fontsize=8.3, bold=True)
    draw_code_tag(ax, 5.12, 4.38, "conv1d_b", ha="right", fontsize=7.8)
    draw_box(ax, 5.84, 4.21, 0.80, 0.24, "SiLU", facecolor=PALETTE["gray_fill"], edgecolor=PALETTE["gray_edge"], linestyle="--", fontsize=7.9, bold=True)
    draw_box(ax, 5.52, 3.76, 1.44, 0.38, "Selective SSM", facecolor=PALETTE["charcoal_fill"], edgecolor=PALETTE["charcoal_edge"], textcolor="#FFFFFF", fontsize=8.2, bold=True)
    draw_code_tag(ax, 6.02, 3.55, "out_b", ha="right", fontsize=7.8)

    # 5B. Forward Box (y=3.68..5.16)
    draw_outer_frame(ax, 7.82, 3.68, 2.00, 1.48, title="", facecolor=PALETTE["peach_fill"], edgecolor=PALETTE["peach_edge"], lw=1.5, rounding=0.16)
    ax.text(9.96, 4.68, "Forward", fontsize=9.0, fontweight="bold", color="#C2410C", ha="left", va="center")
    draw_box(ax, 8.10, 4.72, 1.44, 0.30, "Convolution", facecolor="#FFFFFF", edgecolor=PALETTE["white_edge"], fontsize=8.3, bold=True)
    draw_code_tag(ax, 7.72, 4.60, "conv1d", ha="right", fontsize=7.8)
    draw_box(ax, 8.42, 4.36, 0.80, 0.24, "SiLU", facecolor=PALETTE["gray_fill"], edgecolor=PALETTE["gray_edge"], linestyle="--", fontsize=7.9, bold=True)
    draw_box(ax, 8.10, 3.78, 1.44, 0.45, "Selective SSM", facecolor=PALETTE["charcoal_fill"], edgecolor=PALETTE["charcoal_edge"], textcolor="#FFFFFF", fontsize=8.2, bold=True)
    draw_code_tag(ax, 8.58, 3.55, "out", ha="right", fontsize=7.8)

    # Gate SiLU
    draw_box(ax, 11.25, 4.34, 1.00, 0.38, "SiLU", facecolor=PALETTE["gray_fill"], edgecolor=PALETTE["gray_edge"], linestyle="--", fontsize=9.0, bold=True)

    # 6. Multiply & Merge
    draw_op_circle(ax, 6.24, 3.30, symbol="x", r=0.14)
    draw_op_circle(ax, 8.82, 3.30, symbol="x", r=0.14)
    draw_arrow(ax, (6.24, 3.68), (6.24, 3.44), lw=1.5)
    draw_arrow(ax, (8.82, 3.68), (8.82, 3.44), lw=1.5)

    ax.plot([11.75, 11.75], [4.34, 3.30], color=PALETTE["arrow"], lw=1.5)
    draw_arrow(ax, (11.75, 3.30), (8.96, 3.30), lw=1.5)
    draw_poly_arrow(ax, [(11.75, 3.30), (11.75, 3.06), (6.24, 3.06), (6.24, 3.16)], lw=1.5)

    draw_box(ax, 7.12, 2.75, 0.88, 0.25, "flip", facecolor="#DBEAFE", edgecolor=PALETTE["blue_edge"], linestyle="--", fontsize=8.0, bold=True)
    draw_poly_arrow(ax, [(6.38, 3.30), (6.70, 3.30), (6.70, 2.88), (7.12, 2.88)], lw=1.5)
    draw_op_circle(ax, 8.82, 2.88, symbol="+", r=0.14)
    draw_arrow(ax, (8.82, 3.16), (8.82, 3.02), lw=1.5)
    draw_arrow(ax, (8.00, 2.88), (8.68, 2.88), lw=1.5)

    # 7. Output projection out_proj INSIDE frame (y=1.68..2.65)
    b_pre = draw_tensor_bar(ax, 8.82, 2.52, n_blocks=6, cell=0.19, depth=1, palette="blue", label_shape="[60, 64]", shape_pos="right")
    draw_arrow(ax, (8.82, 2.74), (8.82, 2.61), lw=1.4)

    b_post = draw_tensor_bar(ax, 8.82, 1.78, n_blocks=3, cell=0.19, depth=1, palette="blue", label_shape="[60, 64]", shape_pos="right")
    draw_projection_cone(ax, b_pre[0], b_pre[1], b_pre[2], b_post[0], b_post[1], b_post[3], text="Projection", code_label="out_proj")

    # BOTTOM EXIT TENSORS + CLASSIFIER HEAD (y=0.92..1.42)
    draw_tensor_bar(
        ax,
        8.82,
        1.14,
        n_blocks=3,
        cell=0.23,
        depth=2,
        palette="blue",
        label_code="hidden_states",
        code_pos="left",
        label_shape="[60, 64]",
        shape_pos="bottom",
    )
    draw_arrow(ax, (8.82, 1.66), (8.82, 1.38), lw=1.6)

    draw_tensor_bar(
        ax,
        14.05,
        1.14,
        n_blocks=3,
        cell=0.23,
        depth=2,
        palette="yellow",
        label_code="residual",
        code_pos="bottom",
    )

    draw_box(
        ax,
        10.35,
        0.92,
        3.10,
        0.54,
        "Mean Pool → Linear [2] → Normal / FoG",
        facecolor=PALETTE["mint_fill"],
        edgecolor=PALETTE["mint_edge"],
        bold=True,
        fontsize=8.5,
    )
    draw_arrow(ax, (9.70, 1.19), (10.35, 1.19), lw=1.7)

    draw_sma_pill(
        ax,
        0.30,
        0.14,
        14.75,
        0.58,
        "Analogi SMA:  Seperti membaca kalimat dari depan & belakang sekaligus (Bidirectional) sambil men-stabilo kata kunci gemetar (Selective SSM) dan melupakan suara bising yang tak penting!",
        fontsize=9.2,
    )
    _save(fig, "06_bagian3c_mamba_hybrid.png")


# ---------------------------------------------------------------------------
# 7. Bagian 4: Evaluasi Anti-Nyontek & Filter Alarm Palsu
# ---------------------------------------------------------------------------
def build_07_evaluasi_postproc():
    W, H = 14.0, 7.5
    fig, ax = new_canvas(W, H)
    draw_header(
        ax,
        W,
        top_y=7.22,
        title="Bagian 4: Ujian Anti-Nyontek & Filter Alarm Palsu",
        subtitle="Memastikan AI menilai gejala pasien baru (bukan hafalan) dan membuang kedipan noise sepersekian detik",
        badge="Validasi: 23/5/7 Split & 35-Fold LOSO",
    )

    # Panel Left: Subject-level split & LOSO
    draw_outer_frame(
        ax,
        0.30,
        0.98,
        6.55,
        5.55,
        title="A. Ujian Anti-Nyontek (Subject-Level Split)",
        subtitle="Total 35 Pasien (71 Video) dibagi ketat per ORANG, bukan per potongan klip",
        lw=2.0,
        title_size=10.8,
    )

    # 35 patient badges in a 7x5 grid
    grid_x0, grid_y0 = 0.72, 4.95
    dx, dy = 0.84, 0.50
    for idx in range(35):
        r = idx // 7
        c = idx % 7
        px = grid_x0 + c * dx
        py = grid_y0 - r * dy
        if idx < 23:
            fc, ec, tc = "#DBEAFE", "#2563EB", "#1E3A8A"
        elif idx < 28:
            fc, ec, tc = "#FEF3C7", "#D97706", "#92400E"
        else:
            fc, ec, tc = "#DCFCE7", "#16A34A", "#14532D"
        draw_box(ax, px, py, 0.72, 0.38, f"P{idx+1}", facecolor=fc, edgecolor=ec, textcolor=tc, fontsize=8.0, bold=True, rounding=0.09, lw=1.3)

    # Split legend pills
    draw_box(ax, 0.62, 2.34, 1.92, 0.44, "Train: 23 Pasien\n(47 Video)", facecolor="#DBEAFE", edgecolor="#2563EB", textcolor="#1E3A8A", fontsize=7.6, bold=True, lw=1.2)
    draw_box(ax, 2.66, 2.34, 1.85, 0.44, "Val: 5 Pasien\n(10 Video)", facecolor="#FEF3C7", edgecolor="#D97706", textcolor="#92400E", fontsize=7.6, bold=True, lw=1.2)
    draw_box(ax, 4.63, 2.34, 1.92, 0.44, "Test: 7 Pasien\n(14 Video)", facecolor="#DCFCE7", edgecolor="#16A34A", textcolor="#14532D", fontsize=7.6, bold=True, lw=1.2)

    draw_box(
        ax,
        0.55,
        1.16,
        6.05,
        1.00,
        "Aturan Emas (+ 35-Fold LOSO)",
        facecolor=PALETTE["red_fill"],
        edgecolor=PALETTE["red_edge"],
        textcolor="#991B1B",
        fontsize=8.9,
        bold=True,
        subtext="Video pasien yang sama DILARANG muncul di Latih & Uji!\nDi LOSO: 34 pasien melatih model, 1 pasien baru murni untuk menguji.",
        subsize=7.6,
    )

    # Panel Right: Post-processing min duration 10%
    draw_outer_frame(
        ax,
        7.15,
        0.98,
        6.55,
        5.55,
        title="B. Post-Processing: Filter Durasi Min 10%",
        subtitle="Menghilangkan prediksi FoG keceplosan (< 0.3 dtk) agar dokter tidak terima alarm palsu",
        lw=2.0,
        title_size=10.8,
    )

    t = np.linspace(0, 10, 400)
    raw_prob = 0.12 + 0.05 * np.sin(3 * t)
    # 2 brief false spikes
    raw_prob += 0.72 * np.exp(-((t - 1.8) / 0.14) ** 2)
    raw_prob += 0.65 * np.exp(-((t - 8.4) / 0.12) ** 2)
    # 1 genuine sustained FoG episode (4.2s to 6.8s)
    genuine = 0.78 * (1 / (1 + np.exp(-12 * (t - 4.2)))) * (1 / (1 + np.exp(12 * (t - 6.8))))
    raw_prob = np.clip(raw_prob + genuine, 0, 0.98)

    ax_raw = ax.inset_axes([7.92 / W, 4.18 / H, 5.45 / W, 1.50 / H])
    ax_raw.plot(t, raw_prob, color="#DC2626", lw=1.8)
    ax_raw.axhline(0.5, color="#475569", ls="--", lw=1.1)
    ax_raw.fill_between(t, 0.5, raw_prob, where=(raw_prob >= 0.5), color="#FCA5A5", alpha=0.6)
    ax_raw.annotate("Kedipan palsu\n(DIBUANG!)", xy=(1.8, 0.83), xytext=(0.35, 0.84), arrowprops=dict(arrowstyle="->", color="#991B1B"), fontsize=7.4, fontweight="bold", color="#991B1B")
    ax_raw.annotate("Kedipan\npalsu", xy=(8.4, 0.77), xytext=(7.2, 0.84), arrowprops=dict(arrowstyle="->", color="#991B1B"), fontsize=7.4, fontweight="bold", color="#991B1B")
    ax_raw.text(5.5, 0.65, "FoG Asli", fontsize=7.8, fontweight="bold", color="#7F1D1D", ha="center")
    ax_raw.set_ylim(0, 1.1)
    ax_raw.set_xlim(0, 10)
    ax_raw.set_xticks([])
    ax_raw.set_ylabel("Prob", fontsize=7.5, labelpad=3)
    ax_raw.set_title("1. Prediksi Mentah AI (Masih Ada Noise)", fontsize=8.5, fontweight="bold", pad=2)
    for sp in ("top", "right"):
        ax_raw.spines[sp].set_visible(False)

    clean_prob = np.where((t >= 4.2) & (t <= 6.8), 1.0, 0.0)
    ax_cln = ax.inset_axes([7.92 / W, 2.22 / H, 5.45 / W, 1.42 / H])
    ax_cln.axvspan(4.2, 6.8, color="#BBF7D0", alpha=0.55, label="Label Dokter (IMU)")
    ax_cln.step(t, clean_prob, where="mid", color="#15803D", lw=2.2, label="Output Bersih AI")
    ax_cln.text(5.5, 0.50, "Hanya Episode FoG\nNyata yang Lolos!", fontsize=7.5, fontweight="bold", color="#14532D", ha="center", va="center")
    ax_cln.set_ylim(-0.08, 1.18)
    ax_cln.set_xlim(0, 10)
    ax_cln.set_xlabel("Waktu Video (detik)", fontsize=7.6, labelpad=2)
    ax_cln.set_ylabel("Alarm", fontsize=7.5, labelpad=3)
    ax_cln.tick_params(labelsize=7.0)
    ax_cln.set_title("2. Setelah Filter Durasi Min 10% (Bersih & Akurat)", fontsize=8.5, fontweight="bold", pad=2)
    ax_cln.legend(loc="upper right", fontsize=7.0, frameon=True)
    for sp in ("top", "right"):
        ax_cln.spines[sp].set_visible(False)

    draw_box(
        ax,
        7.40,
        1.16,
        6.05,
        0.58,
        "Aturan Kuantil 10%: Episode terlalu singkat otomatis dihapus → Alarm Palsu Turun!",
        facecolor=PALETTE["mint_fill"],
        edgecolor=PALETTE["mint_edge"],
        fontsize=8.2,
        bold=True,
    )

    draw_sma_pill(
        ax,
        0.30,
        0.15,
        13.40,
        0.58,
        "Analogi SMA:  Ujian memakai soal untuk murid baru (bukan murid yang dihafal wajahnya), lalu jawaban AI yang cuma 'keceplosan' sepersekian detik langsung disaring!",
        fontsize=9.2,
    )
    _save(fig, "07_bagian4_evaluasi_postproc.png")


# ---------------------------------------------------------------------------
# 8. Bukti Efisiensi: Parameter & VRAM Latihan
# ---------------------------------------------------------------------------
def build_08_efisiensi_vram():
    W, H = 14.2, 7.5
    fig, ax = new_canvas(W, H)
    draw_header(
        ax,
        W,
        top_y=7.22,
        title="Bukti Efisiensi: Ukuran Model (Params) & Kebutuhan VRAM Latihan",
        subtitle="Perbandingan model literatur yang berat (jutaan parameter) vs kandidat ringan Mobile-FOG (ribuan parameter)",
        badge="FP32, AdamW, Batch=64, T=60",
    )

    # Outer frame Left: Parameter count (Log Scale)
    draw_outer_frame(
        ax,
        0.30,
        0.98,
        6.80,
        5.55,
        title="A. Jumlah Parameter Classifier (Skala Logaritmik)",
        subtitle="Semakin pendek batang = semakin kecil ukuran otak AI (hingga 85× lebih ringkas!)",
        lw=1.9,
        title_size=10.6,
    )

    ax_p = ax.inset_axes([2.35 / W, 2.08 / H, 4.45 / W, 3.55 / H])
    models_p = [
        "Light ST-GCN (DW)",
        "Light ST-GCN (Std)",
        "Pure Bi-Mamba (2L)",
        "Hybrid GCN+Mamba",
        "Pure Bi-Mamba (3L)",
        "Full ST-GCN (Yan)",
        "Siamese ST-GCN (Han)",
        "SkelMamba (Martinel)",
    ]
    params_k = [36.0, 108.3, 137.6, 147.3, 203.0, 3020.0, 4090.0, 6840.0]
    labels_p = ["36K (DW)", "108K", "138K", "147K", "203K", "3.02M", "4.09M", "6.84M"]
    colors_p = [
        "#4ADE80", "#86EFAC", "#60A5FA", "#A78BFA", "#3B82F6",
        "#FCA5A5", "#F87171", "#EF4444",
    ]
    y_pos = np.arange(len(models_p))
    bars = ax_p.barh(y_pos, params_k, color=colors_p, edgecolor="#334155", linewidth=0.9, height=0.62)
    ax_p.set_xscale("log")
    ax_p.set_xlim(15, 70000)
    ax_p.set_xticks([20, 100, 1000, 10000])
    ax_p.set_xticklabels(["20K", "100K", "1M", "10M"], fontsize=8.0)
    ax_p.set_yticks(y_pos)
    ax_p.set_yticklabels(models_p, fontsize=8.3, fontweight="bold")
    ax_p.axhline(4.5, color="#475569", ls="--", lw=1.2)
    ax_p.text(55000, 4.68, "Model Berat ↑", fontsize=7.6, color="#991B1B", ha="right", va="bottom", fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="#FFFFFF", ec="none", alpha=0.85))
    ax_p.text(55000, 4.25, "Mobile-FOG ↓", fontsize=7.6, color="#166534", ha="right", va="top", fontweight="bold", bbox=dict(boxstyle="round,pad=0.1", fc="#FFFFFF", ec="none", alpha=0.85))
    for bar, lbl in zip(bars, labels_p):
        w_val = bar.get_width()
        ax_p.text(w_val * 1.15, bar.get_y() + bar.get_height() / 2, lbl, va="center", ha="left", fontsize=8.0, fontweight="bold", color="#1E293B")
    for sp in ("top", "right"):
        ax_p.spines[sp].set_visible(False)

    draw_box(
        ax,
        0.52,
        1.10,
        6.35,
        0.46,
        "Plus LightGBM + FFT: berbasis pohon keputusan (< 1 MB, tanpa matriks bobot saraf!)",
        facecolor=PALETTE["yellow_fill"],
        edgecolor=PALETTE["yellow_edge"],
        fontsize=8.0,
        bold=True,
    )

    draw_outer_frame(
        ax,
        7.35,
        0.98,
        6.55,
        5.55,
        title="B. Estimasi Memori GPU (VRAM) saat Melatih",
        subtitle="Dengan Batch=64, T=60 frame (FP32 AdamW) — sangat ringan di GPU laptop!",
        lw=1.9,
        title_size=10.6,
    )

    ax_v = ax.inset_axes([9.25 / W, 1.68 / H, 4.30 / W, 3.95 / H])
    models_v = [
        "LightGBM + FFT",
        "Pure Bi-Mamba (2L)",
        "Hybrid GCN+Mamba",
        "Light ST-GCN-cls",
        "SkelMamba Full",
        "Full ST-GCN (Yan)",
    ]
    vrams = [0.00, 0.47, 0.53, 0.60, 1.13, 1.79]
    v_labels = ["0.00 GB (CPU!)", "0.47 GB", "0.53 GB", "0.60 GB", "1.13 GB", "1.79 GB"]
    v_colors = ["#FACC15", "#60A5FA", "#A78BFA", "#4ADE80", "#F87171", "#EF4444"]
    yv = np.arange(len(models_v))
    bars_v = ax_v.barh(yv, vrams, color=v_colors, edgecolor="#334155", linewidth=0.9, height=0.58)
    ax_v.set_xlim(0, 2.35)
    ax_v.set_xlabel("Estimasi VRAM Latihan (GB)", fontsize=8.3, labelpad=3)
    ax_v.set_yticks(yv)
    ax_v.set_yticklabels(models_v, fontsize=8.3, fontweight="bold")
    ax_v.tick_params(labelsize=8.0)
    ax_v.axvline(0.65, color="#15803D", ls="--", lw=1.4, zorder=2)
    ax_v.annotate(
        "Zona Aman GPU\nLaptop / Colab Free\n(< 0.65 GB!)",
        xy=(0.66, 0.55),
        xytext=(1.18, 0.72),
        arrowprops=dict(arrowstyle="->", color="#15803D", lw=1.3),
        fontsize=7.7,
        color="#166534",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="#F0FDF4", ec="#86EFAC", lw=1.0),
    )
    for bar, lbl in zip(bars_v, v_labels):
        w_val = bar.get_width()
        ax_v.text(
            w_val + 0.04,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            va="center",
            ha="left",
            fontsize=8.1,
            fontweight="bold",
            color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.08", fc="#FFFFFF", ec="none", alpha=0.92),
            zorder=5,
        )
    for sp in ("top", "right"):
        ax_v.spines[sp].set_visible(False)

    draw_sma_pill(
        ax,
        0.30,
        0.15,
        13.60,
        0.58,
        "Catatan Efisiensi:  Ekstraktor pose YOLO26n-pose (2.9M params, 40.3 ms CPU ONNX) cukup dijalankan beku (tanpa dilatih ulang) sekali saja dengan ~0.3–0.6 GB VRAM!",
        fontsize=9.1,
    )
    _save(fig, "08_perbandingan_efisiensi_vram.png")


if __name__ == "__main__":
    build_01_overall()
    build_02_yolo26()
    build_03_preprocessing()
    build_04_stgcn()
    build_05_lightgbm_fft()
    build_06_mamba_hybrid()
    build_07_evaluasi_postproc()
    build_08_efisiensi_vram()
    print("All 8 diagrams generated successfully.")
