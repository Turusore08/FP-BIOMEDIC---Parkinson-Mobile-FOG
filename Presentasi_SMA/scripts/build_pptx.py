"""Build the 15-slide SMA-friendly PowerPoint deck Mobile-FOG_Penjelasan_SMA.pptx."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

BASE_DIR = Path(__file__).resolve().parent.parent
DIAG_DIR = BASE_DIR / "diagrams"
OUT_PPTX = BASE_DIR / "Mobile-FOG_Penjelasan_SMA.pptx"

SLIDE_W = 13.333
SLIDE_H = 7.500

FONT_MAIN = "Lato"
FONT_MONO = "DejaVu Sans Mono"

# Color Palette
C_DARK_NAVY = RGBColor(15, 23, 42)
C_SLATE_900 = RGBColor(30, 41, 59)
C_SLATE_700 = RGBColor(51, 65, 85)
C_SLATE_500 = RGBColor(100, 116, 139)
C_LIGHT_BG = RGBColor(248, 250, 252)
C_WHITE = RGBColor(255, 255, 255)
C_BORDER = RGBColor(203, 213, 225)

C_BLUE = RGBColor(37, 99, 235)
C_BLUE_BG = RGBColor(239, 246, 255)
C_BLUE_DARK = RGBColor(30, 58, 138)

C_MINT = RGBColor(22, 163, 74)
C_MINT_BG = RGBColor(240, 253, 244)
C_MINT_DARK = RGBColor(20, 83, 45)

C_AMBER = RGBColor(217, 119, 6)
C_AMBER_BG = RGBColor(255, 251, 235)
C_AMBER_DARK = RGBColor(146, 64, 14)

C_PURPLE = RGBColor(124, 58, 237)
C_PURPLE_BG = RGBColor(245, 243, 255)
C_PURPLE_DARK = RGBColor(91, 33, 182)

C_RED = RGBColor(220, 38, 38)
C_RED_BG = RGBColor(254, 242, 242)
C_RED_DARK = RGBColor(153, 27, 27)


def set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(
    slide,
    left: float,
    top: float,
    w: float,
    h: float,
    fill_color: RGBColor,
    line_color: RGBColor | None = None,
    line_pt: float = 1.2,
    rounded: bool = True,
    radius_adj: float = 0.06,
):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(left), Inches(top), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_color
    if line_color is not None:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(line_pt)
    else:
        shp.line.fill.background()
    if rounded and shp.adjustments:
        shp.adjustments[0] = radius_adj
    return shp


def add_textbox(
    slide,
    left: float,
    top: float,
    w: float,
    h: float,
    text: str = "",
    font_size: float = 13,
    bold: bool = False,
    color: RGBColor = C_SLATE_900,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    font_name: str = FONT_MAIN,
    margin_x: float = 0.06,
    margin_y: float = 0.04,
):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin_x)
    tf.margin_right = Inches(margin_x)
    tf.margin_top = Inches(margin_y)
    tf.margin_bottom = Inches(margin_y)
    if text:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.space_after = Pt(2)
            r = p.add_run()
            r.text = line
            r.font.name = font_name
            r.font.size = Pt(font_size)
            r.font.bold = bold
            r.font.color.rgb = color
    return tb


def add_rich_bullets(
    slide,
    left: float,
    top: float,
    w: float,
    h: float,
    items: list[tuple[str, str]],
    font_size: float = 11.2,
    bullet_char: str = "• ",
    lead_color: RGBColor = C_SLATE_900,
    body_color: RGBColor = C_SLATE_700,
    space_after: float = 5.0,
):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.06)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    for i, (lead, body) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        if lead:
            r1 = p.add_run()
            r1.text = f"{bullet_char}{lead} "
            r1.font.name = FONT_MAIN
            r1.font.size = Pt(font_size)
            r1.font.bold = True
            r1.font.color.rgb = lead_color
        r2 = p.add_run()
        r2.text = body if lead else f"{bullet_char}{body}"
        r2.font.name = FONT_MAIN
        r2.font.size = Pt(font_size)
        r2.font.bold = False
        r2.font.color.rgb = body_color
    return tb


def add_picture_fit(slide, img_path: Path, box_l: float, box_t: float, box_w: float, box_h: float):
    """Place image cleanly centered inside (box_l, box_t, box_w, box_h) preserving exact aspect ratio."""
    with Image.open(img_path) as im:
        px_w, px_h = im.size
    ratio = px_w / px_h
    box_ratio = box_w / box_h
    if ratio >= box_ratio:
        draw_w = box_w
        draw_h = box_w / ratio
    else:
        draw_h = box_h
        draw_w = box_h * ratio
    x = box_l + (box_w - draw_w) / 2.0
    y = box_t + (box_h - draw_h) / 2.0
    return slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), width=Inches(draw_w), height=Inches(draw_h))


def add_chrome(
    slide,
    slide_num: int,
    total_slides: int,
    category: str,
    title: str,
    accent: RGBColor = C_BLUE,
    bg: RGBColor = C_WHITE,
):
    set_bg(slide, bg)
    # Top slim accent bar
    add_rect(slide, 0, 0, SLIDE_W, 0.08, accent, rounded=False)
    # Category pill
    pill_w = max(1.8, len(category) * 0.085 + 0.45)
    pill_bg = {
        C_BLUE: C_BLUE_BG,
        C_MINT: C_MINT_BG,
        C_AMBER: C_AMBER_BG,
        C_PURPLE: C_PURPLE_BG,
        C_RED: C_RED_BG,
    }.get(accent, C_BLUE_BG)
    add_rect(slide, 0.40, 0.20, pill_w, 0.28, pill_bg, line_color=accent, line_pt=1.0, radius_adj=0.35)
    add_textbox(
        slide,
        0.40,
        0.20,
        pill_w,
        0.28,
        category.upper(),
        font_size=8.8,
        bold=True,
        color=accent,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
        margin_x=0.04,
        margin_y=0.01,
    )
    # Main slide title
    add_textbox(slide, 0.40, 0.50, 12.50, 0.48, title, font_size=22.5, bold=True, color=C_DARK_NAVY, valign=MSO_ANCHOR.MIDDLE)
    # Subtle footer divider & caption
    add_rect(slide, 0.40, 7.08, 12.53, 0.015, RGBColor(226, 232, 240), rounded=False)
    add_textbox(
        slide,
        0.40,
        7.12,
        8.50,
        0.28,
        "Mobile-FOG  •  Deteksi Freezing of Gait Pasien Parkinson via Video Smartphone  •  Kelompok 4 ITS",
        font_size=9.2,
        color=C_SLATE_500,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_textbox(
        slide,
        11.20,
        7.12,
        1.73,
        0.28,
        f"Slide {slide_num} / {total_slides}",
        font_size=9.2,
        bold=True,
        color=C_SLATE_500,
        align=PP_ALIGN.RIGHT,
        valign=MSO_ANCHOR.MIDDLE,
    )


def add_diagram_sidebar(
    slide,
    analogy_title: str,
    analogy_text: str,
    bullets: list[tuple[str, str]],
    fact_title: str,
    fact_text: str,
    accent: RGBColor,
    fact_bg: RGBColor,
    fact_dark: RGBColor,
):
    x = 9.95
    w = 3.05

    # Card 1: SMA Analogy
    add_rect(slide, x, 1.10, w, 1.68, C_AMBER_BG, line_color=C_AMBER, line_pt=1.4, radius_adj=0.08)
    add_textbox(slide, x + 0.10, 1.15, w - 0.20, 0.32, analogy_title, font_size=11.3, bold=True, color=C_AMBER_DARK)
    add_textbox(slide, x + 0.10, 1.48, w - 0.20, 1.24, analogy_text, font_size=10.4, color=C_SLATE_900)

    # Card 2: How it works (3 bullets)
    add_rect(slide, x, 2.90, w, 2.60, C_WHITE, line_color=C_BORDER, line_pt=1.2, radius_adj=0.06)
    add_textbox(slide, x + 0.10, 2.96, w - 0.20, 0.30, "Cara Kerja Inti", font_size=11.3, bold=True, color=accent)
    add_rich_bullets(slide, x + 0.08, 3.28, w - 0.16, 2.16, bullets, font_size=9.9, lead_color=C_DARK_NAVY, space_after=4.0)

    # Card 3: Key Numeric Fact
    add_rect(slide, x, 5.62, w, 1.28, fact_bg, line_color=accent, line_pt=1.5, radius_adj=0.08)
    add_textbox(slide, x + 0.10, 5.67, w - 0.20, 0.28, fact_title, font_size=10.2, bold=True, color=fact_dark)
    add_textbox(slide, x + 0.10, 5.96, w - 0.20, 0.88, fact_text, font_size=10.5, bold=True, color=C_DARK_NAVY)


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]
    TOTAL = 15

    # =========================================================================
    # SLIDE 1: COVER (Dark Navy)
    # =========================================================================
    s1 = prs.slides.add_slide(blank)
    set_bg(s1, C_DARK_NAVY)
    add_rect(s1, 0, 0, SLIDE_W, 0.10, RGBColor(56, 189, 248), rounded=False)
    add_rect(s1, 0.70, 0.65, 4.10, 0.38, RGBColor(30, 58, 138), line_color=RGBColor(96, 165, 250), line_pt=1.2, radius_adj=0.35)
    add_textbox(
        s1,
        0.70,
        0.65,
        4.10,
        0.38,
        "PENELITIAN BIOMEDIKA  •  EDISI RAMAH ANAK SMA",
        font_size=10.2,
        bold=True,
        color=RGBColor(191, 219, 254),
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )

    add_textbox(
        s1,
        0.70,
        1.20,
        11.90,
        1.55,
        "Mobile-FOG: AI Ringan di Kamera HP\nuntuk Mendeteksi 'Kaki Membeku' pada Pasien Parkinson",
        font_size=29.5,
        bold=True,
        color=C_WHITE,
    )
    add_textbox(
        s1,
        0.70,
        2.82,
        11.40,
        0.72,
        "Bagaimana mengubah video kamera smartphone biasa menjadi kerangka manusia lidi\nuntuk mendeteksi gejala berbahaya Freezing of Gait (FoG) secara cepat dan hemat memori.",
        font_size=15.5,
        color=RGBColor(203, 213, 225),
    )

    # Mini flow strip on cover
    steps_c1 = [
        ("1. Video HP 30fps", "Cukup rekam jalan 2 menit", RGBColor(34, 197, 94)),
        ("2. YOLO26-Pose", "Gambar 17 titik sendi lidi", RGBColor(59, 130, 246)),
        ("3. Adu 4 Otak AI", "ST-GCN, LightGBM, Bi-Mamba, Hybrid", RGBColor(168, 85, 247)),
        ("4. Alarm FoG", "Lindungi pasien sebelum jatuh", RGBColor(245, 158, 11)),
    ]
    for i, (st, ss, sc) in enumerate(steps_c1):
        cx = 0.70 + i * 3.04
        add_rect(s1, cx, 3.82, 2.82, 1.05, RGBColor(30, 41, 59), line_color=sc, line_pt=1.6, radius_adj=0.10)
        add_textbox(s1, cx + 0.12, 3.92, 2.58, 0.34, st, font_size=13, bold=True, color=sc)
        add_textbox(s1, cx + 0.12, 4.30, 2.58, 0.48, ss, font_size=10.8, color=RGBColor(226, 232, 240))

    # Team card bottom
    add_rect(s1, 0.70, 5.25, 11.93, 1.55, RGBColor(22, 33, 54), line_color=RGBColor(71, 85, 105), line_pt=1.2, radius_adj=0.06)
    add_textbox(s1, 0.90, 5.36, 11.50, 0.34, "KELOMPOK 4 — INSTITUT TEKNOLOGI SEPULUH NOPEMBER (ITS)", font_size=11.5, bold=True, color=RGBColor(125, 211, 252))
    team_members = [
        ("Panji Seno Aji", "5054231023"),
        ("Fa'iz Akbar Hizbullah", "5054241005"),
        ("Sultan Hamdi Jailani Daulay", "5054241013"),
        ("Arya Pratama Rhama Putra", "5054241026"),
    ]
    for idx, (nm, nrp) in enumerate(team_members):
        mx = 0.90 + idx * 2.92
        add_textbox(s1, mx, 5.82, 2.80, 0.42, nm, font_size=12.5, bold=True, color=C_WHITE)
        add_textbox(s1, mx, 6.20, 2.80, 0.32, f"NRP {nrp}", font_size=10.8, color=RGBColor(148, 163, 184), font_name=FONT_MONO)

    # =========================================================================
    # SLIDE 2: MASALAH KLINIS (Analogi SMA)
    # =========================================================================
    s2 = prs.slides.add_slide(blank)
    add_chrome(s2, 2, TOTAL, "1. Masalah Nyata", "Pernah Lihat HP Tiba-Tiba Nge-Lag? Itu yang Terjadi pada Kaki Pasien Parkinson!", C_RED, C_LIGHT_BG)

    # Analogy Story Left Card
    add_rect(s2, 0.40, 1.15, 7.35, 3.05, C_WHITE, line_color=C_BORDER, line_pt=1.3, radius_adj=0.05)
    add_rect(s2, 0.40, 1.15, 0.14, 3.05, C_RED, rounded=False)
    add_textbox(s2, 0.68, 1.25, 6.90, 0.40, "Apa itu 'Freezing of Gait' (FoG)?", font_size=17, bold=True, color=C_RED_DARK)
    add_rich_bullets(
        s2,
        0.66,
        1.72,
        6.90,
        2.35,
        [
            ("Analogi Game / HP Nge-Lag:", "Otak sudah mengirim perintah 'MAJU!', tapi telapak kaki tiba-tiba terasa seperti dilem ke lantai selama beberapa detik."),
            ("Kaki Bergetar di Tempat:", "Alih-alih melangkah normal yang halus, lutut pasien justru gemetar sangat cepat (3–8 kali per detik!) sebelum akhirnya bisa berjalan lagi."),
            ("Sangat Berbahaya Saat Berputar:", "Saat pasien diminta berputar arah 360° di tempat sempit, tubuh atas terus berputar sementara kaki terkunci → sangat mudah jatuh!"),
        ],
        font_size=12.2,
        space_after=8.0,
    )

    # Bottom Comparison Box (Normal vs FoG)
    add_rect(s2, 0.40, 4.38, 3.55, 2.45, C_MINT_BG, line_color=C_MINT, line_pt=1.5, radius_adj=0.06)
    add_textbox(s2, 0.56, 4.48, 3.25, 0.36, "Jalan Normal", font_size=15, bold=True, color=C_MINT_DARK)
    add_textbox(
        s2,
        0.56,
        4.88,
        3.25,
        1.82,
        "• Irama langkah teratur dan pelan.\n• Frekuensi gerak: 0,5 – 3 Hz.\n• Kaki kiri & kanan bergantian seimbang.\n• Seperti nada bass lagu yang tenang.",
        font_size=11.8,
        color=C_SLATE_900,
    )

    add_rect(s2, 4.20, 4.38, 3.55, 2.45, C_RED_BG, line_color=C_RED, line_pt=1.5, radius_adj=0.06)
    add_textbox(s2, 4.36, 4.48, 3.25, 0.36, "Kaki Membeku (FoG)", font_size=15, bold=True, color=C_RED_DARK)
    add_textbox(
        s2,
        4.36,
        4.88,
        3.25,
        1.82,
        "• Kaki terasa terkunci / menempel.\n• Frekuensi gemetar: 3 – 8 Hz!\n• Panjang langkah tiba-tiba menyempit.\n• Paling sering terjadi saat putar 360°.",
        font_size=11.8,
        color=C_SLATE_900,
    )

    # Right Column: 3 Stat Cards
    stat_cards_s2 = [
        ("> 10 Juta Orang", "Penderita Parkinson di dunia saat ini — diperkirakan melonjak > 12 Juta di tahun 2040 (Han 2025; Lubis 2025).", C_BLUE_BG, C_BLUE, C_BLUE_DARK),
        ("60% Pemicu Jatuh", "Dari pasien Parkinson jatuh (45–68% per tahun), mayoritas dipicu episode kaki membeku mendadak (Li 2022).", C_RED_BG, C_RED, C_RED_DARK),
        ("20% → 90% Pasien", "FoG dialami ~20% pasien tahap awal hingga ~90% pasien tahap lanjut — butuh deteksi dini tanpa sensor ribet.", C_AMBER_BG, C_AMBER, C_AMBER_DARK),
    ]
    for i, (big_num, desc, bgc, ec, dc) in enumerate(stat_cards_s2):
        yt = 1.15 + i * 1.95
        add_rect(s2, 8.05, yt, 4.88, 1.78, bgc, line_color=ec, line_pt=1.5, radius_adj=0.07)
        add_textbox(s2, 8.25, yt + 0.14, 4.50, 0.48, big_num, font_size=24, bold=True, color=dc)
        add_textbox(s2, 8.25, yt + 0.68, 4.50, 0.96, desc, font_size=11.5, color=C_SLATE_900)

    # =========================================================================
    # SLIDE 3: URGENSI DI INDONESIA & ASIA TENGGARA
    # =========================================================================
    s3 = prs.slides.add_slide(blank)
    add_chrome(s3, 3, TOTAL, "2. Konteks Indonesia", "Kenapa Solusi Ini Harus Cukup Pakai Kamera HP Biasa?", C_AMBER, C_LIGHT_BG)

    cards_s3 = [
        (
            "0,3 Neurolog",
            "per 100.000 Penduduk",
            "Di Asia Tenggara, jumlah dokter spesialis saraf 20× lebih langka dibanding Pasifik Barat (6 / 100.000). Perawat khusus Parkinson baru ada di 33,3% negara Asia.",
            "Mohamed Ibrahim et al., 2025",
            C_RED_BG,
            C_RED,
            C_RED_DARK,
        ),
        (
            "35,3% Petani",
            "Positif Skrining di Karo",
            "Penelitian di Sumut (102 petani sayur dataran tinggi): 36 orang (35,3%) positif skrining Parkinsonisme! Menyemprot pestisida ≥2×/minggu menaikkan risiko 7,1× (OR = 7,1).",
            "Lubis & Hasanah, 2025",
            C_AMBER_BG,
            C_AMBER,
            C_AMBER_DARK,
        ),
        (
            "70% Telemedicine",
            "& ~876.665 Estimasi Kasus",
            "70% negara Asia sudah memakai layanan konsultasi jarak jauh. Sayangnya, sensor kaki rumah sakit mahal & rumit—padahal hampir semua keluarga punya kamera HP!",
            "Wikandikta 2020; Ibrahim 2025",
            C_BLUE_BG,
            C_BLUE,
            C_BLUE_DARK,
        ),
    ]
    for i, (num, subn, txt, cit, bgc, ec, dc) in enumerate(cards_s3):
        xc = 0.40 + i * 4.25
        add_rect(s3, xc, 1.15, 4.02, 3.88, C_WHITE, line_color=ec, line_pt=1.6, radius_adj=0.06)
        add_rect(s3, xc, 1.15, 4.02, 1.12, bgc, line_color=ec, line_pt=1.2, radius_adj=0.06)
        add_textbox(s3, xc + 0.16, 1.24, 3.70, 0.50, num, font_size=24, bold=True, color=dc)
        add_textbox(s3, xc + 0.16, 1.76, 3.70, 0.36, subn, font_size=12.5, bold=True, color=C_SLATE_900)
        add_textbox(s3, xc + 0.16, 2.40, 3.70, 2.05, txt, font_size=12.0, color=C_SLATE_700)
        add_textbox(s3, xc + 0.16, 4.62, 3.70, 0.32, f"Sumber: {cit}", font_size=9.8, bold=True, color=ec)

    # Bottom Hero Takeaway Banner
    add_rect(s3, 0.40, 5.24, 12.53, 1.60, C_MINT_BG, line_color=C_MINT, line_pt=1.6, radius_adj=0.06)
    add_textbox(s3, 0.62, 5.34, 12.10, 0.38, "Misi Mobile-FOG: Cukup Rekam dari HP di Rumah!", font_size=16, bold=True, color=C_MINT_DARK)
    add_rich_bullets(
        s3,
        0.60,
        5.74,
        12.10,
        1.02,
        [
            ("Tanpa Sensor Mahal:", "Pasien di desa atau rumah cukup direkam kamera smartphone biasa (30 fps) saat melakukan tes berputar 360°."),
            ("AI Sebagai Asisten Awal:", "Membantu dokter & keluarga memantau keparahan FoG dari jarak jauh sebelum pasien mengalami cedera jatuh."),
        ],
        font_size=12.0,
        lead_color=C_MINT_DARK,
        space_after=4.0,
    )

    # =========================================================================
    # SLIDE 4: EKSPERIMEN & DATASET
    # =========================================================================
    s4 = prs.slides.add_slide(blank)
    add_chrome(s4, 4, TOTAL, "3. Data & Eksperimen", "Uji Putar 360° Selama 2 Menit: Cara Kami Melatih AI", C_BLUE, C_LIGHT_BG)

    grid_cards_s4 = [
        ("35 Pasien PD-FoG", "Dataset publik klinis (Figshare 14984667) — pasien Parkinson yang sudah dikonfirmasi dokter sering mengalami FoG.", C_BLUE_BG, C_BLUE, C_BLUE_DARK),
        ("71 Video HP (29,98 fps)", "Resolusi HD 1280×720 berdurasi ~2 menit (~3.600 frame) saat pasien melakukan tes berputar 360° di tempat.", C_MINT_BG, C_MINT, C_MINT_DARK),
        ("Sensor IMU 128 Hz", "Gelang sensor medis di tubuh merekam 15.360 baris data/sesi untuk memberi 'kunci jawaban' detik mana FoG terjadi.", C_AMBER_BG, C_AMBER, C_AMBER_DARK),
        ("Hukum Nyquist Terpenuhi!", "Kamera 30 fps mampu menangkap getaran hingga 15 Hz — jauh cukup untuk menangkap getar FoG (3–8 Hz)!", C_PURPLE_BG, C_PURPLE, C_PURPLE_DARK),
    ]
    for idx, (gt, gd, gbg, ge, gdc) in enumerate(grid_cards_s4):
        col = idx % 2
        row = idx // 2
        gx = 0.40 + col * 6.38
        gy = 1.15 + row * 1.92
        add_rect(s4, gx, gy, 6.15, 1.72, C_WHITE, line_color=ge, line_pt=1.5, radius_adj=0.06)
        add_rect(s4, gx, gy, 0.14, 1.72, ge, rounded=False)
        add_textbox(s4, gx + 0.28, gy + 0.14, 5.70, 0.40, gt, font_size=16, bold=True, color=gdc)
        add_textbox(s4, gx + 0.28, gy + 0.58, 5.70, 1.02, gd, font_size=12.2, color=C_SLATE_700)

    # Bottom Data Split Bar Visualization
    add_rect(s4, 0.40, 5.05, 12.53, 1.80, C_WHITE, line_color=C_BORDER, line_pt=1.4, radius_adj=0.06)
    add_textbox(s4, 0.60, 5.14, 12.10, 0.36, "Pembagian Dataset Ketat per Pasien (Total: 35 Pasien = 71 Video + 71 File IMU)", font_size=14, bold=True, color=C_DARK_NAVY)

    # Stacked horizontal split bars (scaled to width 12.10 in)
    bar_x = 0.60
    bar_w_total = 12.10
    w_train = bar_w_total * (23 / 35)
    w_val = bar_w_total * (5 / 35)
    w_test = bar_w_total * (7 / 35)
    add_rect(s4, bar_x, 5.62, w_train - 0.06, 0.65, C_BLUE_BG, line_color=C_BLUE, line_pt=1.5, radius_adj=0.08)
    add_textbox(s4, bar_x, 5.62, w_train - 0.06, 0.65, "TRAIN: 23 Pasien (47 Video)\nBelajar pola FoG", font_size=11.0, bold=True, color=C_BLUE_DARK, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)

    add_rect(s4, bar_x + w_train, 5.62, w_val - 0.06, 0.65, C_AMBER_BG, line_color=C_AMBER, line_pt=1.5, radius_adj=0.08)
    add_textbox(s4, bar_x + w_train, 5.62, w_val - 0.06, 0.65, "VAL: 5\n(10 Vid)", font_size=10.2, bold=True, color=C_AMBER_DARK, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)

    add_rect(s4, bar_x + w_train + w_val, 5.62, w_test, 0.65, C_MINT_BG, line_color=C_MINT, line_pt=1.5, radius_adj=0.08)
    add_textbox(s4, bar_x + w_train + w_val, 5.62, w_test, 0.65, "TEST: 7 Pasien\n(14 Video)", font_size=10.5, bold=True, color=C_MINT_DARK, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)

    add_textbox(
        s4,
        0.60,
        6.38,
        12.10,
        0.36,
        "Pasien yang sama TIDAK PERNAH muncul di Train sekaligus Test — menjamin AI tidak menghafal wajah atau pakaian pasien!",
        font_size=11.0,
        bold=True,
        color=C_RED_DARK,
        align=PP_ALIGN.CENTER,
    )

    # =========================================================================
    # SLIDE 5: PETA BESAR KESELURUHAN ARSITEKTUR
    # =========================================================================
    s5 = prs.slides.add_slide(blank)
    add_chrome(s5, 5, TOTAL, "4. Peta Besar Sistem", "Keseluruhan Arsitektur Mobile-FOG: Dari Video HP ke Alarm FoG", C_DARK_NAVY, C_WHITE)
    add_picture_fit(s5, DIAG_DIR / "01_arsitektur_keseluruhan_mobilefog.png", 0.35, 1.05, 12.63, 5.32)

    pills_s5 = [
        ("1. Mata AI (YOLO26)", "Ubah video → 17 titik sendi", C_BLUE_BG, C_BLUE_DARK),
        ("2. Pembersih Data", "Sync 128Hz→30fps + 6 kanal", C_AMBER_BG, C_AMBER_DARK),
        ("3. Adu 4 Otak AI", "ST-GCN, LGBM, Mamba, Hybrid", C_PURPLE_BG, C_PURPLE_DARK),
        ("4. Wasit Anti-Nyontek", "Subject Split + Filter 10%", C_MINT_BG, C_MINT_DARK),
    ]
    for i, (pt, ps, pbg, pdc) in enumerate(pills_s5):
        px = 0.40 + i * 3.16
        add_rect(s5, px, 6.40, 3.00, 0.60, pbg, line_color=pdc, line_pt=1.0, radius_adj=0.14)
        add_textbox(s5, px + 0.08, 6.41, 2.84, 0.29, pt, font_size=10.0, bold=True, color=pdc, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin_y=0.01)
        add_textbox(s5, px + 0.08, 6.70, 2.84, 0.28, ps, font_size=9.2, color=C_SLATE_900, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE, margin_y=0.01)

    # =========================================================================
    # SLIDE 6: BAGIAN 1 — YOLO26-Pose
    # =========================================================================
    s6 = prs.slides.add_slide(blank)
    add_chrome(s6, 6, TOTAL, "5. Bagian 1: Mata AI", "YOLO26-Pose: Menggambar Manusia Lidi 17 Titik Sendi Secepat Kilat", C_MINT, C_WHITE)
    add_picture_fit(s6, DIAG_DIR / "02_bagian1_yolo26_pose.png", 0.32, 1.08, 9.48, 5.88)
    add_diagram_sidebar(
        s6,
        "Analogi Filter AR Instagram",
        "Seperti filter TikTok/IG yang langsung menempelkan titik di wajah & tubuh, tetapi YOLO26 dirancang khusus agar ringan & cepat untuk HP!",
        [
            ("Tanpa Antrean NMS:", "Model lama perlu menyaring kotak kembar berkali-kali. YOLO26 memakai One-to-One matching (NMS-Free)!"),
            ("Tahu Kapan Ragu (RLE):", "Saat pasien membelakangi kamera (putaran 360°), RLE memberi skor keraguan (σx, σy)."),
            ("Fokus Lutut & Engkel:", "Dari 17 titik COCO, gerak kaki bawah menjadi sumber sinyal terpenting FoG."),
        ],
        "KECEPATAN & UKURAN",
        "YOLO26n-pose: 2,9 Juta params\n40,3 ms di CPU (~25 FPS) | 1,8 ms GPU T4",
        C_MINT,
        C_MINT_BG,
        C_MINT_DARK,
    )

    # =========================================================================
    # SLIDE 7: BAGIAN 2 — Preprocessing & Sync
    # =========================================================================
    s7 = prs.slides.add_slide(blank)
    add_chrome(s7, 7, TOTAL, "6. Bagian 2: Pembersih Data", "Sinkronisasi Sensor 128 Hz & Mencegah Kaki Tertukar Saat Berputar 360°", C_AMBER, C_WHITE)
    add_picture_fit(s7, DIAG_DIR / "03_bagian2_preprocessing_sync.png", 0.32, 1.08, 9.48, 5.88)
    add_diagram_sidebar(
        s7,
        "Analogi Menyamakan Stopwatch",
        "Sensor IMU mencatat 128× per detik, kamera hanya 30× per detik. Kita samakan waktunya dulu agar setiap foto punya label FoG yang tepat!",
        [
            ("Jurus Anti-Swap:", "Saat badan membelakangi kamera, AI bisa bingung mana engkel kiri vs kanan — dicek konsistensinya."),
            ("Adil Tinggi Badan:", "Koordinat dibagi tinggi badan (bahu–pinggul) agar anak kecil maupun orang tinggi dinilai sama."),
            ("Tambah Kecepatan:", "AI tidak hanya diberi posisi (x,y), tapi juga kecepatan (v) & percepatan (a) → 6 kanal!"),
        ],
        "OUTPUT SIAP LATIH",
        "Klip 2 detik = 60 frame\nTensor Kubus [B, 6, 60, 17]",
        C_AMBER,
        C_AMBER_BG,
        C_AMBER_DARK,
    )

    # =========================================================================
    # SLIDE 8: BAGIAN 3A — Lightweight ST-GCN-cls
    # =========================================================================
    s8 = prs.slides.add_slide(blank)
    add_chrome(s8, 8, TOTAL, "7. Bagian 3A: Otak Peta Tulang", "Lightweight ST-GCN-cls: AI yang Paham Peta Jalur KRL Antar Sendi Tubuh", C_MINT, C_WHITE)
    add_picture_fit(s8, DIAG_DIR / "04_bagian3a_stgcn_cls.png", 0.32, 1.08, 9.48, 5.88)
    add_diagram_sidebar(
        s8,
        "Analogi Peta Jalur KRL",
        "Sendi tubuh tidak bergerak acak! Pinggul terhubung ke lutut, lutut ke engkel — persis stasiun KRL yang saling terhubung oleh rel tulang.",
        [
            ("Graf Spasial (K=3):", "Setiap sendi membaca dirinya sendiri (Root), arah ke pusat tubuh, dan arah ke ujung kaki."),
            ("Edge Mask Belajar:", "AI otomatis memberi bobot besar ke tulang kaki jika itu paling menentukan gejala FoG."),
            ("Konvolusi Waktu 1×9:", "Membaca ritme gerakan 9 frame berurutan lalu dirangkum oleh Global Avg Pool."),
        ],
        "SUPER RINGKAS!",
        "Hanya 36K (DW) – 108K (Std) params\nButuh ~0,60 GB VRAM saja saat dilatih!",
        C_MINT,
        C_MINT_BG,
        C_MINT_DARK,
    )

    # =========================================================================
    # SLIDE 9: BAGIAN 3B — LightGBM + FFT
    # =========================================================================
    s9 = prs.slides.add_slide(blank)
    add_chrome(s9, 9, TOTAL, "8. Bagian 3B: Otak Equalizer", "LightGBM + Spektral FFT: Mendengar Frekuensi Gemetar Kaki (3–8 Hz)", C_AMBER, C_WHITE)
    add_picture_fit(s9, DIAG_DIR / "05_bagian3b_lightgbm_fft.png", 0.32, 1.08, 9.48, 5.88)
    add_diagram_sidebar(
        s9,
        "Analogi Equalizer Musik",
        "Jalan normal terdengar seperti nada bass pelan (0,5–3 Hz). Saat FoG terjadi, equalizer langsung menyala tinggi di nada cepat (3–8 Hz)!",
        [
            ("Freezing Index (FI):", "Rasio energi getaran cepat (3–8 Hz) dibagi energi jalan normal (0,5–3 Hz)."),
            ("Pohon Keputusan Cepat:", "LightGBM menumbuhkan pohon secara Leaf-Wise dibantu GOSS & EFB — sangat hemat memori!"),
            ("Bisa Dijelaskan (SHAP):", "Dokter bisa membaca alasan AI, misal: 'Alarm aktif karena FI engkel tinggi'."),
        ],
        "NOL VRAM GPU!",
        "Ukuran model < 1 MB\n0,00 GB VRAM — jalan ngebut di CPU biasa!",
        C_AMBER,
        C_AMBER_BG,
        C_AMBER_DARK,
    )

    # =========================================================================
    # SLIDE 10: BAGIAN 3C — Bi-Mamba & Hybrid Spatial-GCN + Bi-Mamba
    # =========================================================================
    s10 = prs.slides.add_slide(blank)
    add_chrome(s10, 10, TOTAL, "9. Bagian 3C: Otak Ingatan Selektif", "Bi-Mamba & Hybrid GCN + Mamba: Membaca Gerakan Maju-Mundur Seperti Video", C_PURPLE, C_WHITE)
    add_picture_fit(s10, DIAG_DIR / "06_bagian3c_mamba_hybrid.png", 0.32, 1.08, 9.48, 5.88)
    add_diagram_sidebar(
        s10,
        "Analogi Stabilo Buku Maju-Mundur",
        "Seperti membaca cerita dari awal & dari akhir sekaligus sambil men-stabilo kalimat penting (Selective SSM) dan melupakan yang tak perlu!",
        [
            ("Tetap Butuh YOLO26-Pose:", "Mamba bukan pencari tulang dari gambar mentah—YOLO26 jadi 'mata', Mamba jadi 'otak sekuens'!"),
            ("Kompleksitas Linear O(T):", "Berbeda dengan Transformer yang berat (kuadratik), Mamba tetap ringan meski klip panjang."),
            ("Kenapa Varian Hybrid?", "Spatial-GCN membaca anatomi tulang dulu, lalu Bi-Mamba menyisir pola waktu maju-mundur!"),
        ],
        "HEMAT MEMORI!",
        "Pure Bi-Mamba: 138K params (~0,47 GB)\nHybrid GCN+Mamba: 147K params (~0,53 GB)",
        C_PURPLE,
        C_PURPLE_BG,
        C_PURPLE_DARK,
    )

    # =========================================================================
    # SLIDE 11: TABEL ADU 3+1 OTAK AI
    # =========================================================================
    s11 = prs.slides.add_slide(blank)
    add_chrome(s11, 11, TOTAL, "10. Perbandingan 4 Kandidat", "Pertarungan 3+1 Otak AI: Mana yang Paling Cocok untuk Mobile-FOG?", C_PURPLE, C_LIGHT_BG)

    headers_s11 = ["Kandidat Model", "Analogi SMA", "Ukuran Model", "VRAM Latih", "Kelebihan Utama", "Tantangan"]
    rows_s11 = [
        ("3A. Light ST-GCN-cls", "Peta Jalur KRL Sendi", "36K–108K params\n(~0,14–0,41 MB)", "~0,60 GB", "Paling paham hubungan fisik tulang pinggul–lutut–engkel", "Butuh graf tulang tetap; agak sensitif jika ada titik hilang"),
        ("3B. LightGBM + FFT", "Equalizer Musik +\nKuis Pohon", "< 1 MB\n(100–300 pohon)", "0,00 GB\n(CPU!)", "Tercepat di CPU & transparan bagi dokter (bisa pakai SHAP)", "Bergantung pada fitur buatan (FFT & asimetri)"),
        ("3C. Pure Bi-Mamba", "Baca Cerita Maju-Mundur\n+ Stabilo Selektif", "138K–203K params\n(~0,52–0,77 MB)", "~0,47 GB", "Menangkap konteks waktu panjang secara linear O(T)", "Tanpa graf tulang, hubungan antar sendi perlu dipelajari ulang"),
        ("3D. Hybrid GCN +\nBi-Mamba (Unggulan)", "Peta KRL + Baca\nMaju-Mundur Sekaligus", "147K params\n(~0,56 MB)", "~0,53 GB", "Gabungkan kelebihan terbaik: paham anatomi + ingat pola panjang!", "Arsitektur dua tahap sedikit lebih kompleks diimplementasikan"),
    ]
    col_ws = [2.15, 2.25, 1.75, 1.35, 2.75, 2.28]
    tbl_shape = s11.shapes.add_table(5, 6, Inches(0.40), Inches(1.15), Inches(sum(col_ws)), Inches(4.55))
    tbl = tbl_shape.table
    for ci, cw in enumerate(col_ws):
        tbl.columns[ci].width = Inches(cw)

    # Header row
    for ci, htxt in enumerate(headers_s11):
        cell = tbl.cell(0, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_DARK_NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if ci in (2, 3) else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = htxt
        r.font.name = FONT_MAIN
        r.font.size = Pt(11.5)
        r.font.bold = True
        r.font.color.rgb = C_WHITE

    row_bgs = [C_MINT_BG, C_AMBER_BG, C_BLUE_BG, C_PURPLE_BG]
    for ri, rdata in enumerate(rows_s11):
        for ci, val in enumerate(rdata):
            cell = tbl.cell(ri + 1, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_bgs[ri] if ci == 0 else (C_WHITE if ri < 3 else RGBColor(250, 245, 255))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if ci in (2, 3) else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.name = FONT_MAIN
            r.font.size = Pt(10.8)
            r.font.bold = (ci in (0, 3))
            r.font.color.rgb = C_PURPLE_DARK if (ri == 3 and ci == 0) else C_SLATE_900

    # Takeaway bottom box
    add_rect(s11, 0.40, 5.88, 12.53, 1.02, C_BLUE_BG, line_color=C_BLUE, line_pt=1.5, radius_adj=0.08)
    add_textbox(s11, 0.60, 5.96, 12.10, 0.32, "Kenapa Keempat Model Ini Diadu Secara Ilmiah?", font_size=13.5, bold=True, color=C_BLUE_DARK)
    add_textbox(
        s11,
        0.60,
        6.30,
        12.10,
        0.52,
        "Setiap arsitektur punya gaya belajar berbeda (graf tulang vs spektrum frekuensi vs memori sekuens vs hybrid). Menguji semuanya di split yang sama memberi bukti paling objektif!",
        font_size=11.8,
        color=C_SLATE_900,
    )

    # =========================================================================
    # SLIDE 12: BAGIAN 4 — Validasi & Post-Processing
    # =========================================================================
    s12 = prs.slides.add_slide(blank)
    add_chrome(s12, 12, TOTAL, "11. Bagian 4: Ujian Adil", "Aturan Ujian Anti-Nyontek & Filter Penyaring Kedipan Palsu", C_RED, C_WHITE)
    add_picture_fit(s12, DIAG_DIR / "07_bagian4_evaluasi_postproc.png", 0.32, 1.08, 9.48, 5.88)
    add_diagram_sidebar(
        s12,
        "Analogi Ujian Murid Baru",
        "Jika AI dilatih dan diuji memakai video pasien yang sama, nilainya bisa tinggi cuma karena hafal baju/lantai rumah pasien! Itu curang!",
        [
            ("Subject-Level Split:", "23 pasien melatih, 5 validasi, 7 pasien asing murni untuk ujian akhir."),
            ("35-Fold LOSO:", "Uji 35 putaran: selalu sisakan 1 pasien baru yang belum pernah dilihat AI sama sekali!"),
            ("Filter Durasi 10%:", "Alarm sesaat yang lebih pendek dari kuantil durasi 10% langsung dibuang sebagai noise."),
        ],
        "TUJUAN KLINIS",
        "AI belajar ciri gerak FoG murni — bukan menghafal pencahayaan atau baju pasien!",
        C_RED,
        C_RED_BG,
        C_RED_DARK,
    )

    # =========================================================================
    # SLIDE 13: BUKTI EFISIENSI PARAMS & VRAM
    # =========================================================================
    s13 = prs.slides.add_slide(blank)
    add_chrome(s13, 13, TOTAL, "12. Bukti Efisiensi", "Seberapa Ringan? Bisa Dilatih di GPU Laptop Biasa (< 0,60 GB VRAM!)", C_BLUE, C_WHITE)
    add_picture_fit(s13, DIAG_DIR / "08_perbandingan_efisiensi_vram.png", 0.32, 1.08, 9.48, 5.88)
    add_diagram_sidebar(
        s13,
        "Analogi Ransel Sekolah",
        "Model riset klasik membawa 'tas ransel besi' seberat 3–6,8 juta parameter. Mobile-FOG hanya membawa kantong ringan puluhan–ratusan ribu parameter!",
        [
            ("15–85× Lebih Kecil:", "Light ST-GCN (36K) & Hybrid GCN+Mamba (147K) jauh lebih ringkas dari Full ST-GCN (3,02M) & SkelMamba (6,84M)."),
            ("YOLO26n Beku:", "Ekstraktor pose 2,9M params cukup dijalankan sekali (inference) pakai ~0,3–0,6 GB VRAM."),
            ("Ramah Laptop & Edge:", "Keempat classifier butuh 0 – 0,60 GB VRAM saat latih (aman di Colab gratisan!)."),
        ],
        "KESIAPAN RISET",
        "Semua kandidat sangat feasible untuk eksperimen & deployment ke perangkat seluler!",
        C_BLUE,
        C_BLUE_BG,
        C_BLUE_DARK,
    )

    # =========================================================================
    # SLIDE 14: ROADMAP & CATATAN JUJUR ILMIAH
    # =========================================================================
    s14 = prs.slides.add_slide(blank)
    add_chrome(s14, 14, TOTAL, "13. Masa Depan & Etika", "Dari Proyek Kampus Menuju Aplikasi Pendamping Pasien di Rumah", C_MINT, C_LIGHT_BG)

    roadmap = [
        ("Tahap 1", "Ekstraksi Pose Batch", "Jalankan YOLO26n/s-pose pada 71 video (35 pasien) & selaraskan dengan label IMU 128 Hz.", C_BLUE_BG, C_BLUE, C_BLUE_DARK),
        ("Tahap 2", "Adu 4 Model & LOSO", "Latih ST-GCN, LightGBM, Bi-Mamba, dan Hybrid pada Subject Split + 35-Fold LOSO.", C_PURPLE_BG, C_PURPLE, C_PURPLE_DARK),
        ("Tahap 3", "Ekspor Ringan ONNX", "Kompresi bobot model terbaik agar berjalan mulus real-time di CPU HP / aplikasi web.", C_AMBER_BG, C_AMBER, C_AMBER_DARK),
        ("Tahap 4", "Dampak Telemedicine", "Membantu skrining awal pasien di daerah minim dokter saraf sebelum cedera jatuh terjadi.", C_MINT_BG, C_MINT, C_MINT_DARK),
    ]
    for i, (rstep, rtitle, rdesc, rbg, re, rdc) in enumerate(roadmap):
        rx = 0.40 + i * 3.16
        add_rect(s14, rx, 1.18, 2.98, 2.55, C_WHITE, line_color=re, line_pt=1.5, radius_adj=0.06)
        add_rect(s14, rx + 0.16, 1.34, 1.10, 0.34, rbg, line_color=re, line_pt=1.0, radius_adj=0.30)
        add_textbox(s14, rx + 0.16, 1.34, 1.10, 0.34, rstep, font_size=10.0, bold=True, color=rdc, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        add_textbox(s14, rx + 0.16, 1.80, 2.66, 0.42, rtitle, font_size=14.0, bold=True, color=C_DARK_NAVY)
        add_textbox(s14, rx + 0.16, 2.28, 2.66, 1.30, rdesc, font_size=11.4, color=C_SLATE_700)

    # Bottom Honest Scientific Limitations + Impact
    add_rect(s14, 0.40, 3.98, 6.12, 2.86, C_MINT_BG, line_color=C_MINT, line_pt=1.5, radius_adj=0.06)
    add_textbox(s14, 0.60, 4.10, 5.72, 0.38, "Dampak Positif Bagi Pasien & Keluarga", font_size=15, bold=True, color=C_MINT_DARK)
    add_rich_bullets(
        s14,
        0.58,
        4.52,
        5.76,
        2.20,
        [
            ("Nyaman & Tanpa Ribet:", "Pasien tidak perlu dipasang banyak kabel atau gelang sensor mahal untuk pemantauan rutin."),
            ("Deteksi Sebelum Terlambat:", "Mencatat frekuensi & durasi episode membeku untuk dievaluasi dokter saat kontrol obat."),
            ("Relevan untuk Indonesia:", "Bisa menjangkau daerah seperti Karo atau pelosok yang jauh dari poli saraf."),
        ],
        font_size=11.6,
        lead_color=C_MINT_DARK,
        space_after=6.0,
    )

    add_rect(s14, 6.80, 3.98, 6.13, 2.86, C_AMBER_BG, line_color=C_AMBER, line_pt=1.5, radius_adj=0.06)
    add_textbox(s14, 7.00, 4.10, 5.73, 0.38, "Batasan Ilmiah yang Jujur (Penting Dipahami!)", font_size=15, bold=True, color=C_AMBER_DARK)
    add_rich_bullets(
        s14,
        6.98,
        4.52,
        5.77,
        2.20,
        [
            ("Asisten, Bukan Dokter:", "AI ini alat bantu skrining — diagnosis medis dan resep obat tetap wewenang dokter spesialis saraf."),
            ("Batas 17 Titik COCO:", "Pose standar berhenti di pergelangan kaki (engkel); gerak jari kaki halus hanya ditangkap tak langsung."),
            ("Kualitas Video:", "Tetap butuh pencahayaan cukup dan posisi kamera yang memperlihatkan kaki penuh."),
        ],
        font_size=11.6,
        lead_color=C_AMBER_DARK,
        space_after=6.0,
    )

    # =========================================================================
    # SLIDE 15: PENUTUP (Dark Navy)
    # =========================================================================
    s15 = prs.slides.add_slide(blank)
    set_bg(s15, C_DARK_NAVY)
    add_rect(s15, 0, 0, SLIDE_W, 0.10, RGBColor(74, 222, 128), rounded=False)

    add_textbox(s15, 0.70, 0.70, 11.90, 0.80, "Terima Kasih! Ada Pertanyaan?", font_size=34, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    add_textbox(
        s15,
        0.70,
        1.52,
        11.90,
        0.48,
        "3 Hal Penting untuk Diingat dari Mobile-FOG:",
        font_size=16.5,
        color=RGBColor(148, 163, 184),
        align=PP_ALIGN.CENTER,
    )

    takeaways = [
        ("1. Masalah Nyata", "'Kaki membeku' (FoG) memicu 60% jatuh pada Parkinson, sementara dokter saraf di Asia Tenggara masih langka (0,3 / 100.000).", RGBColor(248, 113, 113)),
        ("2. Mata + Otak AI Ringan", "YOLO26-Pose mengubah video HP jadi 17 titik sendi, lalu dianalisis oleh ST-GCN / LightGBM / Bi-Mamba (< 0,6 GB VRAM!).", RGBColor(96, 165, 250)),
        ("3. Uji Adil Anti-Nyontek", "Diuji ketat per pasien (23/5/7 & 35-Fold LOSO) agar siap membantu pemantauan pasien di dunia nyata.", RGBColor(74, 222, 128)),
    ]
    for i, (tk_t, tk_d, tk_c) in enumerate(takeaways):
        tx = 0.65 + i * 4.10
        add_rect(s15, tx, 2.30, 3.85, 2.65, RGBColor(30, 41, 59), line_color=tk_c, line_pt=1.8, radius_adj=0.08)
        add_textbox(s15, tx + 0.18, 2.48, 3.50, 0.44, tk_t, font_size=16.5, bold=True, color=tk_c)
        add_textbox(s15, tx + 0.18, 3.04, 3.50, 1.75, tk_d, font_size=12.8, color=RGBColor(226, 232, 240))

    add_rect(s15, 0.65, 5.30, 12.05, 1.48, RGBColor(22, 33, 54), line_color=RGBColor(71, 85, 105), line_pt=1.2, radius_adj=0.06)
    add_textbox(
        s15,
        0.85,
        5.42,
        11.65,
        0.34,
        "Kelompok 4 — Institut Teknologi Sepuluh Nopember (ITS)",
        font_size=12.5,
        bold=True,
        color=RGBColor(125, 211, 252),
        align=PP_ALIGN.CENTER,
    )
    add_textbox(
        s15,
        0.85,
        5.82,
        11.65,
        0.34,
        "Panji Seno Aji (5054231023)  •  Fa'iz Akbar Hizbullah (5054241005)  •  Sultan Hamdi Jailani Daulay (5054241013)  •  Arya Pratama Rhama Putra (5054241026)",
        font_size=10.5,
        color=C_WHITE,
        align=PP_ALIGN.CENTER,
    )
    add_textbox(
        s15,
        0.85,
        6.22,
        11.65,
        0.42,
        "Referensi Utama: Li et al. (2022)  •  Han et al. (2025)  •  Mohamed Ibrahim et al. (2025)  •  Lubis & Hasanah (2025)  •  Yan et al. (2018)  •  Ke et al. (2017)  •  Gu & Dao (2023)  •  Martinel et al. (2024)  •  Jocher et al. (2026 YOLO26)",
        font_size=8.8,
        color=RGBColor(148, 163, 184),
        align=PP_ALIGN.CENTER,
    )

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PPTX)
    print(f"Saved presentation: {OUT_PPTX} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build_deck()
