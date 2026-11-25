# Mobile-FOG: Deteksi Freezing of Gait Pasien Parkinson via Video Smartphone

Pipeline deteksi *Freezing of Gait* (FoG) ringan berbasis video smartphone menggunakan ekstraksi pose **YOLO26n-pose + PatientLock-2Pass** dan perbandingan 4 arsitektur model (`Light_ST-GCN`, `LightGBM+FFT`, `Pure_Bi-Mamba`, `Hybrid_GCN+BiMamba`) pada skenario preprocessing `A/B` (`15 Hz` & `30 Hz`).

---

## Struktur Direktori

```text
.
├── .gitignore
├── README.md
├── pyproject.toml
├── src/mobilefog/                 # Paket inti Python (siap ekspansi refactor)
│   ├── __init__.py
│   ├── paths.py                   # Registry terpusat seluruh path proyek & env overrides
│   └── metrics.py                 # Filter pasca-prediksi & metrik evaluasi (ROC/PR/Macro-F1)
├── tests/
│   └── test_mobilefog_core.py     # Unit test jalur path & metrik evaluasi
├── notebooks/                     # Notebook eksperimen Kaggle (urutan pipeline)
│   ├── 01_extract_yolo26_patientlock.ipynb   # Ekstraksi YOLO26n-pose + PatientLock (35 subjek / 71 video)
│   ├── 02_extract_yolo26_clean32.ipynb       # Ekstraksi versi bersih (32 subjek / 66 video; buang 15,24,27)
│   └── 03_train_4models_loso_kaggle.ipynb    # Training 4 model (Fixed Split 16 run + LOSO)
├── scripts/
│   ├── render_pose_demo_videos.py # Render overlay skeleton 17-keypoint & HUD FoG pada video uji
│   └── slides/
│       ├── build_slide_07.py      # Generator Slide 7 (Dampak Penghapusan Noise)
│       └── build_slides_08_09_metrics.py     # Generator Slide 8 & 9 (Metrik & Uji Statistik Nyata)
├── data/
│   └── metadata/
│       └── PDFEinfo.csv           # Metadata klinis & demografis subjek PDFE
├── results/
│   └── metrics/                   # Ringkasan CSV hasil eksperimen (<20 KB, tracked di Git)
│       ├── baseline_35subj_results_summary.csv
│       ├── bbox_35subj_results_summary.csv
│       ├── clean_32subj_results_summary.csv
│       ├── clean_32subj_loso_oof_B_15hz.csv
│       └── clean_32subj_loso_per_subject_B_15hz.csv
├── docs/
│   └── reports/
│       └── PERBANDINGAN_OUTPUT_VS_BOUNDING_BOX.md  # Analisis komparatif 3 tahap eksperimen
├── references/                    # Literatur & paper acuan PDF
└── outputs/                       # (Gitignored) artifak berat lokal tanpa spasi
    ├── pose_extraction/           # Pose mentah & skenario A/B/D
    ├── training/
    │   ├── bbox_35subj/           # Checkpoint & fold LOSO 35 subjek + PatientLock BBox
    │   └── clean_32subj/          # Checkpoint & fold LOSO 32 subjek bersih
    ├── demo_videos/               # Video demo overlay skeleton
    └── slides/                    # File presentasi siap-import ke Canva (.pptx, .pdf, .png)
```

---

## Ringkasan Hasil Utama (32 Subjek Bersih, `B_15hz`)

- **Fixed Split (`16 run`):**
  - `Light_ST-GCN` (`B_15hz`, `38.8k` params): **`Test PR-AUC = 0.9036`**, **`ROC-AUC = 0.9612`**, **`Balanced Acc = 90.54%`**
  - `LightGBM+FFT` (`B_15hz`, `99` trees): **`Test F1-FoG (PP) = 0.8293`**, **`PR-AUC = 0.8861`**, waktu latih CPU **`1.3 dtk`**
- **LOSO 32-Fold Out-Of-Fold (`7.766` window: `1.442` FoG / `6.324` Non-FoG):**
  - `LightGBM+FFT`: **`Macro F1 = 0.8509`**, **`F1-FoG = 0.7515`**, **`Specificity = 97.22%`**, **`Error Rate = 8.29%`** (`99.03%` Spec pada `9` subjek Non-FoG)
  - `Light_ST-GCN`: **`OOF ROC-AUC = 0.9170`**, **`OOF PR-AUC = 0.7284`** (`Spearman ρ = 0.859` vs prevalensi FoG, `p < 0.0001`)
- Laporan lengkap tersedia di [PERBANDINGAN_OUTPUT_VS_BOUNDING_BOX.md](docs/reports/PERBANDINGAN_OUTPUT_VS_BOUNDING_BOX.md).

---

## Cara Menjalankan Pengujian & Skrip

### 1. Unit Test Inti
```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
*(Jika paket terpasang secara editable via `pip install -e .`, cukup jalankan `python3 -m unittest discover -s tests -v`.)*

### 2. Generator Slide Presentasi (Output ke `outputs/slides/`)
Memerlukan `matplotlib`, `pillow`, `python-pptx`, dan `scipy`:
```bash
PYTHONPATH=src python3 scripts/slides/build_slide_07.py
PYTHONPATH=src python3 scripts/slides/build_slides_08_09_metrics.py
```

### 3. Render Video Demo Skeleton (Output ke `outputs/demo_videos/`)
```bash
PYTHONPATH=src python3 scripts/render_pose_demo_videos.py
```
Path dataset mentah dan folder output dapat diubah melalui environment variable `MOBILEFOG_DATA_DIR` dan `MOBILEFOG_OUTPUTS_DIR`.

---

## Peta Tahap Refactor Berikutnya
Kode inti sudah dimulai di `src/mobilefog/`. Tahap berikutnya dapat memindahkan sel-sel notebook `01`–`03` ke modul teruji berikut:
1. `src/mobilefog/pose/` — Ekstraksi `YOLO26n-pose` & isolasi `PatientLock-2Pass`.
2. `src/mobilefog/preprocessing/` — Interpolasi, filter Hampel, Butterworth 9 Hz, & ekstraksi fitur FFT/kinematik `LB6`.
3. `src/mobilefog/models/` — Definisi `LightweightSTGCN`, `PureBiMamba`, `HybridGCNBiMamba`, dan trainer `LightGBM`.
4. `src/mobilefog/training/` — Runner Fixed Split & LOSO paralel berbasis CLI.
