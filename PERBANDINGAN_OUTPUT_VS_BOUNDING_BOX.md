# Perbandingan Hasil Training: Tanpa Bounding Box (`output training`) vs Dengan Bounding Box Pasien (`output bounding box`)

Dokumen ini membandingkan hasil eksperimen **Fixed Split Resmi (`23` Train / `5` Val / `7` Test)** antara:
- **Baseline (Tanpa PatientLock BBox)**: [output training/mobilefog_training_results/results_summary.csv](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20training/mobilefog_training_results/results_summary.csv)
- **Dengan `PatientLock-2Pass` BBox (`ajikajadeh/mobile-fog-patient-bboxes`)**: [output bounding box/mobilefog_training_results/results_summary.csv](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20bounding%20box/mobilefog_training_results/results_summary.csv)

---

## 1. Kesimpulan Utama: Apakah Bounding Box Membantu Deteksi Parkinson (FoG)?

**Ya, sangat membantu secara signifikan.** Penggunaan *bounding box* khusus pasien (`PatientLock-2Pass` + *crop* + *masking* pengawas) memberikan lonjakan performa besar di hampir seluruh model dan skenario:

| Metrik Terbaik (Test Set) | Tanpa BBox (`output training`) | Dengan BBox (`output bounding box`) | Peningkatan ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Test PR-AUC Tertinggi** | `0.6631` (`Pure_Bi-Mamba`, `B_15hz`) | **`0.8366`** (`Light_ST-GCN`, `B_15hz`) | **`+0.1735`** (`+26,2%` relatif) |
| **Test ROC-AUC Tertinggi** | `0.8715` (`LightGBM+FFT`, `B_30hz`) | **`0.9460`** (`LightGBM+FFT`, `B_30hz`) | **`+0.0745`** |
| **Test F1-FoG (PP) Tertinggi** | `0.6390` (`LightGBM+FFT`, `B_15hz`) | **`0.7529`** (`LightGBM+FFT`, `B_15hz`) | **`+0.1139`** |
| **Test Balanced Acc Tertinggi** | `0.7815` (`LightGBM+FFT`, `B_30hz`) | **`0.8522`** (`LightGBM+FFT`, `A_15hz`) | **`+0.0707`** |
| **Val PR-AUC Tertinggi** | `0.5782` (`LightGBM+FFT`, `B_15hz`) | **`0.7292`** (`Pure_Bi-Mamba`, `B_15hz`) | **`+0.1510`** |

---

## 2. Perbandingan Per-Model (Skenario Terbaik Masing-Masing)

| Model | Params | Tanpa BBox (`PR-AUC` / `ROC-AUC` / `F1-PP`) | Dengan BBox (`PR-AUC` / `ROC-AUC` / `F1-PP`) | Dampak BBox ($\Delta$ `PR-AUC`) |
| :--- | :---: | :---: | :---: | :---: |
| **`Light_ST-GCN`** | `38,801` | `0.5756` / `0.8442` / `0.6053` (`A_15hz`) | **`0.8366`** / `0.9379` / `0.7474` (`B_15hz`) | **`+0.2610`** |
| **`LightGBM+FFT`** | `54–100` trees | `0.6294` / `0.8602` / `0.5866` (`A_15hz`) | **`0.8365`** / **`0.9441`** / **`0.7529`** (`B_15hz`) | **`+0.2071`** |
| **`Hybrid_GCN+BiMamba`** | `216,345` | `0.6065` / `0.7514` / `0.5340` (`A_15hz`) | **`0.6699`** / `0.8479` / `0.5809` (`B_15hz`) | **`+0.0634`** |
| **`Pure_Bi-Mamba`** | `211,521` | `0.6631` / `0.7689` / `0.6124` (`B_15hz`) | **`0.6802`** / `0.7966` / `0.6130` (`A_15hz`) | **`+0.0171`** |

---

## 3. Tabel Lengkap 16 Eksperimen (`4 Skenario × 4 Model`)

| Skenario | Model | PR-AUC (Tanpa → Dengan BBox) | $\Delta$ PR-AUC | ROC-AUC (Tanpa → Dengan BBox) | F1-PP (Tanpa → Dengan BBox) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `B_15hz` | **`Light_ST-GCN`** | `0.5730 → 0.8366` | **`+0.2636`** | `0.8053 → 0.9379` | `0.5615 → 0.7474` |
| `B_15hz` | **`LightGBM+FFT`** | `0.6066 → 0.8365` | **`+0.2299`** | `0.8672 → 0.9441` | `0.6390 → 0.7529` |
| `B_15hz` | **`Hybrid_GCN+BiMamba`** | `0.4849 → 0.6699` | **`+0.1850`** | `0.7159 → 0.8479` | `0.4969 → 0.5809` |
| `B_15hz` | `Pure_Bi-Mamba` | `0.6631 → 0.6056` | `-0.0575` | `0.7689 → 0.7673` | `0.6124 → 0.5517` |
| `A_15hz` | **`Light_ST-GCN`** | `0.5756 → 0.8250` | **`+0.2494`** | `0.8442 → 0.9309` | `0.6053 → 0.7278` |
| `A_15hz` | **`LightGBM+FFT`** | `0.6294 → 0.8258` | **`+0.1964`** | `0.8602 → 0.9368` | `0.5866 → 0.7373` |
| `A_15hz` | `Pure_Bi-Mamba` | `0.6385 → 0.6802` | `+0.0417` | `0.7659 → 0.7966` | `0.5710 → 0.6130` |
| `A_15hz` | `Hybrid_GCN+BiMamba` | `0.6065 → 0.5772` | `-0.0293` | `0.7514 → 0.7287` | `0.5340 → 0.5258` |
| `B_30hz` | **`Light_ST-GCN`** | `0.3790 → 0.7611` | **`+0.3821`** | `0.6919 → 0.9095` | `0.3926 → 0.6690` |
| `B_30hz` | **`LightGBM+FFT`** | `0.6137 → 0.8314` | **`+0.2177`** | `0.8715 → 0.9460` | `0.5925 → 0.7348` |
| `B_30hz` | `Pure_Bi-Mamba` | `0.5042 → 0.6331` | `+0.1289` | `0.7112 → 0.7863` | `0.5424 → 0.5842` |
| `B_30hz` | `Hybrid_GCN+BiMamba` | `0.5453 → 0.6243` | `+0.0790` | `0.7003 → 0.7528` | `0.4753 → 0.5540` |
| `A_30hz` | **`Light_ST-GCN`** | `0.4383 → 0.7273` | **`+0.2890`** | `0.7072 → 0.9108` | `0.3700 → 0.6626` |
| `A_30hz` | **`LightGBM+FFT`** | `0.5676 → 0.7873` | **`+0.2197`** | `0.8527 → 0.9285` | `0.5812 → 0.6960` |
| `A_30hz` | **`Hybrid_GCN+BiMamba`** | `0.2388 → 0.5642` | **`+0.3254`** | `0.4517 → 0.7018` | `0.2703 → 0.5213` |
| `A_30hz` | `Pure_Bi-Mamba` | `0.5352 → 0.5315` | `-0.0037` | `0.6841 → 0.6696` | `0.5801 → 0.5133` |

---

## 4. Analisis Mengapa Bounding Box Sangat Berpengaruh

1. **Menghilangkan *ID Switching* ke Pengawas (`97,19%` frame multi-person):**
   Tanpa *bounding box* pasien, skeleton `YOLO26n-pose` sesekali melompat ke pengawas klinis yang berdiri diam di samping pasien. Lompatan posisi mendadak ini menghasilkan kecepatan (`dx, dy`) dan energi spektral palsu yang mengacaukan sinyal FoG `3–8 Hz`.
2. **`Light_ST-GCN` & `LightGBM+FFT` Paling Diuntungkan:**
   - Pada `Light_ST-GCN` (`B_15hz`), `PR-AUC` melonjak **`+0.2636`** (`0.5730 → 0.8366`) karena graf tulang `LB6` (`Hip–Knee–Ankle`) kini murni merekam kinematika kaki pasien.
   - Pada `LightGBM+FFT` (`B_15hz`), `ROC-AUC` mencapai **`0.9441`** (`PR-AUC = 0.8365`, `F1-PP = 0.7529`) hanya dengan `100` pohon keputusan dan `0 GB` VRAM.
3. **Skenario `B_15hz` (`Hampel` + `15 Hz`) Menjadi Pemenang Paling Konsisten:**
   *Downsampling* ke `15 Hz` (`T = 30` frame) mengurangi *noise* antar-frame sekaligus membuat `Light_ST-GCN` dan `LightGBM+FFT` lebih mudah menangkap pola getar `3–7,5 Hz`.

---

## 5. Catatan Status LOSO
Kedua folder (`output training/` dan `output bounding box/`) sama-sama memiliki folder `loso_35fold_B_15hz/`, namun belum menghasilkan file agregasi `loso_oof_summary_B_15hz.csv` (terhenti sebelum agregasi akhir). Dengan `PREV_RESULTS_DIR` di notebook training, proses LOSO bisa langsung dilanjutkan dari fold terakhir tanpa mengulang dari awal.
