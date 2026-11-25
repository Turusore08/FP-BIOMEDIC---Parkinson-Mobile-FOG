# Analisis Perbandingan 3 Tahap Eksperimen: Baseline (`output training`) vs Bounding Box (`output bounding box`) vs Clean Dataset (`output without noise`)

Dokumen ini menganalisis perkembangan performa deteksi *Freezing of Gait* (FoG) pada penyakit Parkinson melintasi **3 tahap iterasi pipeline** menggunakan **4 arsitektur model** (`Light_ST-GCN`, `LightGBM+FFT`, `Pure_Bi-Mamba`, `Hybrid_GCN+BiMamba`) dan **4 skenario preprocessing** (`A_15hz`, `B_15hz`, `A_30hz`, `B_30hz`):

| Tahap | Folder Hasil | Konfigurasi Pipeline & Dataset | Split Subjek (Video) | Sumber CSV |
| :--- | :--- | :--- | :---: | :--- |
| **1. Baseline (Raw)** | `output training/` *(pose mentah dari `output/`)* | `YOLO26n-pose` langsung tanpa isolasi pasien (`PatientLock`) | `23` Train / `5` Val / `7` Test (**35 subjek / 71 video**) | [results_summary.csv](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20training/mobilefog_training_results/results_summary.csv) |
| **2. + Bounding Box** | `output bounding box/` | `PatientLock-2Pass` BBox (`ajikajadeh/mobile-fog-patient-bboxes`) + *crop* + *masking* pengawas | `23` Train / `5` Val / `7` Test (**35 subjek / 71 video**) | [results_summary.csv](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20bounding%20box/mobilefog_training_results/results_summary.csv) |
| **3. + Without Noise (Clean)** | `output without noise/` | `PatientLock-2Pass` BBox + pembersihan **3 subjek bermasalah** (`15`, `24`, `27` → 5 trial dibuang: `PDFE15_1`, `PDFE24_1`, `PDFE27_1–3`) | `22` Train / `4` Val / `6` Test (**32 subjek / 66 video**) | [results_summary.csv](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20without%20noise/mobilefog_training_results/results_summary.csv) |

---

## 1. Ringkasan Eksekutif: Progresi Performa 3 Tahap

Setiap tahap perbaikan data memberikan peningkatan performa yang sangat jelas. **Bounding Box (`PatientLock-2Pass`)** menyumbang lonjakan terbesar pertama (`+0.1698` rata-rata `Test PR-AUC`), kemudian **penghapusan 3 subjek noisy (`output without noise`)** mendorong performa menembus **PR-AUC > 0.90**, **ROC-AUC > 0.96**, dan **Balanced Accuracy > 90%**.

### 1.1 Metrik Terbaik (Best on Test Set)

| Metrik Terbaik | Tahap 1: Baseline (`output training`) | Tahap 2: + BBox (`output bounding box`) | Tahap 3: + Without Noise (`output without noise`) | Total Peningkatan ($\Delta$ 1→3) |
| :--- | :---: | :---: | :---: | :---: |
| **Test PR-AUC Tertinggi** | `0.6631` (`Pure_Bi-Mamba`, `B_15hz`) | `0.8366` (`Light_ST-GCN`, `B_15hz`) | **`0.9036`** (`Light_ST-GCN`, `B_15hz`) | **`+0.2405`** (`+36,3%` rel.) |
| **Test ROC-AUC Tertinggi** | `0.8715` (`LightGBM+FFT`, `B_30hz`) | `0.9460` (`LightGBM+FFT`, `B_30hz`) | **`0.9625`** (`LightGBM+FFT`, `B_30hz`) | **`+0.0910`** |
| **Test F1-FoG (PP) Tertinggi** | `0.6390` (`LightGBM+FFT`, `B_15hz`) | `0.7529` (`LightGBM+FFT`, `B_15hz`) | **`0.8293`** (`LightGBM+FFT`, `B_15hz`) | **`+0.1903`** |
| **Test Balanced Acc Tertinggi** | `0.7815` (`LightGBM+FFT`, `B_30hz`) | `0.8522` (`LightGBM+FFT`, `A_15hz`) | **`0.9054`** (`Light_ST-GCN`, `B_15hz`) | **`+0.1239`** |
| **Test Sensitivitas Tertinggi** | `0.7270` (`LightGBM+FFT`, `B_30hz`) | `0.7818` (`LightGBM+FFT`, `A_15hz`) | **`0.8723`** (`Light_ST-GCN`, `B_15hz`) | **`+0.1453`** |
| **Test Spesifisitas Tertinggi** | `0.9674` (`Light_ST-GCN`, `A_30hz`) | `0.9726` (`LightGBM+FFT`, `A_30hz`) | **`0.9896`** (`Pure_Bi-Mamba`, `B_30hz`) | **`+0.0222`** |
| **Val PR-AUC Tertinggi** | `0.5782` (`LightGBM+FFT`, `B_15hz`) | **`0.7292`** (`Pure_Bi-Mamba`, `B_15hz`) | `0.7225` (`Pure_Bi-Mamba`, `B_15hz`) | **`+0.1443`** |

### 1.2 Rata-Rata Seluruh 16 Eksperimen (`4 Model × 4 Skenario`)

| Rata-Rata Metrik (16 Run) | Baseline (`output training`) | + BBox (`output bounding box`) | + Without Noise (`output without noise`) | $\Delta$ (BBox vs Base) | $\Delta$ (Clean vs BBox) | $\Delta$ (Clean vs Base) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Test PR-AUC** | `0.5375` | `0.7073` | **`0.7658`** | `+0.1698` | `+0.0585` | **`+0.2283`** |
| **Mean Test ROC-AUC** | `0.7531` | `0.8435` | **`0.8770`** | `+0.0904` | `+0.0335` | **`+0.1239`** |
| **Mean Test F1-PP** | `0.5257` | `0.6358` | **`0.7038`** | `+0.1101` | `+0.0681` | **`+0.1782`** |
| **Mean Test Sens-PP** | `0.5343` | `0.5938` | **`0.6480`** | `+0.0594` | `+0.0542` | **`+0.1136`** |
| **Mean Test Spec-PP** | `0.8946` | `0.9411` | **`0.9588`** | `+0.0464` | `+0.0178` | **`+0.0642`** |
| **Mean Test Bal-Acc-PP** | `0.7145` | `0.7674` | **`0.8034`** | `+0.0529` | `+0.0360` | **`+0.0889`** |
| **Mean Waktu Training** | `422.8 dtk` | `418.5 dtk` | **`100.0 dtk`** | `-4.4 dtk` | `-318.5 dtk` | **`-322.8 dtk`** (`4,2×` lebih cepat) |

> **Win-Rate (`output without noise`) dari 16 eksperimen:**
> - **vs Baseline (`output training`)**: Menang **16/16** (`100%`) pada `PR-AUC` & `F1-PP`, serta **15/16** pada `ROC-AUC` & `Bal-Acc`.
> - **vs BBox (`output bounding box`)**: Menang **16/16** (`100%`) pada `F1-PP`, **13/16** pada `PR-AUC` & `ROC-AUC`, serta **12/16** pada `Bal-Acc`. *(Satu-satunya model yang tidak naik pada `PR-AUC` di 3 skenario adalah `Hybrid_GCN+BiMamba`.)*

---

## 2. Perbandingan Per-Model (Skenario Terbaik & Rata-Rata 4 Skenario)

| Model | Params | Baseline (`PR` / `ROC` / `F1-PP`) | + BBox (`PR` / `ROC` / `F1-PP`) | + Without Noise (`PR` / `ROC` / `F1-PP`) | Mean PR-AUC (`Base → BBox → Clean`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Light_ST-GCN`** | `38,801` | `0.5756` / `0.8442` / `0.6053` (`A_15hz`) | `0.8366` / `0.9379` / `0.7474` (`B_15hz`) | **`0.9036`** / **`0.9612`** / `0.8132` (`B_15hz`) | `0.4915 → 0.7875 →` **`0.8381`** (`+0.3467`) |
| **`LightGBM+FFT`** | `46–100` trees | `0.6294` / `0.8602` / `0.5866` (`A_15hz`) | `0.8365` / `0.9441` / `0.7529` (`B_15hz`) | `0.8861` / `0.9590` / **`0.8293`** (`B_15hz`) | `0.6043 → 0.8203 →` **`0.8680`** (`+0.2637`) |
| **`Pure_Bi-Mamba`** | `211,521` | `0.6631` / `0.7689` / `0.6124` (`B_15hz`) | `0.6802` / `0.7966` / `0.6130` (`A_15hz`) | **`0.8292`** / `0.9249` / `0.7610` (`A_15hz`) | `0.5853 → 0.6126 →` **`0.7511`** (`+0.1658`) |
| **`Hybrid_GCN+BiMamba`** | `216,345` | `0.6065` / `0.7514` / `0.5340` (`A_15hz`) | **`0.6699`** / **`0.8479`** / `0.5809` (`B_15hz`) | `0.6642` / `0.8117` / **`0.6271`** (`A_15hz`) | `0.4689 → 0.6089 → 0.6060` (`+0.1371`) |

---

## 3. Tabel Lengkap 16 Eksperimen (`4 Skenario × 4 Model`)

Diurutkan berdasarkan `Test PR-AUC` tertinggi pada tahap **`output without noise`**:

| Skenario | Model | Test PR-AUC (`Base → BBox → Clean`) | $\Delta$ Clean vs BBox | $\Delta$ Clean vs Base | Test ROC-AUC (`Base → BBox → Clean`) | Test F1-PP (`Base → BBox → Clean`) | Test Bal-Acc (`Base → BBox → Clean`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `B_15hz` | **`Light_ST-GCN`** | `0.5730 → 0.8366 →` **`0.9036`** | **`+0.0670`** | **`+0.3306`** | `0.8053 → 0.9379 →` **`0.9612`** | `0.5615 → 0.7474 → 0.8132` | `0.7176 → 0.8283 →` **`0.9054`** |
| `B_15hz` | **`LightGBM+FFT`** | `0.6066 → 0.8365 →` **`0.8861`** | `+0.0496` | `+0.2795` | `0.8672 → 0.9441 → 0.9590` | `0.6390 → 0.7529 →` **`0.8293`** | `0.7690 → 0.8410 → 0.8799` |
| `B_30hz` | **`LightGBM+FFT`** | `0.6137 → 0.8314 →` **`0.8818`** | `+0.0504` | `+0.2681` | `0.8715 → 0.9460 →` **`0.9625`** | `0.5925 → 0.7348 → 0.8104` | `0.7815 → 0.8261 → 0.8562` |
| `A_30hz` | **`LightGBM+FFT`** | `0.5676 → 0.7873 → 0.8545` | `+0.0672` | `+0.2869` | `0.8527 → 0.9285 → 0.9536` | `0.5812 → 0.6960 → 0.7573` | `0.7636 → 0.7856 → 0.8172` |
| `A_15hz` | **`Light_ST-GCN`** | `0.5756 → 0.8250 → 0.8497` | `+0.0247` | `+0.2741` | `0.8442 → 0.9309 → 0.9428` | `0.6053 → 0.7278 → 0.7632` | `0.7562 → 0.8471 → 0.8412` |
| `A_15hz` | **`LightGBM+FFT`** | `0.6294 → 0.8258 → 0.8496` | `+0.0238` | `+0.2202` | `0.8602 → 0.9368 → 0.9508` | `0.5866 → 0.7373 → 0.7707` | `0.7519 → 0.8522 → 0.8398` |
| `B_30hz` | **`Light_ST-GCN`** | `0.3790 → 0.7611 → 0.8322` | `+0.0711` | **`+0.4532`** | `0.6919 → 0.9095 → 0.9317` | `0.3926 → 0.6690 → 0.7927` | `0.6249 → 0.7851 → 0.8672` |
| `A_15hz` | **`Pure_Bi-Mamba`** | `0.6385 → 0.6802 → 0.8292` | **`+0.1490`** | `+0.1907` | `0.7659 → 0.7966 → 0.9249` | `0.5710 → 0.6130 → 0.7610` | `0.7503 → 0.7519 → 0.8361` |
| `B_15hz` | **`Pure_Bi-Mamba`** | `0.6631 → 0.6056 → 0.7791` | **`+0.1735`** | `+0.1160` | `0.7689 → 0.7673 → 0.9013` | `0.6124 → 0.5517 → 0.6907` | `0.7502 → 0.7081 → 0.7782` |
| `A_30hz` | **`Light_ST-GCN`** | `0.4383 → 0.7273 → 0.7670` | `+0.0397` | `+0.3287` | `0.7072 → 0.9108 → 0.9141` | `0.3700 → 0.6626 → 0.7109` | `0.6136 → 0.8081 → 0.8347` |
| `B_30hz` | **`Pure_Bi-Mamba`** | `0.5042 → 0.6331 → 0.6997` | `+0.0666` | `+0.1955` | `0.7112 → 0.7863 → 0.8301` | `0.5424 → 0.5842 → 0.6062` | `0.7391 → 0.7343 → 0.7224` |
| `A_30hz` | **`Pure_Bi-Mamba`** | `0.5352 → 0.5315 → 0.6964` | **`+0.1649`** | `+0.1612` | `0.6841 → 0.6696 → 0.8147` | `0.5801 → 0.5133 → 0.6467` | `0.7329 → 0.6841 → 0.7559` |
| `A_15hz` | `Hybrid_GCN+BiMamba` | `0.6065 → 0.5772 → 0.6642` | `+0.0870` | `+0.0577` | `0.7514 → 0.7287 → 0.8117` | `0.5340 → 0.5258 → 0.6271` | `0.7169 → 0.7038 → 0.7456` |
| `B_15hz` | `Hybrid_GCN+BiMamba` | `0.4849 → 0.6699 → 0.6531` | `-0.0168` | `+0.1682` | `0.7159 → 0.8479 → 0.8033` | `0.4969 → 0.5809 → 0.5979` | `0.7144 → 0.7215 → 0.7557` |
| `B_30hz` | `Hybrid_GCN+BiMamba` | `0.5453 → 0.6243 → 0.5779` | `-0.0464` | `+0.0326` | `0.7003 → 0.7528 → 0.6959` | `0.4753 → 0.5540 → 0.5578` | `0.6805 → 0.7048 → 0.7270` |
| `A_30hz` | `Hybrid_GCN+BiMamba` | `0.2388 → 0.5642 → 0.5288` | `-0.0354` | `+0.2900` | `0.4517 → 0.7018 → 0.6747` | `0.2703 → 0.5213 → 0.5265` | `0.5690 → 0.6963 → 0.6917` |

---

## 4. Analisis Mendalam: Mengapa Setiap Tahap Berdampak Berbeda?

### 4.1 Dampak Bounding Box (`output training` → `output bounding box`)
1. **Menghilangkan *ID Switching* ke Pengawas (`97,19%` frame multi-person):**
   Tanpa *bounding box* pasien, `YOLO26n-pose` sesekali melompat ke pengawas klinis yang berdiri diam. Lompatan mendadak tersebut menciptakan kecepatan (`dx, dy`) dan energi spektral palsu yang merusak sinyal FoG `3–8 Hz`.
2. **`Light_ST-GCN` & `LightGBM+FFT` Paling Diuntungkan di Tahap BBox:**
   - Mean `PR-AUC` `Light_ST-GCN` melonjak **`+0.2960`** (`0.4915 → 0.7875`) karena graf tulang `LB6` (`Hip–Knee–Ankle`) murni merekam kinematika kaki pasien.
   - Mean `PR-AUC` `LightGBM+FFT` melonjak **`+0.2159`** (`0.6043 → 0.8203`) dengan komputasi sangat ringan (`~1.8 detik` training).

### 4.2 Dampak Penghapusan 3 Subjek Noisy (`output bounding box` → `output without noise`)
Pembuangan 5 trial dari 3 subjek bermasalah (`PDFE15_1` di Train, `PDFE27_1–3` di Val, `PDFE24_1` di Test) menghasilkan dinamika menarik:
1. **`Pure_Bi-Mamba` Akhirnya "Bangun" (`+0.1385` mean PR-AUC):**
   Di tahap BBox saja, `Pure_Bi-Mamba` hanya naik tipis (`0.5853 → 0.6126`). Namun begitu subjek noisy dibuang, `Pure_Bi-Mamba` mencatat **lonjakan terbesar di tahap 3** (`mean PR-AUC: 0.6126 → 0.7511`, puncaknya di `A_15hz` mencapai `PR-AUC = 0.8292`, `ROC-AUC = 0.9249`). Artinya, model *state-space sequence* berkapasitas besar (`~211k` params) sangat rentan terganggu oleh *tracking/label noise* dari beberapa subjek outlier.
2. **`Light_ST-GCN` (`B_15hz`) Menjadi Juara Baru:**
   Mencapai **`PR-AUC = 0.9036`**, **`ROC-AUC = 0.9612`**, **`Bal-Acc = 0.9054`** (`Sens = 87.23%`, `Spec = 93.85%`), mengungguli `LightGBM+FFT` pada `PR-AUC` dan `Balanced Accuracy` sekaligus mempertahankan jumlah parameter yang sangat ringkas (`38,801` params).
3. **`LightGBM+FFT` Tetap Paling Konsisten & Praktis:**
   Memiliki **mean `PR-AUC` tertinggi (`0.8680`)** melintasi keempat skenario, **`F1-PP` tertinggi (`0.8293` di `B_15hz`)**, **`ROC-AUC` tertinggi (`0.9625` di `B_30hz`)**, dengan waktu training hanya **`1.0–1.3 detik`** (tanpa GPU).
4. **`Hybrid_GCN+BiMamba` Mengalami Stagnasi (`0.6089 → 0.6060`):**
   Kombinasi GCN + BiMamba (`~216k` params) terlalu kompleks untuk jumlah subjek latih yang berkurang (`22` subjek), sehingga cenderung *overfit* dan kalah jauh dibanding `Light_ST-GCN` maupun `Pure_Bi-Mamba` yang berdiri sendiri.
5. **Skenario `B_15hz` (`Hampel + Butterworth 9 Hz` + `15 Hz`) Pemenang Mutlak:**
   *Downsampling* ke `15 Hz` (`T = 30` frame) + denoising `Scenario B` secara konsisten menghasilkan skor puncak untuk `Light_ST-GCN` dan `LightGBM+FFT` di tahap BBox maupun tahap Clean.

> **Catatan Metodologis Penting (Transparansi Evaluasi):**
> Karena `PDFE24_1` (subjek `24`) termasuk dalam **Test Set** resmi (`7 → 6` subjek test) dan `PDFE27_1–3` termasuk dalam **Val Set** (`5 → 4` subjek val), sebagian kenaikan metrik Fixed Split pada `output without noise` berasal dari penghapusan subjek test/val yang bermasalah (terlihat juga dari sedikit bergesernya `Val PR-AUC` rata-rata dari `0.6466 → 0.6276`). Oleh karena itu, evaluasi **LOSO 32-Fold** di bawah menjadi bukti generalisasi yang jauh lebih objektif.

---

## 5. Status & Hasil Evaluasi LOSO (`B_15hz`)

| Tahap | Status LOSO (`loso_35fold_B_15hz/`) | File Ringkasan |
| :--- | :---: | :--- |
| **1. Baseline (`output training`)** | Belum selesai (`27 / 35` fold `.npz`) | Belum ada `loso_oof_summary_B_15hz.csv` |
| **2. + BBox (`output bounding box`)** | Hampir selesai (`34 / 35` fold `.npz`, kurang 1 fold) | Belum ada `loso_oof_summary_B_15hz.csv` |
| **3. + Without Noise (`output without noise`)** | **Selesai 100% (`32 / 32` fold `.npz`)** | [loso_oof_summary_B_15hz.csv](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20without%20noise/mobilefog_training_results/loso_oof_summary_B_15hz.csv) & [loso_per_subject_B_15hz.csv](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20without%20noise/mobilefog_training_results/loso_per_subject_B_15hz.csv) |

### 5.1 Hasil Agregasi Out-Of-Fold (OOF) LOSO 32-Subjek (`output without noise`, `B_15hz`)

| Model | Folds | OOF ROC-AUC | OOF PR-AUC | OOF F1-Raw | OOF F1-PP | OOF Sens-PP | OOF Spec-PP | OOF Bal-Acc-PP | Mean PR-AUC (Subjek FoG) | Mean Spec (32 Subjek) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Light_ST-GCN`** | `32` | **`0.9170`** | **`0.7284`** | `0.6556` | `0.6711` | `0.6609` | `0.9296` | `0.7953` | `0.554 ± 0.331` | `0.861 ± 0.221` |
| **`LightGBM+FFT`** | `32` | `0.8406` | `0.5357` | **`0.7269`** | **`0.7515`** | **`0.6755`** | **`0.9722`** | **`0.8238`** | **`0.565 ± 0.354`** | **`0.919 ± 0.127`** |
| **`Pure_Bi-Mamba`** | `32` | `0.8072` | `0.6706` | `0.5919` | `0.6006` | `0.5215` | `0.9510` | `0.7362` | `0.550 ± 0.310` | `0.916 ± 0.145` |
| **`Hybrid_GCN+BiMamba`** | `32` | `0.7817` | `0.6021` | `0.5369` | `0.5502` | `0.5153` | `0.9184` | `0.7168` | `0.499 ± 0.342` | `0.891 ± 0.184` |

### 5.2 Insight Utama dari Analisis Per-Subjek LOSO (`32 Subjek`)
1. **Perbedaan Karakteristik `Light_ST-GCN` vs `LightGBM+FFT` pada LOSO:**
   - **`Light_ST-GCN`** unggul telak dalam **ranking probabilitas global lintas subjek** (`OOF ROC-AUC = 0.9170`, `OOF PR-AUC = 0.7284`), menandakan kalibrasi probabilitasnya paling stabil lintas pasien.
   - **`LightGBM+FFT`** unggul telak dalam **keputusan biner pasca-thresholding** (`OOF F1-PP = 0.7515`, `OOF Bal-Acc = 0.8238`, `Spec = 0.9722`) serta memiliki rata-rata `PR-AUC` per-subjek tertinggi (`0.565 ± 0.354`). Nilai `OOF PR-AUC` agregatnya lebih rendah (`0.5357`) karena skala probabilitas antar-fold pohon keputusan tidak terkalibrasi seragam saat digabung mentah tanpa *fold-wise calibration*.
2. **Performa Sangat Tinggi pada Subjek Non-FoG & FoG Dominan:**
   - Pada **9 subjek tanpa FoG** (`10, 12, 13, 17, 22, 25, 26, 28, 35`), spesifisitas sangat tinggi (mayoritas `1.000`, sangat minim *false alarm*).
   - Pada subjek dengan episode FoG signifikan (`01`, `02`, `07`, `11`, `19`, `33`), `Light_ST-GCN` dan `LightGBM+FFT` konsisten mencapai `PR-AUC 0.78–0.996` dan `F1-PP` hingga `0.969`.
3. **Sumber Variansi Tinggi (`± 0.31–0.35`):**
   - Subjek dengan prevalensi FoG sangat jarang (`< 5%` window: `03`, `06`, `18`, `20`, `21`, `23`, `32`) sangat sensitif terhadap beberapa *false positive* atau *missed window*, sehingga menarik turun rata-rata `PR-AUC` per-subjek meskipun `ROC-AUC`-nya tinggi (misal subjek `06` dan `32` memiliki `ROC-AUC > 0.90` namun `F1-PP = 0.0`).

---

## 6. Rekomendasi Model & Langkah Selanjutnya

1. **Model Utama untuk Deployment / Laporan Akhir:**
   - **`Light_ST-GCN` (`B_15hz`)** sebagai model *deep learning* terbaik: ringan (`38.8k` params), terbaik di Fixed Split (`PR-AUC 0.9036`, `Bal-Acc 0.9054`), dan terbaik dalam kalibrasi probabilitas lintas subjek di LOSO (`OOF PR-AUC 0.7284`, `OOF ROC-AUC 0.9170`).
   - **`LightGBM+FFT` (`B_15hz`)** sebagai baseline *non-deep* ultra-ringan: terbaik di `F1-PP` baik pada Fixed Split (`0.8293`) maupun LOSO (`0.7515`, `Bal-Acc 0.8238`) dengan waktu latih `~1.3 detik` di CPU.
2. **Menyelesaikan 1 Fold Tersisa pada `output bounding box`:**
   - Folder `output bounding box/mobilefog_training_results/loso_35fold_B_15hz/` sudah memiliki `34/35` fold. Menjalankan 1 fold terakhir via `PREV_RESULTS_DIR` akan menghasilkan perbandingan LOSO *head-to-head* antara **35 subjek penuh** vs **32 subjek bersih**.
3. **Referensi Visualisasi:**
   - Grafik perbandingan 4 model (Baseline): [rhama_4models_comparison.png](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20training/mobilefog_training_results/figures/rhama_4models_comparison.png)
   - Grafik perbandingan 4 model (+ BBox): [rhama_4models_comparison.png](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20bounding%20box/mobilefog_training_results/figures/rhama_4models_comparison.png)
   - Grafik perbandingan 4 model (+ Without Noise): [rhama_4models_comparison.png](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20without%20noise/mobilefog_training_results/figures/rhama_4models_comparison.png)
   - Grafik LOSO 32-Fold (+ Without Noise): [loso_35fold_B_15hz.png](file:///home/turusore08/Desktop/Semester%205/Biomed/FP/FP-BIOMEDIC---Parkinson-Mobile-FOG/output%20without%20noise/mobilefog_training_results/figures/loso_35fold_B_15hz.png)
