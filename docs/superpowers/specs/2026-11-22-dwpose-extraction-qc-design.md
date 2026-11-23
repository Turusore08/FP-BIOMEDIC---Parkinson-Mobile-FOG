# Rancangan: Notebook Ekstraksi DWPose + QC (bagian Sultan)

Tanggal: 2026-11-22 · Pemilik: Sultan · Status: menunggu persetujuan

## 1. Maksud & alasan

FP Mobile-FOG (Kelompok 4) mendeteksi Freezing of Gait (FoG) dari video smartphone tugas
*turning-in-place* (35 subjek PD, 71 trial). Tiap anggota memegang satu pose estimator lalu melatih
classifier yang sama di atasnya: Rhama = YOLO26n-Pose, Faiz = MediaPipe, **Sultan = DWPose**.
Classifier: LightGBM, ST-GCN, Bi-Mamba (Rhama juga melatih Hybrid GCN+Bi-Mamba).

Tujuan riset tim adalah **membandingkan estimator pose**. Perbandingan hanya adil jika semua
yang lain sama: label, window, normalisasi, split, model, metrik. Rhama sudah punya pipeline
lengkap yang menjadi acuan:

- `parkinsonfp-biomed.ipynb` — ekstraksi YOLO + preprocessing Scenario A/B/D (commit `aeb2126`).
- `MobileFOG_Rhama_Training_4Models_Kaggle.ipynb` — training 4 model, split 23/5/7 + LOSO 35-fold.

Notebook ini adalah **padanan `parkinsonfp-biomed.ipynb` untuk DWPose**, ditambah tahap QC
(verifikasi kualitas pose) yang hanya mengukur & melaporkan. Hasilnya harus bisa langsung
dibaca notebook training Rhama hanya dengan mengganti `DATA_DIR`.

## 2. Goals / non-goals / anti-goals

**Goals**
1. Pose DWPose-m mentah (133 keypoint, 30 fps, resolusi asli) untuk ke-71 video.
2. Scenario A, B, D (15 Hz & 30 Hz) dengan format dan isi yang identik secara struktur dengan
   output Rhama, sehingga notebook training Rhama berjalan tanpa modifikasi selain path.
3. Laporan QC: metrik per trial, plot sinyal, klip video overlay untuk dicek manual, dan
   lembar catatan cek manual.

**Non-goals (tidak dikerjakan di sini)**
- Training classifier (memakai notebook training Rhama di tahap berikutnya).
- Topologi LB10 (tumit + jari kaki). Data mentah 133 keypoint disimpan supaya LB10 bisa
  dibangun belakangan tanpa ekstraksi ulang.
- Memperbaiki pipeline Rhama atau mengubah `dataset/src/` milik teman data prep.

**Anti-goals (gagal walaupun "jalan")**
- Preprocessing yang berbeda dari Rhama (fungsi, konstanta, urutan langkah, nama kunci npz).
- QC yang mengubah data masuk model (membuang trial, filter tambahan, koreksi manual keypoint).
- Memakai hasil QC/label pada split **test** untuk memilih konfigurasi preprocessing.
- Output besar (pose, video klip) ikut ter-commit; atau ada `git push` dari Claude.

## 3. Batasan

- Data lokal: `dataset/{train,val,test}/{videos,imu_128hz,labels_15hz,labels_30hz}` +
  `dataset/manifests/`. 71 video, 1280×720 @ ~29,98 fps, ±2 menit.
- Lingkungan eksekusi: **Kaggle Notebook, GPU T4 ×2, Internet ON** — sama dengan Rhama.
  Dataset dibaca dari Kaggle Input (dataset `Train Val Test` yang sama dengan yang dipakai
  Rhama). Batas Kaggle: sesi GPU maks. ±12 jam, kuota GPU mingguan terbatas, dan isi
  `/kaggle/working` hanya bertahan jika notebook dijalankan sebagai *Save Version (Run All)*.
- `rtmlib` mewajibkan paket `onnxruntime` (CPU) yang bentrok dengan `onnxruntime-gpu`
  (keduanya menyediakan modul `onnxruntime`). Wajib dipasang dengan
  `pip install --no-deps rtmlib` + `pip install onnxruntime-gpu`, lalu diverifikasi bahwa
  `CUDAExecutionProvider` tersedia. Jika tidak, ekstraksi diam-diam jalan di CPU.
- `.gitignore` tim mengabaikan `/output`; `dataset/` diabaikan lokal via `.git/info/exclude`.
- Total frame ±256 ribu (30 fps). Ekstraksi harus bisa dilanjutkan bila terputus.

## 4. Pendekatan yang dipertimbangkan

| | Pendekatan | Trade-off | Keputusan |
|---|---|---|---|
| A | **Tiru `parkinsonfp-biomed.ipynb`, ganti YOLO → DWPose, tambah QC** | Konfigurasi otomatis sama dengan Rhama; perlu adaptasi pemilihan pasien karena detektor rtmlib tidak mengembalikan skor | **Dipilih Sultan** |
| B | Tulis pipeline sendiri dari nol | Bebas, tapi selisih hasil bisa berasal dari beda kode, bukan beda estimator | Ditolak |
| C | Pakai `dataset/src/` (teman data prep) | Konvensi berbeda dari Rhama: ekstraksi 15 fps, resize 960 px, normalisasi & label per-frame berbeda; `mode="balanced"` di sana sebenarnya RTMW-l, bukan DWPose-m | Ditolak |

## 5. Desain

### 5.1 Lokasi file & lingkungan

- Notebook: `MobileFOG_Sultan_DWPose_Extraction_QC.ipynb` di root repo (sejajar notebook Rhama),
  self-contained (semua fungsi di dalam notebook, seperti gaya Rhama).
- `DATASET_ROOT`: auto-detect di `/kaggle/input/**` folder yang berisi `train/videos` dan
  `val/videos` (pola cell 2 Rhama).
- `OUTPUT_ROOT`: `/kaggle/working/mobilefog_sultan/`. Hasilnya diunduh atau dijadikan Kaggle
  Dataset agar bisa dipakai notebook training. Bila diunduh ke repo lokal, letakkan di
  `output/mobilefog_sultan/` (sudah di-ignore git).
- Dependensi: lihat Batasan. Sel instalasi gagal keras (`assert`) bila `CUDAExecutionProvider`
  tidak tersedia, karena ekstraksi CPU di Kaggle tidak realistis.
- **Bila sesi terputus**: hasil `.npz` sebelumnya dilampirkan kembali sebagai input (output
  versi notebook sebelumnya atau Kaggle Dataset). Notebook menyalinnya ke `OUTPUT_ROOT` di
  awal, lalu ekstraksi melewati trial yang sudah ada.

### 5.2 Tahap 0 — Audit data (salin cell 3 Rhama)

Manifest 71 trial (split, trial_id, subject_id, path video & label 15/30 Hz), assert 71 video
dan label lengkap, simpan `manifest_trials_local.csv`.

### 5.3 Tahap 1 — Ekstraksi DWPose-m

- **Model**: detektor YOLOX-m (HumanArt) 640×640 + **DWPose-m** 256×192 (133 keypoint
  COCO-WholeBody), lewat `rtmlib.YOLOX` dan `rtmlib.RTMPose` dengan URL eksplisit dari README
  rtmlib 0.0.16:
  - det: `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip`
  - pose: `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip`
  - **Jangan** memakai `Wholebody(mode=...)`: semua mode bawaan memakai RTMW, bukan DWPose.
- **Input**: setiap frame (stride 1, 30 fps), resolusi asli 1280×720 — sama seperti Rhama.
- **Pemilihan pasien per frame** (meniru `_select_main_person` Rhama):
  `skor = luas_box × conf × (1 − 0,3·|cx − W/2| / (W/2))`, pilih skor tertinggi, tanpa tracking
  antar frame. Karena YOLOX rtmlib tidak mengembalikan skor deteksi, `conf` diganti
  **rata-rata skor 17 keypoint tubuh (indeks 0–16) orang tersebut**. Pose dijalankan untuk
  semua box terdeteksi agar skor ini tersedia. Bila hanya 1 orang, langsung dipilih.
  Tidak ada orang → `kps` nol, `box` nol, `valid=False` (sama dengan Rhama).
- **Output per trial** `01_extracted_poses_raw/dwpose_m_sultan/{split}/{trial_id}.npz`:
  - `kps` (T, 133, 3) float32 — x, y dalam **piksel resolusi asli**, skor keypoint.
  - `boxes` (T, 4) float32 xyxy piksel · `valid` (T,) bool · `fps` float32 · `W`, `H` int32
    — kunci & tipe sama dengan Rhama.
  - Tambahan (tidak mengganggu loader Rhama): `n_persons` (T,) int16 jumlah orang terdeteksi,
    `ms_per_frame` float32.
- `01_extracted_poses_raw/joint_schema.json`: estimator, URL model, 133 kp, `lb6_indices =
  [11,12,13,14,15,16]` (indeks hip/knee/ankle sama di COCO-17 dan COCO-WholeBody),
  plus info kaki (17–22) untuk LB10 nanti.
- **Resume**: lewati trial yang `.npz`-nya sudah ada. Tulis ke file sementara lalu rename,
  supaya file setengah jadi tidak terbaca sebagai selesai.
- **Paralel 2× T4**: pola worker Rhama (skrip `.py` terpisah per GPU via `subprocess`,
  `CUDA_VISIBLE_DEVICES`, trial dibagi `tasks[i::n_gpus]`). Model diunduh sekali di proses
  utama sebelum worker jalan agar tidak saling berebut unduhan.
- **Benchmark**: ukur fps pada video pertama dan cetak estimasi waktu total. Jika estimasi
  > 9 jam (mendekati batas sesi 12 jam), jalankan per bagian (mis. `RUN_SPLITS = ["train"]`
  dulu, lalu val+test di sesi berikutnya) memakai mekanisme resume di atas.

### 5.4 Tahap 2–4 — Preprocessing A/B/D + smoke test (salin Rhama)

Salin **verbatim** konstanta preprocessing dari cell 2, lalu cell 5, cell 6, cell 7 dari `parkinsonfp-biomed.ipynb`
(commit `aeb2126`): `interpolate_low_conf`, `hampel_filter_1d`, `butter_lowpass`, `normalize_lb6`,
`build_scenario_a_and_b`, `process_frequency`, `build_lb6_graph_A`, `MobileFogWindowDataset`, serta
konstanta `CONF_VALID_THRES=0.20`, `HAMPEL_WINDOW=7`, `HAMPEL_SIGMA=3.0`, `BUTTER_CUTOFF_HZ=9.0`,
`WINDOW_SEC=2.0`, `STRIDE_SEC=1.0`, `RANDOM_SEED=42`.

Satu-satunya perubahan yang diizinkan: `RAW_POSE_DIR` → `dwpose_m_sultan`, nama subfolder
`yolo26n_lb6_{hz}hz` → `dwpose_m_lb6_{hz}hz`, dan `kps_17` boleh di-rename, karena isinya kini
(T, 133, 3). Jika ditemukan bug di kode Rhama, **jangan diperbaiki di sini**: catat dan laporkan
ke tim agar perbaikan diterapkan ke semua estimator.

Output: `02_processed_scenarios/{A_baseline_norm,B_denoised_hampel_accel}/dwpose_m_lb6_{15,30}hz/{split}/*.npz`
dan `02_processed_scenarios/D_windowed_stgcn_cls/from_{A_baseline_norm,B_denoised_hampel_accel}_{15,30}hz/{train,val,test}.npz`
+ `scaler_stats.npz`. Sebagai pemeriksaan kewajaran, jumlah window per split dibandingkan dengan
Rhama (train 5535 / val 1169 / test 1652). Angkanya boleh berbeda sedikit karena panjang pose
DWPose bisa berbeda beberapa frame. Perbedaan besar berarti ada masalah.

### 5.5 Tahap 5 — QC (hanya mengukur & melaporkan)

Dihitung dari pose mentah 30 fps. "Panjang kaki" = median per trial dari rata-rata
(hip→knee→ankle kiri, kanan) dalam piksel, definisi sama dengan `normalize_lb6`. Fungsi QC
menerima `(kps, lb6_indices)` dan format npz Rhama, sehingga **bisa juga dijalankan pada pose
YOLO Rhama** (dan MediaPipe Faiz) untuk tabel perbandingan, bila file mentah mereka tersedia.

**Metrik per trial** (`03_qc/qc_per_trial.csv`):

| Metrik | Definisi | Ambang flag awal |
|---|---|---|
| `n_pose`, `n_label30`, `n_video` | jumlah frame pose, baris label 30 Hz, `CAP_PROP_FRAME_COUNT` | `n_pose ≠ n_label30` |
| `valid_pct` | % frame ada orang terdeteksi | < 95 % |
| `lowconf_pct_<sendi>` | % frame skor < 0,2 per sendi LB6 (kaki 17–22 dilaporkan sebagai info) | sendi LB6 mana pun > 10 % |
| `spike_pct` | % (sendi LB6 × frame) dengan perpindahan antar frame > 0,4 × panjang kaki | > 2 % |
| `lr_swap_per_min` | per pasangan lutut & ankle: event bila `d_cross < 0,5·d_same` dan `d_same > 0,2·panjang kaki` (`d_same = ‖L_t−L_{t−1}‖+‖R_t−R_{t−1}‖`, `d_cross = ‖L_t−R_{t−1}‖+‖R_t−L_{t−1}‖`); dilaporkan nilai maksimum dari kedua pasangan | > 3 /menit |
| `person_switch_per_min` | event bila IoU box frame t vs t−1 < 0,3 (keduanya valid) | > 1 /menit |
| `multi_person_pct` | % frame `n_persons ≥ 2` | info saja |

Ambang di atas hanya untuk **memprioritaskan cek manual**, bukan untuk membuang data. Ambang boleh
disetel setelah melihat distribusi di train/val, tapi tidak boleh berdasarkan test.

**Visual**
- `03_qc/signals/{trial}.png` untuk semua 71 trial: posisi x & y ankle kiri/kanan
  (dinormalisasi panjang kaki) terhadap waktu, episode FoG diarsir, event spike/swap/switch
  ditandai.
- `03_qc/clips/{trial}_{start}s.mp4`: klip overlay 15 detik (skeleton tubuh + kaki, LB6
  ditebalkan, box pasien, status FoG, waktu, skor). Dibuat untuk **semua trial yang ter-flag**
  dan **6 trial sampel** (seed 42, dari train/val: 2 FoG tinggi, 2 FoG rendah, 2 tanpa FoG).
  Per trial maksimal 2 klip: satu di sekitar episode FoG (bila ada), satu di sekitar event QC
  terparah (atau acak bila tidak ada event).
- Ringkasan di notebook: tabel trial ter-flag + histogram tiap metrik.

**Cek manual**: `03_qc/qc_manual_review.csv` (dibuat sekali, tidak ditimpa bila sudah ada):
`trial_id, clip, checked_by, person_ok, left_right_ok, joints_ok, notes`. Klip dan CSV
diunduh dari output Kaggle, ditonton dan diisi Sultan di laptop. CSV yang sudah diisi
disimpan di repo (`qc/qc_manual_review.csv`) sebagai bukti verifikasi. Klip tidak di-commit.

### 5.6 Kontrak untuk tahap training

Notebook training Rhama membaca `DATA_DIR/02_processed_scenarios/D_windowed_stgcn_cls/` dengan
kunci `X, X_scaled, y_any, y_maj, y_ge25, y_ratio, trial_id, subject_id, balanced_train_idx`.
`DATA_DIR` diarahkan ke `mobilefog_sultan`. Auto-detect Rhama mencari nama `mobilefog_rhama`,
jadi `DATA_DIR` harus diisi manual.

## 6. Diserahkan ke builder

- Struktur sel notebook, nama fungsi QC, gaya plot & overlay.
- Versi `onnxruntime-gpu` yang cocok dengan CUDA image Kaggle saat itu.
- Threshold deteksi YOLOX bawaan rtmlib dipakai apa adanya.
- Pemilihan waktu klip, warna, dan font overlay.

## 7. Keputusan yang saya ambil (mohon dicek)

1. Ekstraksi 30 fps, resolusi asli, simpan 133 keypoint float32 (mengikuti Rhama; LB10 tetap mungkin).
2. Model DWPose-m asli (bukan RTMW lewat `mode="balanced"`), detektor YOLOX-m.
3. Pengganti skor deteksi = rata-rata skor 17 keypoint tubuh.
4. QC hanya melapor. Ambang flag awal seperti tabel 5.5, dan tidak disetel memakai test.
5. Bug di kode Rhama dilaporkan ke tim, tidak diperbaiki sepihak.
6. Jalan di Kaggle 2× T4 dengan pola paralel Rhama. Bila estimasi > 9 jam, dicicil per
   split dengan resume lintas sesi.
7. Klip QC ditonton di laptop. Hanya CSV catatan cek manual yang masuk repo.

## Addendum 2026-11-23 — PatientLock-2Pass (menggantikan §5.3 "Pemilihan pasien")

**Alasan.** Run Kaggle pertama + uji lokal menunjukkan aturan `area × conf × center` tanpa tracking sering
memilih asisten (DWPose/proxy body-17: 57 loncatan per 300 frame; pipeline YOLO tim: 17; ±41% frame salah
orang di PDFE02_1 f0–1200). Tim sudah menyelesaikannya di notebook ekstraksi YOLO (`origin/main` `5296a14`,
`ec495f8`) dengan **PatientLock-2Pass** + dataset Kaggle `ajikajadeh/mobile-fog-patient-bboxes`.
Keputusan Sultan: **ikuti notebook ekstraksi YOLO tim, ganti estimatornya dengan DWPose.**

**Desain (meniru sel 4 notebook YOLO tim):**
- Input per trial: `patient_bboxes_npz/{split}/{trial}.npz` berisi `patient_box_norm (T,4)`, `is_detected (T,)`,
  `non_patient_flat (N,≥4)`, `non_patient_offsets (T+1,)`.
- Per frame: jika `f < T_bb` dan `is_detected[f]` → crop = box pasien + padding `CROP_PAD_RATIO` (0,15). Box
  pengawas ditutup (114,114,114) kecuali area inti pasien (`MASK_SUPERVISORS`). Lalu detektor YOLOX-m
  dijalankan di crop dengan ambang `DET_CONF` (0,20, skor dibaca langsung dari output ONNX karena rtmlib
  mengunci 0,3). Box dengan IoU terbesar terhadap box inti dipilih; kalau IoU < `MIN_CROP_IOU` (0,20) →
  frame invalid. DWPose-m dijalankan **hanya untuk box itu**, lalu keypoint di-remap ke frame penuh.
- `boxes` yang disimpan = box pasien (piksel), sama dengan tim. Frame tanpa bbox atau tanpa match →
  `kps` nol, `valid=False`. `n_persons` = jumlah deteksi di dalam crop (info QC).
- Konstanta disalin persis dari sel 2 notebook YOLO tim (`DET_CONF`, `CROP_PAD_RATIO`, `MASK_SUPERVISORS`,
  `MIN_CROP_IOU`, + konstanta preprocessing). Sel 3 (manifest + `bbox_npz_path`) disalin persis versi baru.
- Pengganti det-conf body-17 (`select_person.py`) dihapus.
- Npz menyimpan `selector="patientlock-2pass"`. `is_complete()` mewajibkannya, jadi output run pertama (tanpa
  PatientLock) yang terlampir sebagai Input tidak akan dipakai ulang.
- Output root: `mobilefog_dwpose_patientlock` (meniru `mobilefog_rhama_patientlock`). Pose mentah tetap di
  `01_extracted_poses_raw/dwpose_m/`.
