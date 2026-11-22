# DWPose Extraction + QC Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Kaggle notebook `MobileFOG_Sultan_DWPose_Extraction_QC.ipynb`. It extracts DWPose-m poses for the 71 videos, preprocesses them into Scenario A/B/D exactly as Rhama does, and produces a QC report that only measures and reports.

**Architecture:** Testable logic lives in small Python modules under `sultan_dwpose/`. These cover person selection, the extraction worker, QC metrics, and QC visuals, and they are unit-tested locally with pytest. `sultan_dwpose/build_notebook.py` then generates one self-contained notebook. It inlines the module sources, and it pulls Rhama's preprocessing cells **programmatically** from `parkinsonfp-biomed.ipynb` with an exact, counted set of string replacements, which guarantees verbatim parity. On Kaggle the notebook writes the worker to a `.py` file and runs one subprocess per T4, the same pattern Rhama uses.

**Tech Stack:** Python 3.10+ (Kaggle) / 3.12 (local tests), numpy, pandas, scipy, opencv, matplotlib, rtmlib 0.0.16 (`--no-deps`), onnxruntime-gpu, pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-dwpose-extraction-qc-design.md`

## Global Constraints

- Reference notebook: `parkinsonfp-biomed.ipynb` at commit `aeb2126`. Preprocessing cells are copied, never re-typed.
- Allowed replacements in Rhama's cells only: `yolo26n_lb6_` → `dwpose_m_lb6_`, `mobilefog_rhama` → `mobilefog_sultan`. Nothing else may change.
- Models: det `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip`, input (640, 640). Pose `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip`, input (192, 256). Never `Wholebody(mode=...)`.
- Extraction runs on every frame (stride 1) at native resolution. Keypoints are saved in pixel coordinates.
- Raw npz keys: `kps` (T,133,3) float32, `boxes` (T,4) float32, `valid` (T,) bool, `fps` float32, `W` int32, `H` int32, `n_persons` (T,) int16, `ms_per_frame` float32.
- Paths: raw `OUTPUT_ROOT/01_extracted_poses_raw/dwpose_m_sultan/{split}/{trial_id}.npz`. Schema `OUTPUT_ROOT/01_extracted_poses_raw/joint_schema.json`. QC output in `OUTPUT_ROOT/03_qc/`. `OUTPUT_ROOT = /kaggle/working/mobilefog_sultan`.
- Install: `pip install -q --no-deps rtmlib==0.0.16` then `pip install -q onnxruntime-gpu`. Hard `assert` if `CUDAExecutionProvider` is missing.
- QC thresholds: `valid_pct < 95`, any LB6 `lowconf_pct > 10`, `spike_pct > 2`, `lr_swap_per_min > 3`, `person_switch_per_min > 1`, `n_pose != n_label30`. Low-conf means score < 0.2. A spike is a step > 0.4 × leg length. Swap rule: `d_cross < 0.5·d_same` and `d_same > 0.2·leg`. A person switch is box IoU < 0.3.
- QC never modifies data feeding the model. Sample trials for clips come from train/val only (seed 42: 2 high FoG, 2 low FoG, 2 zero FoG).
- Git: no `git push` ever. Commit only when Sultan asks, so the steps below end with "stage nothing" checkpoints instead of commits.
- Local tests run in an isolated venv: `$S/venv` where `S` is the session scratchpad, created with `--system-site-packages`. Never install into Sultan's global Python.

## Review Focus

1. **A trial or a long run of frames with no detected person.** Selection returns zeros/`valid=False`, and QC must not crash or produce NaN flags. *Test: Task 4 `test_qc_trial_all_invalid`.*
2. **An assistant who is larger than the patient but near the frame edge.** The center-penalised score must pick the central person. *Test: Task 2 `test_select_prefers_central_over_larger_edge`.*
3. **A session killed mid-write, leaving a partial npz.** The worker writes to `*.tmp.npz` then renames, and resume skips only complete files. *Test: Task 3 `test_worker_atomic_and_resume`.*
4. **Pose frame count differing from the 30 Hz label rows.** QC flags it, and preprocessing keeps Rhama's `min()` truncation. *Test: Task 4 `test_qc_flags_frame_mismatch`.*
5. **Kaggle image where onnxruntime-gpu falls back to CPU.** The install check raises with a clear message. *Test: Task 6 `test_require_cuda_provider_raises`.*

---

### Task 1: Rhama cell extraction + verbatim transform

**Files:**
- Create: `sultan_dwpose/__init__.py` (empty), `sultan_dwpose/rhama_cells.py`
- Test: `sultan_dwpose/tests/test_rhama_cells.py`

**Interfaces:**
- Produces:
  - `load_rhama_cells(nb_path: Path) -> list[str]`: code-cell sources of `parkinsonfp-biomed.ipynb`, indexed by notebook cell index.
  - `transform_rhama_source(src: str) -> tuple[str, dict[str, int]]`: applies exactly the two allowed replacements and returns the new source plus replacement counts.
  - `rhama_constants_block(cell2_src: str) -> str`: the verbatim lines of cell 2 that define `LB6_INDICES, LB6_NAMES, CONF_VALID_THRES, HAMPEL_WINDOW, HAMPEL_SIGMA, BUTTER_CUTOFF_HZ, WINDOW_SEC, STRIDE_SEC, RANDOM_SEED` (with their comment lines).
  - `RHAMA_CELL_IDS = {"manifest": 3, "preprocess_fns": 5, "scenarios": 6, "smoke": 7}`

- [ ] **Step 1: Create venv** `python -m venv --system-site-packages "$S/venv"` then `"$S/venv/Scripts/python" -m pip install -q pytest`. Expected: `"$S/venv/Scripts/python" -m pytest --version` prints a version.
- [ ] **Step 2: Write failing tests**
  - `test_cell_ids_match_titles`: cell 5 contains `"TAHAP 2: FUNGSI PREPROCESSING"`, cell 6 contains `"TAHAP 3: SIMPAN SCENARIO"`, cell 7 contains `"TAHAP 4: SMOKE TEST"`, cell 3 contains `"trial_records"`.
  - `test_transform_counts`: for cell 6 the counts are `{"yolo26n_lb6_": 3, "mobilefog_rhama": 0}`, and for cell 7 they are `{"yolo26n_lb6_": 0, "mobilefog_rhama": 1}`. After the transform, `"yolo26n"` does not occur in either.
  - `test_transform_is_minimal`: `difflib.ndiff` between original and transformed cell 6 shows changed lines only where `yolo26n_lb6_` appeared.
  - `test_constants_values`: exec `rhama_constants_block(cell2)` into a dict → `LB6_INDICES == [11,12,13,14,15,16]`, `CONF_VALID_THRES == 0.20`, `HAMPEL_WINDOW == 7`, `HAMPEL_SIGMA == 3.0`, `BUTTER_CUTOFF_HZ == 9.0`, `WINDOW_SEC == 2.0`, `STRIDE_SEC == 1.0`, `RANDOM_SEED == 42`.
  - `test_preprocess_runs_on_133kp`: exec constants + cell 5 (with `import numpy as np, pandas as pd` prelude) and call `build_scenario_a_and_b(kps, fs=30.0)` on a synthetic `kps` of shape (300,133,3) (conf 0.9, smooth sinusoidal legs). The result has shapes `(300,6,7)` and `(300,6,9)` with no NaN.
- [ ] **Step 3: Run** `"$S/venv/Scripts/python" -m pytest sultan_dwpose/tests/test_rhama_cells.py -v`. Expected: FAIL (module missing).
- [ ] **Step 4: Implement** `rhama_cells.py` (json load, `str.count` + `str.replace`, line-based constant extraction by variable-name prefix, keeping the immediately preceding comment lines).
- [ ] **Step 5: Run tests.** Expected: all PASS.
- [ ] **Step 6: Checkpoint.** `git status` shows only new files under `sultan_dwpose/`. Stage nothing.

### Task 2: Patient selection

**Files:**
- Create: `sultan_dwpose/select_person.py`
- Test: `sultan_dwpose/tests/test_select_person.py`

**Interfaces:**
- Produces: `select_main_person(boxes: np.ndarray (N,4) xyxy, kps: np.ndarray (N,133,2), scores: np.ndarray (N,133), W: int, H: int) -> tuple[np.ndarray (133,3) float32, np.ndarray (4,) float32, bool, int]`, returning `(kp_with_score, box, valid, n_persons)`.
- Rule (spec 5.3): `N==0` → zeros, `False`, `0`. `N==1` → that person. Otherwise `score = area × mean(scores[:, 0:17]) × (1 − 0.3·|cx − W/2| / (W/2))`, with argmax.

- [ ] **Step 1: Write failing tests**
  - `test_select_empty`: N=0 → `kp` all 0 with shape (133,3), `valid is False`, `n == 0`.
  - `test_select_single`: N=1 → returns that person's xy and scores, `valid is True`, `n == 1`.
  - `test_select_prefers_central_over_larger_edge`: W=1280. Person A box at x 0–400 (area 1.2× B, body score 0.9). Person B box at 440–840 (centered, body score 0.9). Selection → B, `n == 2`.
  - `test_select_uses_body_scores_only`: two equal central boxes. A has high face/hand scores (17:133) but body scores 0.3, B has body scores 0.8 → B.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `select_main_person`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Checkpoint.** Stage nothing.

### Task 3: Extraction worker

**Files:**
- Create: `sultan_dwpose/extract_worker.py`, a standalone script that the notebook writes to disk. It must import only stdlib, numpy, cv2, rtmlib, and its own inlined `select_main_person` (the build step injects the source).
- Test: `sultan_dwpose/tests/test_extract_worker.py`

**Interfaces:**
- Consumes: `select_main_person` (Task 2).
- Produces:
  - `extract_video(video_path: str, det_fn: Callable[[np.ndarray], np.ndarray], pose_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]], max_frames: int | None = None) -> dict[str, np.ndarray]` with the npz keys from Global Constraints.
  - `save_npz_atomic(path: Path, arrays: dict) -> None`: writes `path.with_suffix(".tmp.npz")` then `os.replace`.
  - `build_models(det_url, pose_url, device) -> tuple[det_fn, pose_fn]` (rtmlib `YOLOX(det_url, model_input_size=(640,640), backend="onnxruntime", device=device)` and `RTMPose(pose_url, model_input_size=(192,256), to_openpose=False, backend="onnxruntime", device=device)`).
  - `main(job_path: str)`: reads `{"gpu_label", "cfg": {"out_dir", "det_url", "pose_url", "max_frames"}, "tasks": [{"split","trial_id","video_path"}]}`, skips existing final npz, and prints `[GPU g] Selesai {split}/{tid}: {T} frame | valid=..% | {ms:.1f} ms/frame`.
  - `__main__` → `main(sys.argv[1])`.

- [ ] **Step 1: Write failing tests** (fake models, synthetic 12-frame 64×48 video written with `cv2.VideoWriter` mp4v at 30 fps in `tmp_path`)
  - `test_extract_shapes_dtypes`: fake det returns 1 box per frame, fake pose returns (1,133,2)/(1,133). Result: `kps.shape == (12,133,3)` float32, `boxes` (12,4) float32, `valid.all()`, `n_persons` int16 all 1, `W == 64`, `H == 48`, `fps` ≈ 30.
  - `test_extract_no_person_frames`: fake det returns an empty (0,4) array on frames 3–5 → `valid[3:6]` False, `kps[3:6]` all 0, `n_persons[3:6] == 0`.
  - `test_extract_max_frames`: `max_frames=5` → T == 5.
  - `test_worker_atomic_and_resume`: `save_npz_atomic` leaves no `*.tmp.npz` and the final file loads. A pre-existing final file is skipped by the main loop: monkeypatch `extract_video` to raise if called, run `main(job)`, expect no exception.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `extract_worker.py`. `pose_fn` is called once per frame with all det boxes (rtmlib `RTMPose(image, bboxes=boxes)` returns `(N,133,2)`, `(N,133)`), and is skipped when N == 0.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Checkpoint.** Stage nothing.

### Task 4: QC metrics

**Files:**
- Create: `sultan_dwpose/qc_metrics.py`
- Test: `sultan_dwpose/tests/test_qc_metrics.py`

**Interfaces:**
- Produces:
  - `leg_length_px(kps_lb6: np.ndarray (T,6,3), valid: np.ndarray) -> float`: median over valid frames of mean(L hip→knee→ankle, R hip→knee→ankle). Returns NaN if no valid frame.
  - `qc_trial(kps: (T,K,3), valid: (T,), boxes: (T,4), n_persons: (T,) | None, fps: float, lb6_idx: list[int], n_label30: int, n_video: int) -> dict`, with keys `n_pose, n_label30, n_video, valid_pct, lowconf_pct_{L_Hip,R_Hip,L_Knee,R_Knee,L_Ankle,R_Ankle}, spike_pct, lr_swap_per_min, person_switch_per_min, multi_person_pct, leg_px, flags` (`flags` is a `;`-joined string of rule names, empty when OK). Only consecutive frame pairs where both frames are valid count toward spike/swap/switch rates.
  - `event_frames(kps, valid, boxes, lb6_idx) -> dict[str, np.ndarray]`: frame indices of spikes, swaps, and switches (used by Task 5).
  - `QC_THRESHOLDS: dict`, copied from Global Constraints.

- [ ] **Step 1: Write failing tests** (synthetic 30 fps, 60 s, legs moving smoothly with leg length 200 px)
  - `test_qc_clean_trial`: all metrics are below thresholds and `flags == ""`.
  - `test_qc_detects_spikes`: shift the L ankle by +150 px (0.75 × leg) on 1% of frames → `spike_pct > 0` and its event frames include those indices.
  - `test_qc_detects_swaps`: swap L/R knee+ankle for one frame at 10 instants in the 60 s trial, with legs ≥ 0.3·leg apart so swaps are detectable → `lr_swap_per_min >= 10` (each one-frame swap yields up to 2 events: in and out) and `"lr_swap" in flags`.
  - `test_qc_detects_person_switch`: move the box to a disjoint region for 5 single frames → `person_switch_per_min > 1` → flagged.
  - `test_qc_flags_frame_mismatch`: `n_label30 = n_pose + 3` → `"frame_mismatch" in flags`.
  - `test_qc_trial_all_invalid`: `valid` all False → no exception, `valid_pct == 0`, `"valid" in flags`, and the rate metrics are 0 (not NaN).
  - `test_qc_lowconf`: L ankle score 0.1 on 20% of frames → `lowconf_pct_L_Ankle ≈ 20` and flagged.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `qc_metrics.py`, vectorised with numpy.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Checkpoint.** Stage nothing.

### Task 5: QC visuals + sampling

**Files:**
- Create: `sultan_dwpose/qc_visual.py`
- Test: `sultan_dwpose/tests/test_qc_visual.py`

**Interfaces:**
- Consumes: `event_frames`, `leg_length_px` (Task 4).
- Produces:
  - `pick_sample_trials(df_fog: pd.DataFrame[trial_id, split, fog_pct], seed: int = 42) -> list[str]`: train/val only. Picks 2 from the top FoG tertile, 2 with 0 < FoG ≤ median, and 2 with FoG == 0.
  - `plot_signals(kps, valid, y_fog30: np.ndarray, events: dict, lb6_idx, title: str, out_png: Path) -> None`: x and y of both ankles (normalised by leg length), FoG shaded, and events as vertical ticks.
  - `clip_windows(n_frames: int, fps: float, y_fog30: np.ndarray, events: dict, seed: int) -> list[int]`: ≤ 2 start frames for 15 s clips (one at the middle of the first FoG episode if any; one at the densest event region, or random), clamped to the video.
  - `render_clip(video_path, kps, boxes, valid, y_fog30, start: int, n: int, out_mp4: Path, lb6_idx) -> int`: returns the number of frames written. It draws COCO-WholeBody body+feet bones (indices 0–22), thicker LB6 bones, the box, a FoG/NORMAL label, time, and mean LB6 score.
  - `write_review_template(path: Path, rows: list[tuple[str, str]]) -> bool`: writes the header `trial_id,clip,checked_by,person_ok,left_right_ok,joints_ok,notes` and one row per clip. If the file already exists it is not overwritten, and the function returns False.

- [ ] **Step 1: Write failing tests**
  - `test_pick_sample_excludes_test_and_is_deterministic`: a frame with splits train/val/test → 6 ids, none from test, and the same result on two calls.
  - `test_clip_windows_bounds`: every start lies in `0 ≤ s ≤ n_frames − 15·fps`. With FoG in frames 900–1200, one start lies within 15 s before 900.
  - `test_render_clip_writes_frames`: on a synthetic 40-frame video with `n=30` → returns 30, and the mp4 opens with 30 frames.
  - `test_plot_signals_creates_png`: the file exists and is > 1 KB.
  - `test_review_template_not_overwritten`: the second call returns False and the content is unchanged.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `qc_visual.py` (matplotlib Agg backend; cv2 mp4v writer).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Checkpoint.** Stage nothing.

### Task 6: Notebook builder + generated notebook

**Files:**
- Create: `sultan_dwpose/build_notebook.py`, `sultan_dwpose/nb_helpers.py`
- Generate: `MobileFOG_Sultan_DWPose_Extraction_QC.ipynb` (repo root)
- Test: `sultan_dwpose/tests/test_build_notebook.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces:
  - `nb_helpers.require_cuda_provider(providers: list[str]) -> None`: raises `RuntimeError` with a message naming `onnxruntime-gpu` and `pip uninstall onnxruntime` when `"CUDAExecutionProvider"` is missing.
  - `nb_helpers.find_dataset_root(candidates: list[Path]) -> Path`: the first dir containing `train/videos` and `val/videos`, searched recursively under each candidate.
  - `nb_helpers.restore_previous_outputs(input_root: Path, raw_dir: Path) -> int`: copies `*.npz` found under any `/kaggle/input/**/01_extracted_poses_raw/dwpose_m_sultan/{split}/` into `raw_dir/{split}/` when not already present. Returns the count copied.
  - `nb_helpers.eta_hours(ms_per_frame: float, total_frames: int, n_gpus: int) -> float`.
  - `build(out_path: Path) -> None`. The cell order (markdown headers in Indonesian, matching Rhama's style) is:
    0. Markdown title and pipeline summary (DWPose-m, 30 fps, A/B/D identical to Rhama, QC report-only).
    1. Install: `!pip install -q --no-deps rtmlib==0.0.16`, `!pip install -q onnxruntime-gpu`, then imports, `require_cuda_provider(ort.get_available_providers())`, and printed GPU info.
    2. Config: `DATASET_ROOT` via `find_dataset_root([Path("/kaggle/input"), Path("dataset"), Path("Data/Train Val Test")])`, `OUTPUT_ROOT`, `RAW_POSE_DIR = OUTPUT_ROOT/"01_extracted_poses_raw"/"dwpose_m_sultan"`, `DET_URL`, `POSE_URL`, `RUN_SPLITS = ["train","val","test"]`, `MAX_FRAMES = None`, followed by `rhama_constants_block(cell2)`.
    3. Rhama cell 3 (manifest audit), transformed. `df_manifest` is filtered by `RUN_SPLITS` only for extraction, never for preprocessing.
    4. `nb_helpers` source inlined + `restore_previous_outputs` + `joint_schema.json` write.
    5. `%%writefile` worker: `select_person.py` + `extract_worker.py` sources concatenated into `OUTPUT_ROOT/_dwpose_extract_worker.py`.
    6. Benchmark: run the worker in-process on the first pending trial with `max_frames=300`, print `ms/frame` and `eta_hours`, and if it is > 9, print the instruction to set `RUN_SPLITS`. The benchmark output is not saved as a trial.
    7. Parallel launch (Rhama's pattern): shards `tasks[i::n_gpus]`, `CUDA_VISIBLE_DEVICES`, `assert rc == 0`, total time printed. Models are pre-downloaded in the main process via `build_models(..., device="cpu")` before the workers start.
    8. Guard: `assert` that all 71 raw npz exist before preprocessing (with a message to rerun with the remaining splits).
    9–11. Rhama cells 5, 6, 7, transformed. After cell 6, print the window counts per split next to Rhama's (train 5535 / val 1169 / test 1652).
    12. QC: `qc_metrics` + `qc_visual` sources inlined, `qc_trial` for all 71 trials → `03_qc/qc_per_trial.csv`, a display of flagged trials, metric histograms, signal PNGs for all trials, clips for flagged trials ∪ 6 sample trials, and `qc_manual_review.csv`.
    13. Markdown "Langkah berikutnya": download `03_qc/clips` + CSV; point Rhama's training notebook `DATA_DIR` to `mobilefog_sultan`.
  - CLI: `python sultan_dwpose/build_notebook.py` writes the notebook to the repo root.

- [ ] **Step 1: Write failing tests**
  - `test_require_cuda_provider_raises`: `["CPUExecutionProvider"]` → `RuntimeError` whose message contains `"onnxruntime-gpu"`. With `["CUDAExecutionProvider","CPUExecutionProvider"]` → no error.
  - `test_find_dataset_root`: tmp tree `a/b/Train Val Test/{train,val}/videos` → returns that dir. Missing → `FileNotFoundError`.
  - `test_restore_previous_outputs`: 2 npz under `input/x/01_extracted_poses_raw/dwpose_m_sultan/train/`, 1 already in raw_dir → returns 1.
  - `test_eta_hours`: `eta_hours(50.0, 256000, 2) ≈ 1.78`.
  - `test_notebook_structure`: build to tmp → valid nbformat-4 JSON. Every code cell, after stripping lines starting with `!` or `%%`, passes `ast.parse`. The worker cell starts with `%%writefile`. No code cell contains `yolo26n`, `Wholebody(`, or `mode=`. Both model URLs are present. The cells for Rhama 5/6/7 equal `transform_rhama_source(original)[0]`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** `nb_helpers.py` and `build_notebook.py` (plain `json`, nbformat 4.5 dict, kernelspec python3).
- [ ] **Step 4: Run the full suite** `"$S/venv/Scripts/python" -m pytest sultan_dwpose/tests -v`. Expected: all PASS.
- [ ] **Step 5: Generate** with `python sultan_dwpose/build_notebook.py`. Expected: the notebook exists at the repo root.
- [ ] **Step 6: Checkpoint.** Stage nothing.

### Task 7: Local real-model smoke test (CPU, isolated venv)

**Files:**
- Create: `sultan_dwpose/tests/test_smoke_real_models.py` (marked `@pytest.mark.real_models`, skipped unless `RUN_REAL_MODELS=1`)

- [ ] **Step 1: Install into the venv only**: `"$S/venv/Scripts/python" -m pip install -q --no-deps rtmlib==0.0.16` and `"$S/venv/Scripts/python" -m pip install -q onnxruntime tqdm`. This is CPU onnxruntime, used for the local smoke test only.
- [ ] **Step 2: Write the test** `test_real_dwpose_on_one_video`: `build_models(DET_URL, POSE_URL, "cpu")`, then `extract_video("dataset/train/videos/PDFE02_1.mp4", ..., max_frames=60)`. Asserts: `kps.shape == (60,133,3)`, `valid.mean() > 0.9`, and the median LB6 score > 0.3. Then `qc_trial(...)` runs and `render_clip` writes ≥ 30 frames to `$S/smoke_clip.mp4`.
- [ ] **Step 3: Run** `RUN_REAL_MODELS=1 "$S/venv/Scripts/python" -m pytest sultan_dwpose/tests/test_smoke_real_models.py -v -s`. Expected: PASS. This verifies the model URLs, the rtmlib API, and the output shapes.
- [ ] **Step 4: Look at the clip** (extract 3 frames to PNG and view them) to confirm the skeleton sits on the patient's legs. Report what was seen.
- [ ] **Step 5: Checkpoint.** `git status` shows only `sultan_dwpose/`, the notebook, and `docs/`. Stage nothing. Report to Sultan that it's ready to upload to Kaggle.
