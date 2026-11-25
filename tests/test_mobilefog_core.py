"""Unit tests for core Mobile-FOG path configuration and evaluation metrics."""

from __future__ import annotations

import csv
import glob
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mobilefog import metrics, paths


class TestMobileFogPaths(unittest.TestCase):
    def test_tracked_paths_exist(self):
        self.assertTrue(paths.REPO_ROOT.exists())
        self.assertTrue(paths.METADATA_CSV.is_file())
        self.assertTrue(paths.NOTEBOOKS_DIR.is_dir())
        self.assertTrue(paths.RESULTS_METRICS_DIR.is_dir())

        expected_csvs = [
            "baseline_35subj_results_summary.csv",
            "bbox_35subj_results_summary.csv",
            "clean_32subj_results_summary.csv",
            "clean_32subj_loso_oof_B_15hz.csv",
            "clean_32subj_loso_per_subject_B_15hz.csv",
        ]
        for fname in expected_csvs:
            self.assertTrue((paths.RESULTS_METRICS_DIR / fname).is_file(), fname)


class TestMobileFogMetrics(unittest.TestCase):
    def test_min_duration_filter_per_trial(self):
        y_pred = np.array([1, 0, 1, 1, 0, 1, 1])
        trials = np.array(["T1", "T1", "T1", "T1", "T1", "T1", "T2"])
        filtered = metrics.apply_min_duration_filter(y_pred, trials, min_run=2)
        np.testing.assert_array_equal(filtered, np.array([0, 0, 1, 1, 0, 0, 0]))

    def test_perfect_roc_pr_curves(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        fpr, tpr, roc_auc, rec, prec, pr_auc = metrics.compute_roc_pr_curves(y_true, y_score)
        self.assertAlmostEqual(roc_auc, 1.0, places=6)
        self.assertAlmostEqual(pr_auc, 1.0, places=6)
        self.assertEqual(fpr[0], 0.0)
        self.assertEqual(tpr[-1], 1.0)
        self.assertEqual(rec[0], 0.0)
        self.assertEqual(prec[0], 1.0)

    def test_binary_classification_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])
        res = metrics.compute_binary_classification_metrics(y_true, y_pred)
        self.assertEqual(res["tn"], 2)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        self.assertEqual(res["tp"], 2)
        self.assertAlmostEqual(res["spec"], 2 / 3, places=6)
        self.assertAlmostEqual(res["sens"], 2 / 3, places=6)
        self.assertAlmostEqual(res["macro_f1"], 2 / 3, places=6)
        self.assertAlmostEqual(res["err_rate"], 2 / 6, places=6)

    def test_loso_oof_alignment_when_outputs_present(self):
        npz_glob = str(paths.TRAINING_CLEAN_DIR / "mobilefog_training_results/loso_35fold_B_15hz/fold_*.npz")
        npz_files = sorted(glob.glob(npz_glob))
        if not npz_files:
            self.skipTest("Local LOSO fold npz outputs not present")

        y_oof = np.concatenate([np.load(p)["y_te"].astype(np.int64) for p in npz_files])
        p_stgcn = np.concatenate([np.load(p)["p_stgcn"].astype(np.float64) for p in npz_files])
        _, _, roc_auc, _, _, pr_auc = metrics.compute_roc_pr_curves(y_oof, p_stgcn)

        with open(paths.RESULTS_METRICS_DIR / "clean_32subj_loso_oof_B_15hz.csv") as f:
            oof_csv = {r["model"]: r for r in csv.DictReader(f)}

        self.assertAlmostEqual(roc_auc, float(oof_csv["Light_ST-GCN"]["oof_roc_auc"]), places=3)
        self.assertAlmostEqual(pr_auc, float(oof_csv["Light_ST-GCN"]["oof_pr_auc"]), places=3)


if __name__ == "__main__":
    unittest.main()
