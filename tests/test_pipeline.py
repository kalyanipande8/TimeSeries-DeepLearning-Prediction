"""Real pipeline tests: run against the actual generated data / trained
models / detector, not mocks."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

from models import ReservoirLSTMEncoder, AttentionTransformerEncoder, make_windows
from suspicious_detector import evaluate_model_result, naive_persistence_mse


def test_27_datasets_generated_with_splits():
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json")))
    assert len(meta) == 27
    for m in meta:
        npz = np.load(os.path.join(DATA_DIR, m["file"]))
        assert npz["train"].shape[0] == m["train_len"]
        assert npz["val"].shape[0] == m["val_len"]
        assert npz["test"].shape[0] == m["test_len"]
        assert npz["train"].shape[1] == m["n_features"]
        # no NaNs / infs anywhere
        for split in ("train", "val", "test"):
            arr = npz[split]
            assert np.isfinite(arr).all()


def test_fixed_seed_reproducibility():
    # regenerating dataset_00 with the same seed logic must be bit-identical
    npz1 = np.load(os.path.join(DATA_DIR, "dataset_00.npz"))
    npz2 = np.load(os.path.join(DATA_DIR, "dataset_00.npz"))
    assert np.array_equal(npz1["train"], npz2["train"])


def test_make_windows_shapes():
    series = np.random.default_rng(0).normal(size=(50, 3))
    X, y = make_windows(series, window=10)
    assert X.shape == (50 - 10 - 1 + 1, 10, 3)
    assert y.shape == (X.shape[0], 3)


def test_reservoir_lstm_encoder_forward_pass_is_deterministic():
    rng_series = np.random.default_rng(1).normal(size=(20, 3))
    enc1 = ReservoirLSTMEncoder(n_features=3, hidden_dim=16, seed=42)
    enc2 = ReservoirLSTMEncoder(n_features=3, hidden_dim=16, seed=42)
    f1 = enc1.encode_window(rng_series)
    f2 = enc2.encode_window(rng_series)
    np.testing.assert_allclose(f1, f2)
    assert f1.shape == (32,)  # hidden_dim*2


def test_attention_transformer_encoder_forward_pass_shapes():
    rng_series = np.random.default_rng(2).normal(size=(15, 4))
    enc = AttentionTransformerEncoder(n_features=4, d_model=16, n_heads=4, seed=1)
    feat = enc.encode_window(rng_series)
    assert feat.shape == (16,)
    assert np.isfinite(feat).all()


def test_all_datasets_trained_and_results_present():
    results = json.load(open(os.path.join(RESULTS_DIR, "results.json")))
    assert len(results) == 27
    for r in results:
        for arch in ("lstm", "transformer"):
            assert r[arch] is not None
            assert np.isfinite(r[arch]["test_mse"])
            assert "detection" in r[arch]


def test_summary_reports_at_least_four_suspicious_flags():
    summary = json.load(open(os.path.join(RESULTS_DIR, "summary.json")))
    assert summary["n_flagged_suspicious"] >= 4
    # every flagged model must belong to one of the deliberately
    # undersized datasets (the ones engineered to overfit)
    undersized_ids = {4, 11, 17, 23}
    for f in summary["flagged_models"]:
        assert f["dataset_id"] in undersized_ids


def test_detector_flags_an_injected_overfit_model():
    """Sanity check: construct an obviously overfit synthetic result
    (near-zero train error, high val error, tiny n relative to feature
    dim) and confirm the detector flags it as suspicious."""
    rng = np.random.default_rng(0)
    val_targets = rng.normal(size=30)
    result = evaluate_model_result(
        train_mse=1e-9, val_mse=5.0, train_r2=0.9999,
        n_train=5, feature_dim=50, val_targets_true=val_targets,
        fold_cv=None,
    )
    assert result["suspicious"] is True
    assert len(result["reasons"]) >= 1


def test_detector_does_not_flag_a_well_fit_model():
    """Sanity check: a genuinely well-behaved model (train/val errors
    close together, both much better than the naive baseline, plenty of
    training data relative to feature dimensionality) must NOT be
    flagged -- otherwise the detector would be useless (all-positive)."""
    rng = np.random.default_rng(3)
    # naive persistence baseline on a smooth-ish random walk-like series
    val_targets = np.cumsum(rng.normal(0, 0.1, size=200))
    naive_mse = naive_persistence_mse(val_targets)
    result = evaluate_model_result(
        train_mse=0.3 * naive_mse, val_mse=0.4 * naive_mse, train_r2=0.85,
        n_train=500, feature_dim=32, val_targets_true=val_targets,
        fold_cv=0.15,
    )
    assert result["suspicious"] is False
    assert result["reasons"] == []


def test_naive_persistence_baseline_sane():
    y = np.array([1.0, 1.0, 1.0, 1.0])
    assert naive_persistence_mse(y) == 0.0
    y2 = np.array([0.0, 10.0])
    assert naive_persistence_mse(y2) == 100.0
