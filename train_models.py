"""
Train + evaluate both encoder architectures (ReservoirLSTMEncoder,
AttentionTransformerEncoder) on all 27 datasets, run the suspicious-result
detector on every trained model, and write real per-dataset results.
"""
import json
import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import ReservoirLSTMEncoder, AttentionTransformerEncoder, make_windows
from suspicious_detector import evaluate_model_result, fold_stability_check

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW = 12
SEED = 20240517

# datasets deliberately undersized in generate_datasets.py (train_len=20)
UNDERSIZED_IDS = {4, 11, 17, 23}
# high-capacity encoder settings used ONLY on the undersized datasets to
# deliberately induce the p >> n overfitting regime the detector should
# catch (documented, intentional -- see README).
OVERFIT_HIDDEN_DIM = 64
OVERFIT_DMODEL = 64
NORMAL_HIDDEN_DIM = 24
NORMAL_DMODEL = 16
NORMAL_ALPHA = 5.0
OVERFIT_ALPHA = 1e-6  # near-zero ridge penalty -> genuine overfitting


def load_dataset(idx):
    meta = json.load(open(os.path.join(DATA_DIR, "metadata.json")))[idx]
    npz = np.load(os.path.join(DATA_DIR, meta["file"]))
    return meta, npz["train"], npz["val"], npz["test"]


def run_one(encoder_cls, encoder_kwargs, alpha, train, val, test, dataset_id):
    n_features = train.shape[1]
    Xtr_w, ytr = make_windows(train, WINDOW)
    Xval_w, yval = make_windows(val, WINDOW)
    Xtest_w, ytest = make_windows(test, WINDOW)

    if len(Xtr_w) < 3 or len(Xval_w) < 1:
        return None

    encoder = encoder_cls(n_features, seed=SEED + dataset_id, **encoder_kwargs)
    Ftr = encoder.transform(Xtr_w)
    Fval = encoder.transform(Xval_w)
    Ftest = encoder.transform(Xtest_w)

    reg = Ridge(alpha=alpha)
    reg.fit(Ftr, ytr)

    pred_tr = reg.predict(Ftr)
    pred_val = reg.predict(Fval)
    pred_test = reg.predict(Ftest)

    train_mse = mean_squared_error(ytr, pred_tr)
    val_mse = mean_squared_error(yval, pred_val)
    test_mse = mean_squared_error(ytest, pred_test) if len(ytest) else float("nan")
    test_mae = mean_absolute_error(ytest, pred_test) if len(ytest) else float("nan")
    train_r2 = r2_score(ytr, pred_tr) if len(ytr) > 1 else float("nan")
    test_r2 = r2_score(ytest, pred_test) if len(ytest) > 1 else float("nan")

    fold_cv = fold_stability_check(Ftr, ytr, alpha)

    detection = evaluate_model_result(
        train_mse=train_mse, val_mse=val_mse, train_r2=train_r2,
        n_train=Ftr.shape[0], feature_dim=Ftr.shape[1],
        val_targets_true=yval, fold_cv=fold_cv,
    )

    return dict(
        train_mse=float(train_mse), val_mse=float(val_mse),
        test_mse=float(test_mse), test_mae=float(test_mae),
        train_r2=float(train_r2), test_r2=float(test_r2),
        n_train=int(Ftr.shape[0]), feature_dim=int(Ftr.shape[1]),
        detection=detection,
    )


def main():
    meta_all = json.load(open(os.path.join(DATA_DIR, "metadata.json")))
    all_results = []

    for m in meta_all:
        idx = m["id"]
        _, train, val, test = load_dataset(idx)
        undersized = idx in UNDERSIZED_IDS

        lstm_kwargs = dict(hidden_dim=OVERFIT_HIDDEN_DIM if undersized else NORMAL_HIDDEN_DIM,
                            spectral_radius=0.9)
        lstm_alpha = OVERFIT_ALPHA if undersized else NORMAL_ALPHA
        lstm_res = run_one(ReservoirLSTMEncoder, lstm_kwargs, lstm_alpha,
                            train, val, test, idx)

        tf_kwargs = dict(d_model=OVERFIT_DMODEL if undersized else NORMAL_DMODEL,
                          n_heads=4)
        tf_alpha = OVERFIT_ALPHA if undersized else NORMAL_ALPHA
        tf_res = run_one(AttentionTransformerEncoder, tf_kwargs, tf_alpha,
                          train, val, test, idx)

        all_results.append(dict(dataset_id=idx, meta=m,
                                 lstm=lstm_res, transformer=tf_res))
        print(f"dataset {idx:02d} (undersized={undersized}): "
              f"LSTM val_mse={lstm_res['val_mse']:.4f} suspicious={lstm_res['detection']['suspicious']} | "
              f"Transformer val_mse={tf_res['val_mse']:.4f} suspicious={tf_res['detection']['suspicious']}")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- summary ----
    lstm_test_mses = [r["lstm"]["test_mse"] for r in all_results]
    tf_test_mses = [r["transformer"]["test_mse"] for r in all_results]
    lstm_test_r2s = [r["lstm"]["test_r2"] for r in all_results]
    tf_test_r2s = [r["transformer"]["test_r2"] for r in all_results]

    flagged = []
    for r in all_results:
        if r["lstm"]["detection"]["suspicious"]:
            flagged.append((r["dataset_id"], "LSTM", r["lstm"]["detection"]["reasons"]))
        if r["transformer"]["detection"]["suspicious"]:
            flagged.append((r["dataset_id"], "Transformer", r["transformer"]["detection"]["reasons"]))

    summary = dict(
        n_datasets=len(all_results),
        lstm_avg_test_mse=float(np.mean(lstm_test_mses)),
        lstm_avg_test_r2=float(np.mean(lstm_test_r2s)),
        transformer_avg_test_mse=float(np.mean(tf_test_mses)),
        transformer_avg_test_r2=float(np.mean(tf_test_r2s)),
        n_models_total=2 * len(all_results),
        n_flagged_suspicious=len(flagged),
        flagged_models=[dict(dataset_id=d, arch=a, reasons=r) for d, a, r in flagged],
    )
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
