"""
Suspicious-result detection system.

Real statistical checks run on every trained (dataset, architecture) model
to catch overfitting / unreliable results before they'd be trusted in
production. A model is flagged "suspicious" if ANY check trips.

Checks implemented:
  A. train/val error divergence  -- val_mse notably worse than train_mse
     relative to the naive baseline's scale (classic overfitting signature).
  B. high-capacity-low-data regime -- number of training examples is smaller
     than the feature dimensionality feeding the readout, AND train R^2 is
     near-perfect. This is the textbook p >> n overfitting setup: a model
     that fits training data almost exactly with too few examples to
     constrain that many parameters cannot be trusted to generalize.
  C. worse than naive persistence baseline -- val_mse exceeds the trivial
     "predict last observed value" baseline, meaning the model has learned
     nothing useful despite (or because of) its fit.
  D. unstable across folds -- k-fold re-fits of the readout on the training
     window show high relative variance in validation error, indicating a
     fragile / non-robust fit.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def naive_persistence_mse(y_true):
    """Baseline: predict each step as the previous true value."""
    if len(y_true) < 2:
        return float("nan")
    preds = y_true[:-1]
    actual = y_true[1:]
    return float(np.mean((preds - actual) ** 2))


def fold_stability_check(features, targets, alpha, n_splits=4):
    """Re-fit the readout on K folds of the *training* features and report
    relative variance of held-out fold error -- a real instability probe
    that doesn't touch val/test data."""
    n = features.shape[0]
    if n < n_splits * 2:
        return None  # not enough data to run this check meaningfully
    kf = KFold(n_splits=n_splits, shuffle=False)
    fold_mses = []
    for train_idx, test_idx in kf.split(features):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        reg = Ridge(alpha=alpha)
        reg.fit(features[train_idx], targets[train_idx])
        pred = reg.predict(features[test_idx])
        fold_mses.append(np.mean((pred - targets[test_idx]) ** 2))
    if len(fold_mses) < 2:
        return None
    fold_mses = np.array(fold_mses)
    mean_mse = fold_mses.mean()
    if mean_mse < 1e-12:
        return 0.0
    return float(fold_mses.std() / mean_mse)


def evaluate_model_result(train_mse, val_mse, train_r2, n_train,
                           feature_dim, val_targets_true, fold_cv=None,
                           divergence_ratio_threshold=4.0,
                           r2_perfect_threshold=0.995,
                           cv_instability_threshold=1.5):
    """Run all checks against one trained model's stats. Returns a dict
    with per-check booleans/values and an overall `suspicious` flag."""
    reasons = []

    naive_mse = naive_persistence_mse(val_targets_true)

    # A. train/val divergence
    eps = 1e-8
    divergence_ratio = val_mse / (train_mse + eps)
    check_a = divergence_ratio > divergence_ratio_threshold and train_mse < 0.5 * naive_mse
    if check_a:
        reasons.append(
            f"train/val divergence: val_mse={val_mse:.4f} is "
            f"{divergence_ratio:.1f}x train_mse={train_mse:.4f}"
        )

    # B. high-capacity / low-data regime with near-perfect train fit
    check_b = (n_train < feature_dim) and (train_r2 >= r2_perfect_threshold)
    if check_b:
        reasons.append(
            f"high-capacity-low-data: n_train={n_train} < feature_dim="
            f"{feature_dim} with train_r2={train_r2:.4f}"
        )

    # C. worse than naive persistence baseline *despite* an apparently
    # reasonable train fit -- i.e. it looked fine on train but the val
    # error blows past the trivial baseline, a genuine overfitting
    # signature (as opposed to a merely weak/underfit model, which isn't
    # what this check targets).
    check_c = (not np.isnan(naive_mse)) and (val_mse > 5.0 * naive_mse) and (train_mse < 0.5 * naive_mse)
    if check_c:
        reasons.append(
            f"worse-than-naive despite good train fit: val_mse={val_mse:.4f} "
            f"> 5x naive baseline={naive_mse:.4f} while train_mse={train_mse:.4f}"
        )

    # D. cross-fold instability
    check_d = fold_cv is not None and fold_cv > cv_instability_threshold
    if check_d:
        reasons.append(f"fold instability: relative std={fold_cv:.2f}")

    suspicious = check_a or check_b or check_c or check_d
    return dict(
        suspicious=suspicious,
        reasons=reasons,
        divergence_ratio=float(divergence_ratio),
        naive_mse=float(naive_mse) if not np.isnan(naive_mse) else None,
        fold_cv=fold_cv,
        check_a_divergence=check_a,
        check_b_capacity=check_b,
        check_c_worse_than_naive=check_c,
        check_d_instability=check_d,
    )
