"""
Generate 27 synthetic multivariate time-series datasets with varying
seasonality / trend / noise / regime-shift characteristics.

Fixed seed -> fully reproducible. Each dataset gets its own npz file with
train/val/test splits (chronological, no shuffling, as is correct for
time series).
"""
import numpy as np
import os
import json

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

MASTER_SEED = 20240517
N_DATASETS = 27
N_FEATURES_CHOICES = [3, 4, 5]
T_CHOICES = [600, 800, 1000]

# indices deliberately given a tiny train split later (in train_models.py)
# to induce genuine overfitting for the suspicious-result detector demo.
UNDERSIZED_DATASET_IDS = {4, 11, 17, 23}


def make_series(rng, T, n_features, regime_shift, seasonality_strength,
                 trend_strength, noise_level):
    t = np.arange(T)
    X = np.zeros((T, n_features))
    for f in range(n_features):
        period = rng.choice([12, 24, 30, 50, 7])
        phase = rng.uniform(0, 2 * np.pi)
        freq2 = rng.choice([4, 8, 15])
        seasonal = seasonality_strength * (
            np.sin(2 * np.pi * t / period + phase)
            + 0.4 * np.sin(2 * np.pi * t / freq2 + phase / 2)
        )
        trend = trend_strength * (t / T) * rng.uniform(0.5, 1.5)
        if rng.uniform() < 0.5:
            trend = -trend
        noise = rng.normal(0, noise_level, size=T)
        base = 5.0 + seasonal + trend + noise
        if regime_shift:
            shift_point = rng.integers(int(T * 0.4), int(T * 0.7))
            shift_mag = rng.uniform(2, 5) * rng.choice([-1, 1])
            base[shift_point:] += shift_mag
            # also perturb the seasonal period post-shift (regime change)
            base[shift_point:] += 0.5 * seasonality_strength * np.sin(
                2 * np.pi * t[shift_point:] / (period * 1.7) + phase
            )
        X[:, f] = base
    # inject light cross-feature coupling so it's genuinely multivariate
    if n_features > 1:
        for f in range(1, n_features):
            X[:, f] += 0.15 * X[:, f - 1]
    return X


def split_series(X, train_frac=0.7, val_frac=0.15):
    T = X.shape[0]
    n_train = int(T * train_frac)
    n_val = int(T * val_frac)
    train = X[:n_train]
    val = X[n_train:n_train + n_val]
    test = X[n_train + n_val:]
    return train, val, test


def generate_all():
    meta = []
    master_rng = np.random.default_rng(MASTER_SEED)
    for i in range(N_DATASETS):
        seed = MASTER_SEED + 1000 * (i + 1)
        rng = np.random.default_rng(seed)
        T = int(master_rng.choice(T_CHOICES))
        n_features = int(master_rng.choice(N_FEATURES_CHOICES))
        regime_shift = bool(master_rng.uniform() < 0.4)
        seasonality_strength = float(master_rng.uniform(0.5, 3.0))
        trend_strength = float(master_rng.uniform(0.0, 4.0))
        noise_level = float(master_rng.uniform(0.15, 1.2))

        X = make_series(rng, T, n_features, regime_shift,
                         seasonality_strength, trend_strength, noise_level)

        # normalize per-feature using train-portion stats only (no leakage)
        n_train_raw = int(T * 0.7)
        mu = X[:n_train_raw].mean(axis=0)
        sigma = X[:n_train_raw].std(axis=0) + 1e-8
        Xn = (X - mu) / sigma

        train, val, test = split_series(Xn)

        undersized = i in UNDERSIZED_DATASET_IDS
        if undersized:
            # deliberately shrink the usable train set to a handful of
            # points (relative to a high-capacity model) to induce real
            # overfitting later -- this is intentional, documented, and
            # used to validate the suspicious-result detector.
            train = train[-20:]

        fname = f"dataset_{i:02d}.npz"
        np.savez(os.path.join(OUT_DIR, fname),
                 train=train, val=val, test=test, mu=mu, sigma=sigma)

        meta.append(dict(
            id=i, file=fname, T=T, n_features=n_features,
            regime_shift=regime_shift,
            seasonality_strength=round(seasonality_strength, 3),
            trend_strength=round(trend_strength, 3),
            noise_level=round(noise_level, 3),
            train_len=int(train.shape[0]), val_len=int(val.shape[0]),
            test_len=int(test.shape[0]), undersized=undersized,
        ))

    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    meta = generate_all()
    print(f"Generated {len(meta)} datasets into {OUT_DIR}")
    for m in meta:
        print(m)
