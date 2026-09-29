# Time-Series Deep Learning for Sequential Prediction — Transformer + LSTM, 27 Synthetic Datasets

A small, fully runnable, real-training implementation built end-to-end on
synthetic data, in support of the resume bullet:

> Time-Series Deep Learning for Sequential Prediction | PyTorch, Transformers
> Developed Transformer and LSTM models for multivariate time-series
> prediction; validated robustness across 27 datasets. Implemented
> suspicious-result detection system catching overfitting in 4 models
> before production deployment.

## What this actually is (read this first)

This is a real, from-scratch, laptop-CPU-scale pipeline. It is **not** a
statistical simulation of results — every number in the Results section
below came from an actual run of `run_all.py` in this repo.

1. `generate_datasets.py` synthesizes **27 multivariate time-series
   datasets** (3–5 features each, 600–1000 timesteps) from a fixed master
   seed, with independently randomized seasonality strength, trend
   strength, noise level, and (for ~40% of them) a mid-series regime
   shift (a step change plus a change in the seasonal period). Each
   dataset gets a chronological train/val/test split (70/15/15, no
   shuffling — shuffling a time series before splitting would leak
   future information into training).
2. `models.py` implements two real sequence-encoder architectures in raw
   numpy (see "Implementation path" below for why numpy and not PyTorch).
3. `train_models.py` trains + evaluates both architectures on **all 27
   datasets** (54 total trained models) with genuine per-dataset
   MSE/MAE/R² on held-out test data.
4. `suspicious_detector.py` runs four real statistical checks against
   every trained model and flags the ones that look overfit/unreliable.
5. `run_all.py` runs the full pipeline end-to-end and produces the
   figures below. `tests/test_pipeline.py` has real assertions against
   the actual generated data, actual trained-model outputs, and the
   detector's actual decisions (including an injected synthetic overfit
   case and an injected synthetic well-fit case, to confirm the detector
   correctly discriminates between the two).

### Implementation path: numpy/scikit-learn, not PyTorch — and why

The task spec asked to use PyTorch if available. It was checked first,
the same way the sibling projects in this `Career/projects` tree
(`BiologicalTransformerArchitecture`, `BioFoundationModelPostTraining`,
`BioModelRobustnessFramework`) document doing:

```
$ python3 -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'
$ df -h /
Filesystem      Size  Used  Avail Capacity  Mounted on
/dev/disk3s3s1  228Gi   12Gi  288Mi    98%   /
```

PyTorch was not installed, and free disk space was **under 300 MiB** at
the time of the check (it fluctuated between roughly 1–4 GB free over the
course of the session on this machine) — not safely enough headroom to
install a CPU wheel (150–200+ MB for the wheel alone, plus pip's
build/download cache and the full dependency closure) without risking
filling the disk. This is the same constraint the sibling projects
independently hit and took the same documented fallback path for.

`numpy`, `scikit-learn`, and `matplotlib` were already available (via the
Anaconda Python distribution on this machine), so this project follows
the fallback explicitly permitted by the task spec:

- The **architecture** — what each encoder computes and how information
  flows through the sequence window — is genuinely hand-implemented in
  raw numpy in `models.py`. No autodiff library is used anywhere.
- The **LSTM-style encoder** (`ReservoirLSTMEncoder`) is architecturally
  an **Echo State Network**: a real, published class of recurrent neural
  network in which the recurrent weight matrix is fixed (spectral-radius
  normalized so the dynamical system is stable) and only a downstream
  linear readout is trained. This is a legitimate, real recurrent
  architecture family, not a from-scratch simulation of an LSTM's gates.
- The **Transformer-style encoder** (`AttentionTransformerEncoder`)
  computes genuine multi-head scaled dot-product self-attention with
  sinusoidal positional encoding over each input window — the real
  attention mechanism, hand-implemented — with fixed random Q/K/V/output
  projection weights.
- In both cases, the trainable part is a `Ridge` regression readout
  fit on the encoder's pooled output features. This is the documented
  simplification the task spec allows in place of hand-deriving full
  backpropagation-through-time / backpropagation-through-softmax-
  attention — the same simplification pattern used by the sibling
  projects (frozen feature extractor + trained readout head). The
  learning that happens (fitting the readout, including the
  intentionally under-regularized fits used for the overfitting demo
  below) is real least-squares fitting on real features, not a mocked
  number.

## Suspicious-result detection system

`suspicious_detector.py` runs four independent real statistical checks
against every one of the 54 trained (dataset × architecture) models. A
model is flagged **suspicious** if any check trips:

| Check | What it tests |
|---|---|
| A. train/val divergence | `val_mse` is >4x `train_mse` while `train_mse` is well below the naive baseline — the classic overfitting signature |
| B. high-capacity/low-data | training examples < feature dimensionality AND train R² ≥ 0.995 — the textbook *p ≫ n* regime where a near-perfect train fit cannot be trusted |
| C. worse than naive baseline despite good train fit | `val_mse` > 5x the trivial "predict-the-last-value" persistence baseline while `train_mse` looked fine — model memorized train, learned nothing generalizable |
| D. cross-fold instability | re-fitting the readout on 4 folds of the *training* window shows relative std of fold error > 1.5 — a fragile, non-robust fit |

To validate the detector actually works (rather than just always/never
firing), 4 of the 27 datasets (ids **4, 11, 17, 23**) were deliberately
given an undersized training split (20 points, vs. 420–700 for the other
23) while the encoders were still run at normal/higher hidden-dimension
capacity — a real, common way overfitting gets introduced in practice
(too little data for the model's capacity). This is documented, not
hidden.

**Result of an actual run:** the detector flagged **8 of the 54 trained
models suspicious — exactly the LSTM and Transformer model trained on
each of the 4 deliberately undersized datasets, and nothing else.** All
46 models trained on the 23 normally-sized datasets passed clean (not
flagged). `tests/test_pipeline.py::test_summary_reports_at_least_four_suspicious_flags`
asserts at least 4 are caught and that every flag traces back to one of
the 4 undersized datasets; a separate pair of injected-case tests
(`test_detector_flags_an_injected_overfit_model` /
`test_detector_does_not_flag_a_well_fit_model`) confirms the detector
correctly discriminates an obviously-overfit synthetic result from an
obviously-well-fit one.

## How to run

```bash
pip install -r requirements.txt
python3 run_all.py        # generate 27 datasets -> train/eval both archs -> detector -> figures
pytest -q                 # run the test suite
```

Individual stages can also be run directly: `generate_datasets.py`,
`train_models.py`.

## Results (real numbers from an actual run of `run_all.py`)

54 models trained total (27 datasets × 2 architectures), window size 12,
next-step multivariate prediction, metrics on held-out **test** splits.

### Averaged across the 23 normally-sized datasets (excludes the 4 deliberately undersized overfit-demo datasets)

| Architecture | Avg test MSE | Avg test R² |
|---|---|---|
| LSTM (Reservoir) encoder | 0.507 | **0.341** |
| Transformer (attention) encoder | 0.891 | -0.094 |

### Averaged across all 27 datasets (including the 4 intentionally-undersized overfit cases, which drag the average down — included for transparency, not cherry-picked out)

| Architecture | Avg test MSE | Avg test R² |
|---|---|---|
| LSTM (Reservoir) encoder | 0.733 | 0.048 |
| Transformer (attention) encoder | 1.658 | -1.146 |

**Honest read of these numbers:** the reservoir-style LSTM encoder
consistently outperforms the fixed-random-attention Transformer encoder
on this benchmark (R² 0.341 vs. -0.094 on the normal datasets) — with
untrained attention weights, the Transformer's feature representation is
noisier for a linear readout to exploit than the LSTM's fixed dynamical
reservoir state, especially at this small per-dataset scale (a few
hundred training windows). This is a genuine, unflattering finding, not
smoothed over — real R² values for individual datasets range from about
-0.2 to +0.89 (dataset 25, LSTM) depending on how much of that dataset's
signal (seasonality/trend) versus irreducible noise there is to capture,
which is exactly the noise/signal variation the 27-dataset design was
built to probe.

### Suspicious-result detection

- **54 models trained total; 8 flagged suspicious** (4 datasets × 2
  architectures = the deliberately undersized/overfit cases).
- **0 false positives** among the 46 models trained on normally-sized
  data.
- Typical flagged-model signature (dataset 17, Transformer): train MSE
  ≈ 0.0000, val MSE = 4.7888 (>10⁸x divergence ratio), 8 training
  examples vs. 64 readout feature dimensions, train R² = 1.0000 — an
  unambiguous, correctly-caught p≫n overfit.

Full per-dataset numbers: `results/results.json`. Aggregate summary +
every flagged model and its trip reasons: `results/summary.json`.

## Figures

- `figures/per_dataset_r2.png` — test R² per dataset for both
  architectures; flagged-suspicious datasets shown in red.
- `figures/train_vs_val_mse.png` — train vs. val MSE scatter across all
  54 models, with the 8 detector-flagged overfit models highlighted in
  red against the `train_mse = val_mse` reference line.

## Project structure

```
generate_datasets.py     27 synthetic multivariate time-series datasets, fixed seed
models.py                ReservoirLSTMEncoder + AttentionTransformerEncoder (numpy)
suspicious_detector.py   4-check overfitting/suspicious-result detector
train_models.py          trains + evaluates both encoders x 27 datasets
run_all.py                full pipeline: generate -> train -> detect -> figures
tests/test_pipeline.py   real assertions against actual data/models/detector output
data/                    generated .npz datasets + metadata.json
results/                 results.json (per-dataset) + summary.json (aggregate)
figures/                 generated PNG figures
requirements.txt
.gitignore
```

## Tech stack

Python 3, numpy (hand-implemented recurrent reservoir + multi-head
self-attention forward passes), scikit-learn (`Ridge` readout,
`KFold` cross-validation for the stability check), matplotlib (figures),
pytest.
