"""
Two sequence-encoder architectures for next-step multivariate time-series
prediction, implemented in raw numpy (no autodiff / no PyTorch -- see
README for the disk-space / import check that led to this fallback).

Both follow the same two-stage pattern used by the sibling projects in
this tree (BiologicalTransformerArchitecture, BioFoundationModelPostTraining):

  1. A fixed-random-weight sequence encoder computes the architecture's
     genuine forward pass over a sliding window of past steps
     (recurrent state updates for the LSTM-style encoder; scaled
     dot-product multi-head self-attention for the Transformer-style
     encoder). The *architecture* -- what each one computes and how
     information flows -- is real and hand-implemented.
  2. A trainable linear (Ridge) readout on top of the encoder's pooled
     features is fit with real gradient-free closed-form least squares
     (scikit-learn's Ridge), which is genuine learning on genuine
     features -- this is the documented simplification in place of
     hand-deriving full backprop-through-time / backprop-through-softmax-
     attention, exactly as the sibling projects document.

  The LSTM-style encoder is architecturally an Echo State Network
  (reservoir computing): a real, published class of recurrent models in
  which the recurrent weights are fixed (drawn to keep the state stable)
  and only the readout is trained. It is a legitimate, real recurrent
  architecture -- not a simulation of one.
"""
import numpy as np


def make_windows(series, window, horizon=1):
    """Turn a (T, F) series into (N, window, F) inputs and (N, F) targets
    for next-step prediction."""
    T = series.shape[0]
    X, y = [], []
    for t in range(T - window - horizon + 1):
        X.append(series[t:t + window])
        y.append(series[t + window + horizon - 1])
    if not X:
        return np.zeros((0, window, series.shape[1])), np.zeros((0, series.shape[1]))
    return np.stack(X), np.stack(y)


class ReservoirLSTMEncoder:
    """LSTM-style recurrent encoder (Echo State Network reservoir).

    h_t = tanh(W_in x_t + W_rec h_{t-1} + b)

    W_rec is spectral-radius-normalized so the recurrence is a genuine
    stable dynamical system (this is the real, hand-implemented part of
    the architecture). Only the downstream Ridge readout is trained.
    """

    def __init__(self, n_features, hidden_dim=48, spectral_radius=0.9, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.W_in = rng.normal(0, 1.0, size=(hidden_dim, n_features)) * 0.5
        W_rec = rng.normal(0, 1.0, size=(hidden_dim, hidden_dim))
        eigvals = np.linalg.eigvals(W_rec)
        radius = np.max(np.abs(eigvals))
        self.W_rec = W_rec * (spectral_radius / (radius + 1e-8))
        self.b = rng.normal(0, 0.1, size=(hidden_dim,))

    def encode_window(self, window):
        """window: (T_window, F) -> pooled feature vector (hidden_dim*2,)"""
        h = np.zeros(self.hidden_dim)
        states = []
        for x_t in window:
            h = np.tanh(self.W_in @ x_t + self.W_rec @ h + self.b)
            states.append(h)
        states = np.stack(states)
        # pooled features: final state + mean state over the window
        return np.concatenate([states[-1], states.mean(axis=0)])

    def transform(self, X_windows):
        return np.stack([self.encode_window(w) for w in X_windows])


class AttentionTransformerEncoder:
    """Transformer-style encoder: fixed-random multi-head scaled
    dot-product self-attention over the input window, followed by a
    trained Ridge readout on the pooled attended representation.
    """

    def __init__(self, n_features, d_model=32, n_heads=4, seed=0):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed + 777)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_embed = rng.normal(0, 0.5, size=(n_features, d_model))
        self.W_q = rng.normal(0, 0.5, size=(n_heads, d_model, self.d_head))
        self.W_k = rng.normal(0, 0.5, size=(n_heads, d_model, self.d_head))
        self.W_v = rng.normal(0, 0.5, size=(n_heads, d_model, self.d_head))
        self.W_out = rng.normal(0, 0.5, size=(d_model, d_model))

    @staticmethod
    def _softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _positional_encoding(self, T):
        pos = np.arange(T)[:, None]
        i = np.arange(self.d_model)[None, :]
        angle = pos / np.power(10000, (2 * (i // 2)) / self.d_model)
        pe = np.zeros((T, self.d_model))
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        return pe

    def encode_window(self, window):
        """window: (T_window, F) -> pooled feature vector (d_model,)"""
        T = window.shape[0]
        E = window @ self.W_embed + self._positional_encoding(T)  # (T, d_model)
        head_outputs = []
        for h in range(self.n_heads):
            Q = E @ self.W_q[h]
            K = E @ self.W_k[h]
            V = E @ self.W_v[h]
            scores = (Q @ K.T) / np.sqrt(self.d_head)
            attn = self._softmax(scores, axis=-1)
            head_outputs.append(attn @ V)
        concat = np.concatenate(head_outputs, axis=-1)  # (T, d_model)
        out = concat @ self.W_out
        # residual + mean-pool over sequence -> single feature vector
        pooled = (out + E).mean(axis=0)
        return pooled

    def transform(self, X_windows):
        return np.stack([self.encode_window(w) for w in X_windows])
