



import numpy as np
from typing import Union, List, Optional

from mftl.utils.helpers import TwoTaskWeights, _sanitize_vector


class Autoencoder:





    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        learning_rate: float = 1e-3,
        l2: float = 1e-6,
        clip: float = 1.0
    ) -> None:










        rng = np.random.RandomState(42)
        self.input_dim = input_dim
        self.latent_dim = latent_dim


        self.W_enc = rng.randn(input_dim, latent_dim) / np.sqrt(input_dim)
        self.W_dec = rng.randn(latent_dim, input_dim) / np.sqrt(latent_dim)

        self.learning_rate = learning_rate
        self.l2 = l2
        self.clip = clip

    def encode_flat(self, flat: np.ndarray) -> np.ndarray:








        return flat @ self.W_enc

    def decode_flat(self, z: np.ndarray) -> np.ndarray:








        return z @ self.W_dec

    def partial_fit(self, batch_flats: np.ndarray, epochs: int = 1) -> None:






        if batch_flats.ndim == 1:
            batch_flats = batch_flats.reshape(1, -1)

        X = np.asarray(batch_flats, dtype=float)
        X = np.where(np.isfinite(X), X, 0.0)


        if X.shape[1] != self.input_dim:
            self._reinitialize_with_new_dim(X.shape[1])

        for _ in range(epochs):

            Z = X @ self.W_enc
            X_hat = Z @ self.W_dec


            residual = X_hat - X


            grad_W_dec = Z.T @ residual + self.l2 * self.W_dec


            grad_W_enc = X.T @ (residual @ self.W_dec.T) + self.l2 * self.W_enc


            if self.clip is not None:
                grad_W_dec = self._clip_gradient(grad_W_dec)
                grad_W_enc = self._clip_gradient(grad_W_enc)


            self.W_enc -= self.learning_rate * grad_W_enc
            self.W_dec -= self.learning_rate * grad_W_dec

    def _clip_gradient(self, grad: np.ndarray) -> np.ndarray:

        norm = np.linalg.norm(grad)
        if np.isfinite(norm) and norm > self.clip:
            return grad * (self.clip / norm)
        return grad

    def _reinitialize_with_new_dim(self, new_input_dim: int) -> None:

        rng = np.random.RandomState(42)
        new_latent_dim = min(self.latent_dim, max(1, new_input_dim // 2))
        self.input_dim = new_input_dim
        self.latent_dim = new_latent_dim
        self.W_enc = rng.randn(self.input_dim, self.latent_dim) / np.sqrt(self.input_dim)
        self.W_dec = rng.randn(self.latent_dim, self.input_dim) / np.sqrt(self.latent_dim)

    def encode(self, tw: TwoTaskWeights) -> np.ndarray:








        flat = np.concatenate([
            tw.coeff_a.ravel(),
            tw.inter_a.ravel(),
            tw.coeff_b.ravel(),
            tw.inter_b.ravel()
        ])
        return self.encode_flat(flat)

    def decode(self, z: np.ndarray, template: TwoTaskWeights) -> TwoTaskWeights:









        flat_est = self.decode_flat(z)


        s_ca = template.coeff_a.size
        s_ia = template.inter_a.size
        s_cb = template.coeff_b.size
        s_ib = template.inter_b.size
        total = s_ca + s_ia + s_cb + s_ib

        if flat_est.size < total:
            flat_est = np.pad(flat_est, (0, total - flat_est.size))

        p0 = 0
        p1 = p0 + s_ca
        coeff_a = flat_est[p0:p1].reshape(template.coeff_a.shape)

        p0 = p1
        p1 = p0 + s_ia
        inter_a = flat_est[p0:p1].reshape(template.inter_a.shape)

        p0 = p1
        p1 = p0 + s_cb
        coeff_b = flat_est[p0:p1].reshape(template.coeff_b.shape)

        p0 = p1
        p1 = p0 + s_ib
        inter_b = flat_est[p0:p1].reshape(template.inter_b.shape)

        return TwoTaskWeights(coeff_a, inter_a, coeff_b, inter_b)
