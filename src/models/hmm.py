import numpy as np
import pandas as pd
import logging
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeHMM:
    """
    Hidden Markov Model for Regime Detection on latent embeddings.
    - Normalises the latent space with StandardScaler before fitting so that
      the HMM covariance matrix is well-conditioned and all states are used.
    - Uses multiple EM restart initialisations (n_init) for robustness.
    - Maps discovered states to semantic regimes (Risk-On / Defensive /
      Transitional / Crisis) ordered by the realised volatility of the
      original returns observed in each state.
    """

    REGIME_NAMES = ["Risk-On", "Defensive", "Transitional", "Crisis"]

    def __init__(self, n_components: int = 4,
                 covariance_type: str = "diag",   # diag is more robust than full
                 n_iter: int = 200,
                 n_init: int = 10,
                 random_state: int = 42):
        self.n_components = n_components
        self.n_init = n_init
        self.random_state = random_state
        self._build_model(covariance_type, n_iter, random_state)
        self.scaler = StandardScaler()
        self.state_map: dict = {}

    # ------------------------------------------------------------------
    def _build_model(self, covariance_type, n_iter, random_state):
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            min_covar=1e-3,
            tol=1e-4,
        )

    # ------------------------------------------------------------------
    def fit(self, latent_embeddings: np.ndarray, original_returns: pd.Series):
        """
        Fits the HMM using multiple random restarts and selects the run with
        the highest log-likelihood.  The latent space is normalised first.

        Args:
            latent_embeddings : (n_samples, latent_dim)
            original_returns  : pd.Series aligned with the embedding timeline.
        """
        # 1. Normalise ────────────────────────────────────────────────────
        z = self.scaler.fit_transform(latent_embeddings)

        # 2. Multiple restarts — keep best ────────────────────────────────
        logger.info(f"Fitting Gaussian HMM ({self.n_components} states) "
                    f"with {self.n_init} random restarts …")
        best_score, best_model = -np.inf, None
        for seed in range(self.n_init):
            try:
                m = GaussianHMM(
                    n_components=self.n_components,
                    covariance_type=self.model.covariance_type,
                    n_iter=self.model.n_iter,
                    random_state=self.random_state + seed,
                    min_covar=1e-3,
                    tol=1e-4,
                )
                m.fit(z)
                sc = m.score(z)
                if sc > best_score:
                    best_score = sc
                    best_model = m
            except Exception as e:
                logger.warning(f"HMM restart {seed} failed: {e}")

        if best_model is None:
            raise RuntimeError("All HMM restarts failed.")

        self.model = best_model
        logger.info(f"Best log-likelihood: {best_score:.2f}")

        # 3. Map states → semantic regime names ───────────────────────────
        states = self.model.predict(z)
        state_vols = {}
        for s in range(self.n_components):
            mask = states == s
            ret_in_state = original_returns.values[mask] if mask.any() else np.array([0.0])
            state_vols[s] = float(np.std(ret_in_state))

        sorted_states = sorted(state_vols, key=state_vols.get)
        # Build enough labels: use REGIME_NAMES first, then generic fallbacks
        regime_labels = list(self.REGIME_NAMES)
        while len(regime_labels) < self.n_components:
            regime_labels.append(f"Regime-{len(regime_labels)+1}")
        regime_labels = regime_labels[:self.n_components]
        for i, state in enumerate(sorted_states):
            self.state_map[state] = regime_labels[i]

        logger.info(f"Regime → state vol mapping: "
                    + ", ".join(f"{self.state_map[s]}: σ={state_vols[s]:.5f}"
                                for s in sorted_states))

    # ------------------------------------------------------------------
    def _normalize(self, z: np.ndarray) -> np.ndarray:
        return self.scaler.transform(z)

    def predict(self, latent_embeddings: np.ndarray) -> np.ndarray:
        """Returns raw integer state sequence."""
        return self.model.predict(self._normalize(latent_embeddings))

    def predict_regimes(self, latent_embeddings: np.ndarray) -> list:
        """Returns list of semantic regime name strings."""
        states = self.predict(latent_embeddings)
        return [self.state_map[s] for s in states]

    def get_transition_matrix(self) -> np.ndarray:
        return self.model.transmat_


# ── CLI smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)

    # Four clearly-separated Gaussian clusters in the latent space
    z_parts = [
        np.random.randn(300, 8) * 0.5 + np.array([3,  0, 0, 0, 0, 0, 0, 0]),
        np.random.randn(200, 8) * 0.5 + np.array([-3, 0, 0, 0, 0, 0, 0, 0]),
        np.random.randn(150, 8) * 0.5 + np.array([0,  3, 0, 0, 0, 0, 0, 0]),
        np.random.randn(100, 8) * 0.5 + np.array([0, -3, 0, 0, 0, 0, 0, 0]),
    ]
    vols = [0.005, 0.01, 0.02, 0.05]
    ret_parts = [np.random.randn(len(z)) * v for z, v in zip(z_parts, vols)]

    latent_space = np.vstack(z_parts)
    returns = pd.Series(np.concatenate(ret_parts))

    hmm = RegimeHMM(n_components=4)
    hmm.fit(latent_space, returns)
    regimes = hmm.predict_regimes(latent_space)
    print(pd.Series(regimes).value_counts())
