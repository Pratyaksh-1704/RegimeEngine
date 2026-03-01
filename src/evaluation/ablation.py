"""
src/evaluation/ablation.py
──────────────────────────
Ablation study: compares four regime-detection methods head-to-head
using five quantitative metrics. Results are publication-ready.

Methods
───────
  M0  Raw Returns HMM       – HMM fitted directly on log-return vectors
  M1  Engineered Features HMM – HMM on the full hand-crafted feature matrix
  M2  PCA + HMM             – HMM on top-k PCA components of features
                               (classical dimensionality-reduction baseline)
  M3  TCN + HMM (Ours)      – HMM on the contrastive-TCN latent space

Metrics
───────
  1. Silhouette Score        – compactness + separation of regime clusters
                               range [-1, +1], higher is better
  2. Calinski-Harabasz (CH)  – between/within cluster variance ratio
                               higher is better
  3. Davies-Bouldin (DB)     – average ratio within-to-between cluster dist
                               lower is better
  4. Regime Stability        – mean consecutive-day run length per label
                               higher = more persistent, tradeable regimes
  5. Predictive Validity     – Spearman correlation of regime label with
                               next-period realized volatility
                               higher magnitude is better (|ρ|, signed by direction)
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score,
                              calinski_harabasz_score,
                              davies_bouldin_score)
from scipy import stats
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MethodResult:
    """Results for one method."""
    name:           str
    labels:         np.ndarray          # integer state assignments
    space:          np.ndarray          # representation used (for viz)
    silhouette:     float = 0.0
    ch_score:       float = 0.0
    db_score:       float = 0.0
    stability:      float = 0.0         # mean run length (days)
    pred_validity:  float = 0.0         # signed Spearman ρ with next-vol
    n_regimes:      int   = 4
    converged:      bool  = True
    notes:          str   = ""


# ──────────────────────────────────────────────────────────────────────────────
class AblationStudy:
    """
    Run all four methods on the same data and return a comparable
    MethodResult for each, plus a summary DataFrame.
    """

    def __init__(self, n_components: int = 4,
                 pca_dims: int = 8,
                 n_init: int = 5,
                 n_iter: int = 200,
                 random_state: int = 42):
        self.n_components   = n_components
        self.pca_dims       = pca_dims
        self.n_init         = n_init
        self.n_iter         = n_iter
        self.random_state   = random_state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fit_hmm_best(self, X: np.ndarray) -> tuple[np.ndarray, bool]:
        """Fit HMM with multiple restarts; return (labels, converged)."""
        X = StandardScaler().fit_transform(X)
        best_score, best_labels, converged = -np.inf, None, False
        for seed in range(self.n_init):
            try:
                m = GaussianHMM(
                    n_components=self.n_components,
                    covariance_type='diag',
                    n_iter=self.n_iter,
                    random_state=self.random_state + seed,
                    min_covar=1e-3,
                )
                m.fit(X)
                sc = m.score(X)
                if sc > best_score:
                    best_score   = sc
                    best_labels  = m.predict(X)
                    converged    = m.monitor_.converged
            except Exception as e:
                logger.warning(f"HMM restart {seed} error: {e}")
        return best_labels, converged

    def _compute_clustering_metrics(self,
                                     X: np.ndarray,
                                     labels: np.ndarray) -> tuple[float,float,float]:
        """Silhouette, CH, DB on the raw (un-normalised) space X."""
        uniq = np.unique(labels)
        if len(uniq) < 2:
            return 0.0, 0.0, 999.0
        try:
            sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
            ch  = calinski_harabasz_score(X, labels)
            db  = davies_bouldin_score(X, labels)
        except Exception:
            sil, ch, db = 0.0, 0.0, 999.0
        return float(sil), float(ch), float(db)

    @staticmethod
    def _regime_stability(labels: np.ndarray) -> float:
        """Mean consecutive-day run length."""
        if len(labels) == 0:
            return 0.0
        runs, count = [], 1
        for i in range(1, len(labels)):
            if labels[i] == labels[i-1]:
                count += 1
            else:
                runs.append(count); count = 1
        runs.append(count)
        return float(np.mean(runs))

    @staticmethod
    def _predictive_validity(labels: np.ndarray,
                              returns: np.ndarray,
                              horizon: int = 21) -> float:
        """
        Spearman ρ between regime integer label and next-`horizon`-day
        realized volatility. Positive ρ means higher label → higher vol
        (as expected when labels are vol-ordered). We return the signed ρ.
        """
        n = min(len(labels), len(returns)) - horizon
        if n < 30:
            return 0.0
        lbl  = labels[:n].astype(float)
        rvol = np.array([
            returns[i:i+horizon].std() * np.sqrt(252)
            for i in range(n)
        ])
        rho, _ = stats.spearmanr(lbl, rvol)
        return float(rho) if not np.isnan(rho) else 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self,
            returns_df:   pd.DataFrame,
            features_df:  pd.DataFrame,
            latent_z:     np.ndarray,
            window_size:  int = 21) -> tuple[list[MethodResult], pd.DataFrame]:
        """
        Parameters
        ──────────
        returns_df   : DataFrame of log returns – shape (T, n_assets)
        features_df  : Engineered features DataFrame – shape (T, n_feat)
        latent_z     : TCN latent embeddings – shape (T_z, latent_dim)
        window_size  : lookback window used by TCN (for time-alignment)

        Returns
        ───────
        results  : list[MethodResult] – one per method
        summary  : pd.DataFrame – formatted comparison table
        """
        results: list[MethodResult] = []

        # Align returns to TCN timeline
        ret_arr = returns_df.mean(axis=1).values   # equal-weight proxy
        # latent_z corresponds to features_df.index[window_size-1:]
        z_start = window_size - 1
        ret_z   = ret_arr[z_start : z_start + len(latent_z)]
        feat_z  = features_df.values[z_start : z_start + len(latent_z)]

        # ── M0: Raw Returns HMM ───────────────────────────────────────────────
        logger.info("[Ablation] M0 – Raw Returns HMM")
        try:
            ret_input = returns_df.values[z_start : z_start + len(latent_z)]
            labels0, conv0 = self._fit_hmm_best(ret_input)
            sil0, ch0, db0 = self._compute_clustering_metrics(ret_input, labels0)
            r0 = MethodResult(
                name='M0: Raw Returns HMM',
                labels=labels0, space=ret_input,
                silhouette=sil0, ch_score=ch0, db_score=db0,
                stability=self._regime_stability(labels0),
                pred_validity=self._predictive_validity(labels0, ret_z),
                n_regimes=self.n_components, converged=conv0,
                notes="Direct HMM on log-return series")
        except Exception as e:
            logger.error(f"M0 failed: {e}")
            r0 = MethodResult('M0: Raw Returns HMM', np.zeros(len(latent_z)),
                              np.zeros((len(latent_z),1)), notes=f"Failed: {e}")
        results.append(r0)

        # ── M1: Engineered Features HMM ───────────────────────────────────────
        logger.info("[Ablation] M1 – Engineered Features HMM")
        try:
            labels1, conv1 = self._fit_hmm_best(feat_z)
            sil1, ch1, db1 = self._compute_clustering_metrics(feat_z, labels1)
            r1 = MethodResult(
                name='M1: Features HMM',
                labels=labels1, space=feat_z,
                silhouette=sil1, ch_score=ch1, db_score=db1,
                stability=self._regime_stability(labels1),
                pred_validity=self._predictive_validity(labels1, ret_z),
                n_regimes=self.n_components, converged=conv1,
                notes="HMM on hand-crafted feature matrix")
        except Exception as e:
            logger.error(f"M1 failed: {e}")
            r1 = MethodResult('M1: Features HMM', np.zeros(len(latent_z)),
                              np.zeros((len(latent_z),1)), notes=f"Failed: {e}")
        results.append(r1)

        # ── M2: PCA + HMM ─────────────────────────────────────────────────────
        logger.info("[Ablation] M2 – PCA + HMM")
        try:
            ndims = min(self.pca_dims, feat_z.shape[1])
            pca   = PCA(n_components=ndims, random_state=self.random_state)
            feat_pca = pca.fit_transform(StandardScaler().fit_transform(feat_z))
            labels2, conv2 = self._fit_hmm_best(feat_pca)
            sil2, ch2, db2 = self._compute_clustering_metrics(feat_pca, labels2)
            var_exp = pca.explained_variance_ratio_.sum()
            r2 = MethodResult(
                name=f'M2: PCA({ndims}d) + HMM',
                labels=labels2, space=feat_pca,
                silhouette=sil2, ch_score=ch2, db_score=db2,
                stability=self._regime_stability(labels2),
                pred_validity=self._predictive_validity(labels2, ret_z),
                n_regimes=self.n_components, converged=conv2,
                notes=f"PCA retains {var_exp:.1%} variance")
        except Exception as e:
            logger.error(f"M2 failed: {e}")
            r2 = MethodResult(f'M2: PCA + HMM', np.zeros(len(latent_z)),
                              np.zeros((len(latent_z),1)), notes=f"Failed: {e}")
        results.append(r2)

        # ── M3: TCN + HMM (Ours) ──────────────────────────────────────────────
        logger.info("[Ablation] M3 – TCN Contrastive + HMM (Ours)")
        try:
            labels3, conv3 = self._fit_hmm_best(latent_z)
            sil3, ch3, db3 = self._compute_clustering_metrics(latent_z, labels3)
            r3 = MethodResult(
                name='M3: TCN + HMM (Ours)',
                labels=labels3, space=latent_z,
                silhouette=sil3, ch_score=ch3, db_score=db3,
                stability=self._regime_stability(labels3),
                pred_validity=self._predictive_validity(labels3, ret_z),
                n_regimes=self.n_components, converged=conv3,
                notes="Contrastive TCN latent space → HMM")
        except Exception as e:
            logger.error(f"M3 failed: {e}")
            r3 = MethodResult('M3: TCN + HMM (Ours)', np.zeros(len(latent_z)),
                              np.zeros((len(latent_z),1)), notes=f"Failed: {e}")
        results.append(r3)

        summary = self._make_summary(results)
        return results, summary

    # ── Summary table ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_summary(results: list[MethodResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            rows.append({
                'Method':            r.name,
                'Silhouette ↑':      round(r.silhouette, 4),
                'Calinski-Harabasz ↑': round(r.ch_score, 2),
                'Davies-Bouldin ↓':  round(r.db_score, 4),
                'Stability (days) ↑': round(r.stability, 1),
                'Pred. Validity ↑':  round(r.pred_validity, 4),
                'Converged':         '✓' if r.converged else '✗',
                'Notes':             r.notes,
            })
        df = pd.DataFrame(rows).set_index('Method')

        # Compute rank score: higher is better for all except DB
        n = len(results)
        df['_sil_rank']  = pd.Series([r.silhouette for r in results]).rank(ascending=True).values
        df['_ch_rank']   = pd.Series([r.ch_score for r in results]).rank(ascending=True).values
        df['_db_rank']   = pd.Series([r.db_score for r in results]).rank(ascending=False).values
        df['_stab_rank'] = pd.Series([r.stability for r in results]).rank(ascending=True).values
        df['_pv_rank']   = pd.Series([abs(r.pred_validity) for r in results]).rank(ascending=True).values
        df['Overall Rank'] = (df[['_sil_rank','_ch_rank','_db_rank',
                                   '_stab_rank','_pv_rank']].sum(axis=1)
                               .rank(ascending=False).astype(int))
        df.drop(columns=[c for c in df.columns if c.startswith('_')], inplace=True)
        return df


# ── Convenience runner ────────────────────────────────────────────────────────
def run_ablation(returns_df, features_df, latent_z,
                 n_components=4, window_size=21) -> tuple:
    study = AblationStudy(n_components=n_components)
    return study.run(returns_df, features_df, latent_z, window_size)
