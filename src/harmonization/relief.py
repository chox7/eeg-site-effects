import numpy as np
from scipy import linalg
from itertools import combinations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Optional
import time
from sklearn.utils.extmath import randomized_svd


class RELIEFHarmonizer(BaseEstimator, TransformerMixin):
    """
    RELIEF Harmonization for batch effect correction.

    Parameters
    ----------
    scale_features : bool, default=True
        Whether to scale features
    eps : float, default=1e-3
        Convergence threshold
    max_iter : int, default=1000
        Maximum number of iterations
    verbose : bool, default=True
        Whether to print progress messages
    n_jobs : int, default=1
        Number of parallel processes for optimization loop.
        -1 means use all available cores.
        1 means no parallelization (sequential).
    log_every : int, default=10
        Log progress every N iterations
    """

    def __init__(self, scale_features=True,
                 eps=1e-3, max_iter=1000, verbose=True, n_jobs=1,
                 log_every=10):
        self.scale_features = scale_features
        self.eps = eps
        self.max_iter = max_iter
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.log_every = log_every

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for RELIEF harmonizer"""
        self.logger = logging.getLogger('RELIEFHarmonizer')
        if not self.logger.handlers:
            handler = logging.FileHandler("relief.log", mode="a")
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO if self.verbose else logging.WARNING)

    def _to_df(self, arr, index: pd.Index, name: str) -> Optional[pd.DataFrame]:
        """Convert array-like to DataFrame."""
        if arr is None:
            return None
        if isinstance(arr, pd.Series):
            arr = arr.to_frame()
        if not isinstance(arr, pd.DataFrame):
            arr = pd.DataFrame(arr, index=index)
        if not arr.index.equals(index):
            raise ValueError(f"`{name}` index mismatch with `X`.")
        return arr

    def _as_series(self, arr, index: pd.Index, name: str) -> pd.Series:
        """Convert array-like to categorical Series."""
        if isinstance(arr, pd.Series):
            s = arr.copy()
        else:
            s = pd.Series(arr, index=index)

        if not s.index.equals(index):
            raise ValueError(f"`{name}` index mismatch with `X`.")

        # Convert to categorical
        if not isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype('category')

        return s

    def _build_design_matrix(self, batch: pd.Series,
                             discrete_covariates: Optional[pd.DataFrame],
                             continuous_covariates: Optional[pd.DataFrame]) -> np.ndarray:
        """Build design matrix from batch and covariates (similar to ComBat)."""
        # Start with batch dummy variables (one-hot encoding, no drop_first)
        batch_dummies = pd.get_dummies(batch, drop_first=False, dtype=np.float32)

        parts = [batch_dummies]

        # Add discrete covariates with dummy encoding
        if discrete_covariates is not None:
            disc_dummies = pd.get_dummies(
                discrete_covariates.astype('category'),
                drop_first=True,
                dtype=np.float32
            )
            parts.append(disc_dummies)

        # Add continuous covariates as-is
        if continuous_covariates is not None:
            parts.append(continuous_covariates.astype(np.float32))

        # Concatenate all parts
        design = pd.concat(parts, axis=1).values

        return design

    def _frob(self, X):
        """Frobenius norm squared"""
        return np.sum(X * X)

    def _sigma_rmt(self, X):
        """
        Estimate sigma using MAD (Mean Absolute Deviation) method.
        Based on Gavish & Donoho empirical approximation.
        """
        X_t = X.T
        X_centered = X_t - np.mean(X_t, axis=0, keepdims=True)

        n, p = X_centered.shape

        # Compute singular values
        singular_values = linalg.svdvals(X_centered)

        # Gavish-Donoho approximation
        beta = min(n, p) / max(n, p)
        lambdastar = np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1))))
        wbstar = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43

        sigma = np.median(singular_values) / (np.sqrt(max(n, p)) * (lambdastar / wbstar))

        return sigma

    def _softSVD(self, X, lambda_val, k=90):
        """Soft-thresholded SVD"""
        m, n = X.shape
        U, d, Vt = randomized_svd(X, n_components=min(k, min(m, n) - 1), random_state=42)
        nuc = np.maximum(d - lambda_val, 0)
        out = U @ np.diag(nuc) @ Vt
        return {'out': out, 'nuc': np.sum(nuc)}

    def _process_batch(self, b, dat, estim, index_set_batch, lambda_set):
        """Process a single batch - designed for parallel execution"""
        other_estim = np.sum([estim[i] for i in range(len(estim)) if i != b], axis=0)
        temp_data = (dat - other_estim)[:, index_set_batch[b]]
        temp = self._softSVD(temp_data, lambda_set[b])

        return b, temp['out'], temp['nuc']

    def fit(self, X, y=None, *, batch, discrete_covariates=None, continuous_covariates=None):
        """
        Fit the RELIEF harmonizer to the data.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)
            Training data where rows are features and columns are samples
        y : Ignored
            Not used, present for API consistency by convention.
        batch : array-like of shape (n_samples,)
            Batch labels for each sample. Required keyword argument.
        discrete_covariates : array-like of shape (n_samples, n_discrete_covariates), optional
            Discrete covariates (e.g., sex, site). Will be one-hot encoded automatically.
        continuous_covariates : array-like of shape (n_samples, n_continuous_covariates), optional
            Continuous covariates (e.g., age). Will be used as-is.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        idx = X.columns
        batch_series = self._as_series(batch, idx, "batch")
        disc = self._to_df(discrete_covariates, idx, "discrete_covariates")
        cont = self._to_df(continuous_covariates, idx, "continuous_covariates")
        n_disc = disc.shape[1] if disc is not None else 0
        n_cont = cont.shape[1] if cont is not None else 0
        q = n_disc + n_cont

        if q > 0:
            self.logger.info(
                f"Performing RELIEF harmonization with covariates: "
                f"{n_disc} discrete, {n_cont} continuous"
            )
        else:
            self.logger.info("Performing RELIEF harmonization without covariates")

        X_np = X.values.astype(np.float32)
        p, n = X_np.shape
        self.n_features_in_, self.n_samples_in_ = p, n
        self.dat_original_ = X_np.copy()

        # Convert batch to numeric
        batch_levels = batch_series.cat.categories
        batch_numeric = batch_series.cat.codes.values
        batch_id = np.unique(batch_numeric)
        n_batch = len(batch_id)

        self.logger.info(f"Found {n_batch} batches: {list(batch_levels)}")

        # Store for later
        self._batch_levels = batch_levels
        self.batch_to_num_ = {label: idx for idx, label in enumerate(batch_levels)}

        self.Xbeta_ = np.zeros((p, n), dtype=np.float32)
        self.gamma_ = np.zeros((p, n), dtype=np.float32)

        # Create batch covariates (one-hot encoding)
        batch_covariates = np.zeros((n, n_batch), dtype=np.float32)
        for i, b in enumerate(batch_numeric):
            batch_covariates[i, b] = 1

        # Calculate Xbeta
        if disc is None and cont is None:
            self.Xbeta_[:] = np.outer(np.mean(X_np, axis=1), np.ones(n))
        else:
            design = self._build_design_matrix(batch_series, disc, cont)
            beta_hat = linalg.lstsq(design, X_np.T)[0]  # (design_cols, n_features)
            beta_hat_nonbatch = beta_hat[n_batch:]  # Skip batch columns
            covariate_effect = design[:, n_batch:] @ beta_hat_nonbatch  # (n_samples, n_features)
            self.Xbeta_ = covariate_effect.T  # Transpose to (n_features, n_samples)
            q = design.shape[1] - n_batch

        residual1 = X_np - self.Xbeta_
        Pb = batch_covariates @ linalg.pinv(batch_covariates.T @ batch_covariates, atol=0) @ batch_covariates.T
        self.gamma_ = residual1 @ Pb
        residual2 = residual1 - self.gamma_

        if self.scale_features:
            sigma_mat = np.outer(np.sqrt(np.sum(residual2 ** 2, axis=1) / (n - n_batch - q)), np.ones(n))
        else:
            sigma_mat = 1

        self.sigma_mat_ = sigma_mat
        dat = residual2 / sigma_mat

        # Generate all subset combinations
        sub_batch = []
        for r in [1, n_batch]:
            sub_batch.extend(list(combinations(batch_id, r)))

        nvec = np.zeros(n_batch)
        sigma_mat_batch = np.ones((p, n))

        for b_idx, b in enumerate(batch_id):
            order_temp_batch = np.where(batch_numeric == b)[0]
            nvec[b_idx] = len(order_temp_batch)

            s = self._sigma_rmt(dat[:, order_temp_batch])
            sigma_mat_batch[:, order_temp_batch] = sigma_mat_batch[:, order_temp_batch] * s
            dat[:, order_temp_batch] = dat[:, order_temp_batch] / s

        self.sigma_mat_batch_ = sigma_mat_batch

        unique_sigmas = []
        for b_idx, b in enumerate(batch_id):
            order_temp_batch = np.where(batch_numeric == b)[0]
            unique_sigmas.append(sigma_mat_batch[0, order_temp_batch[0]])  # Get one value per batch
        unique_sigmas = np.array(unique_sigmas)
        self.sigma_harmonized_ = np.sqrt(np.sum((unique_sigmas ** 2) * nvec) / nvec.sum())

        # Calculate lambda values
        lambda_set = np.zeros(len(sub_batch))
        for b_idx, b_subset in enumerate(sub_batch):
            lambda_set[b_idx] = np.sqrt(p) + np.sqrt(np.sum(nvec[list(b_subset)]))

        # Create index sets
        index_set_batch = []
        for b_subset in sub_batch:
            indices = np.where(np.isin(batch_numeric, b_subset))[0]
            index_set_batch.append(indices)

        # Initialize estimates
        estim = [np.zeros((p, n), dtype=np.float32) for _ in range(len(sub_batch))]

        bool_continue = True
        count = 1
        crit0 = 0

        self.logger.info("Starting optimization loop...")
        start_time = time.time()

        # Determine number of workers
        n_workers = self.n_jobs
        if n_workers == -1:
            import os
            n_workers = os.cpu_count()

        use_parallel = n_workers > 1 and len(sub_batch) > 1

        if use_parallel:
            self.logger.info(f"Using parallel processing with {n_workers} workers")

        while bool_continue:
            if count % self.log_every == 0 or count == 1:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Iteration {count}/{self.max_iter} | "
                    f"Criterion: {crit0:.6e} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            crit0_old = crit0
            nuc_temp = np.zeros(len(sub_batch), dtype=np.float32)


            for b in range(len(sub_batch) - 1, -1, -1):
                other_estim = np.sum([estim[i] for i in range(len(sub_batch)) if i != b], axis=0)
                temp_data = (dat - other_estim)[:, index_set_batch[b]]
                temp = self._softSVD(temp_data, lambda_set[b])
                estim[b][:, index_set_batch[b]] = temp['out']
                nuc_temp[b] = temp['nuc']

            total_estim = np.sum(estim, axis=0)
            crit0 = 0.5 * self._frob(dat - total_estim) + np.dot(lambda_set, nuc_temp)

            if abs(crit0_old - crit0) < self.eps:
                bool_continue = False
                self.logger.info(
                    f"Converged at iteration {count} "
                    f"(criterion change: {abs(crit0_old - crit0):.6e})"
                )
            elif count == self.max_iter:
                bool_continue = False
                self.logger.warning(
                    f"Maximum iterations ({self.max_iter}) reached without convergence. "
                    f"Consider increasing max_iter."
                )
            else:
                count += 1

        self.n_iter_ = count
        total_time = time.time() - start_time
        self.logger.info(f"Optimization completed in {total_time:.2f}s ({count} iterations)")

        # Final calculations
        total_estim = np.sum(estim, axis=0)
        E = dat - total_estim
        self.E_scaled_ = self.sigma_mat_ * E
        self.E_original_ = self.sigma_mat_ * self.sigma_mat_batch_ * E
        self.R_ = self.sigma_mat_ * self.sigma_mat_batch_ * estim[-1]
        self.I_ = self.sigma_mat_ * self.sigma_mat_batch_ * np.sum(estim[:-1], axis=0)

        return self

    def transform(self, X):
        """
        Apply harmonization to data.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)
            Data to harmonize. Should be the same data used in fit().

        Returns
        -------
        X_harmonized : ndarray of shape (n_features, n_samples)
            Harmonized data
        """
        # Check if fit has been called
        if not hasattr(self, 'Xbeta_'):
            raise RuntimeError("This RELIEFHarmonizer instance is not fitted yet. "
                               "Call 'fit' with appropriate arguments before using 'transform'.")

        X = np.array(X)

        if X.shape != (self.n_features_in_, self.n_samples_in_):
            raise ValueError(f"X has shape {X.shape} but this harmonizer was fitted for "
                             f"shape ({self.n_features_in_}, {self.n_samples_in_})")

        harmonized = self.Xbeta_ + self.R_ + self.sigma_harmonized_ * self.E_scaled_
        return harmonized

    def fit_transform(self, X, y=None, *, batch, discrete_covariates=None, continuous_covariates=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_features, n_samples)
            Training data
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_harmonized : ndarray of shape (n_features, n_samples)
            Harmonized data
        """
        return self.fit(X, y, batch=batch, discrete_covariates=discrete_covariates, continuous_covariates=continuous_covariates).transform(X)

    def get_estimates(self):
        """
        Get all estimated components from the harmonization.

        Returns
        -------
        estimates : dict
            Dictionary containing all estimated components:
            - Xbeta: mean or covariate effects
            - gamma: batch effects
            - sigma_mat: feature scaling factors
            - sigma_mat_batch: batch-specific scaling factors
            - sigma_harmonized: harmonized sigma value
            - R: reference batch effects
            - I: interaction effects
            - E_scaled: scaled residuals
            - E_original: original scale residuals
        """
        if not hasattr(self, 'Xbeta_'):
            raise RuntimeError("This RELIEFHarmonizer instance is not fitted yet.")

        return {
            'Xbeta': self.Xbeta_,
            'gamma': self.gamma_,
            'sigma_mat': self.sigma_mat_,
            'sigma_mat_batch': self.sigma_mat_batch_,
            'sigma_harmonized': self.sigma_harmonized_,
            'R': self.R_,
            'I': self.I_,
            'E_scaled': self.E_scaled_,
            'E_original': self.E_original_
        }