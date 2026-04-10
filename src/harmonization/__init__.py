from src.harmonization.sitewise_scaler import SiteWiseStandardScaler


def make_harmonizer(method, batch_series, cov_df=None):
    """Factory for harmonizer objects. Returns None for 'raw'."""
    if method == 'raw':
        return None
    elif method in ('combat', 'neurocombat', 'covbat'):
        from combatlearn.combat import ComBat
        if method == 'combat':
            return ComBat(batch=batch_series, method='johnson')
        elif method == 'neurocombat':
            return ComBat(batch=batch_series, discrete_covariates=cov_df[['gender']],
                          continuous_covariates=cov_df[['age']], method='fortin')
        elif method == 'covbat':
            return ComBat(batch=batch_series, discrete_covariates=cov_df[['gender']],
                          continuous_covariates=cov_df[['age']], method='chen')
    elif method == 'sitewise':
        return SiteWiseStandardScaler(batch=batch_series)
    raise ValueError(f"Unknown harmonization method: {method}")
