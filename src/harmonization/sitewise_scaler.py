import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Literal, Optional, Union, Dict, Tuple, Any, cast
import numpy.typing as npt


try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

ArrayLike = Union[pd.DataFrame, pd.Series, npt.NDArray[Any]]
FloatArray = npt.NDArray[np.float64]

class SiteWiseStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, batch):
        self.batch = batch

    def fit(self, X, y=None):
        self.site_scalers_ = {}
        self.columns_ = X.columns

        try:
            batch_fit = self.batch.loc[X.index]
        except KeyError:
            raise ValueError("Index of X does not match index of 'sites' provided at initialization.")

        for site in batch_fit.unique():
            scaler = StandardScaler()
            site_data = X[batch_fit == site]

            if site_data.empty:
                print(f"Warning: No data provided for site '{site}' during .fit().")
                continue

            self.site_scalers_[site] = scaler.fit(site_data)

        return self

    def transform(self, X):
        check_is_fitted(self, 'site_scalers_')

        try:
            batch_transform = self.batch.loc[X.index]
        except KeyError:
            raise ValueError("Index of X does not match index of 'sites' provided at initialization.")

        # Create an empty output DataFrame
        output_X = pd.DataFrame(index=X.index, columns=self.columns_, dtype=float)

        for site in batch_transform.unique():
            if site in self.site_scalers_:
                site_data = X[batch_transform == site]
                if site_data.empty:
                    continue
                output_X.loc[site_data.index] = self.site_scalers_[site].transform(site_data)
            else:
                # This error is critical for LOSO validation
                raise ValueError(f"Error: Site '{site}' found in data, "
                                 f"but it was not seen during .fit().")

        return output_X

    def plot_transformation(
            self,
            X: ArrayLike, *,
            reduction_method: Literal['pca', 'tsne', 'umap'] = 'pca',
            n_components: Literal[2, 3] = 2,
            plot_type: Literal['static', 'interactive'] = 'static',
            figsize: Tuple[int, int] = (12, 5),
            alpha: float = 0.7,
            point_size: int = 50,
            cmap: str = 'Set1',
            title: Optional[str] = None,
            show_legend: bool = True,
            return_embeddings: bool = False,
            **reduction_kwargs) -> Union[Any, Tuple[Any, Dict[str, FloatArray]]]:
        """
        Visualize the SiteWiseStandardScaler transformation effect using dimensionality reduction.

        It shows a before/after comparison of data transformed by `SiteWiseStandardScaler` using
        PCA, t-SNE, or UMAP to reduce dimensions for visualization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform and visualize.

        reduction_method : {`'pca'`, `'tsne'`, `'umap'`}, default=`'pca'`
            Dimensionality reduction method.

        n_components : {2, 3}, default=2
            Number of components for dimensionality reduction.

        plot_type : {`'static'`, `'interactive'`}, default=`'static'`
            Visualization type:
            - `'static'`: matplotlib plots (can be saved as images)
            - `'interactive'`: plotly plots (explorable, requires plotly)

        return_embeddings : bool, default=False
            If `True`, return embeddings along with the plot.

        **reduction_kwargs : dict
            Additional parameters for reduction methods.

        Returns
        -------
        fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
            The figure object containing the plots.

        embeddings : dict, optional
            If `return_embeddings=True`, dictionary with:
            - `'original'`: embedding of original data
            - `'transformed'`: embedding of SiteWiseStandardScaler-transformed data
        """
        check_is_fitted(self, 'site_scalers_')

        if n_components not in [2, 3]:
            raise ValueError(f"n_components must be 2 or 3, got {n_components}")
        if reduction_method not in ['pca', 'tsne', 'umap']:
            raise ValueError(f"reduction_method must be 'pca', 'tsne', or 'umap', got '{reduction_method}'")
        if plot_type not in ['static', 'interactive']:
            raise ValueError(f"plot_type must be 'static' or 'interactive', got '{plot_type}'")

        if reduction_method == 'umap' and not UMAP_AVAILABLE:
            raise ImportError("UMAP is not installed. Install with: pip install umap-learn")
        if plot_type == 'interactive' and not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not installed. Install with: pip install plotly")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        idx = X.index
        batch_vec = self.batch.loc[idx]
        if batch_vec is None:
            raise ValueError("Batch information is required for visualization")

        X_transformed = self.transform(X)

        X_np = X.values
        X_trans_np = X_transformed.values

        if reduction_method == 'pca':
            reducer_orig = PCA(n_components=n_components, **reduction_kwargs)
            reducer_trans = PCA(n_components=n_components, **reduction_kwargs)
        elif reduction_method == 'tsne':
            tsne_params = {'perplexity': 30, 'max_iter': 1000, 'random_state': 42}
            tsne_params.update(reduction_kwargs)
            reducer_orig = TSNE(n_components=n_components, **tsne_params)
            reducer_trans = TSNE(n_components=n_components, **tsne_params)
        else:
            umap_params = {'random_state': 42}
            umap_params.update(reduction_kwargs)
            reducer_orig = umap.UMAP(n_components=n_components, **reduction_kwargs)
            reducer_trans = umap.UMAP(n_components=n_components, **reduction_kwargs)

        X_embedded_orig = reducer_orig.fit_transform(X_np)
        X_embedded_trans = reducer_trans.fit_transform(X_trans_np)

        if plot_type == 'static':
            fig = self._create_static_plot(
                X_embedded_orig, X_embedded_trans, batch_vec,
                reduction_method, n_components, figsize, alpha,
                point_size, cmap, title, show_legend
            )
        else:
            fig = self._create_interactive_plot(
                X_embedded_orig, X_embedded_trans, batch_vec,
                reduction_method, n_components, cmap, title, show_legend
            )

        if return_embeddings:
            embeddings = {
                'original': X_embedded_orig,
                'transformed': X_embedded_trans
            }
            return fig, embeddings
        else:
            return fig


    def _create_static_plot(
            self,
            X_orig: FloatArray,
            X_trans: FloatArray,
            batch_labels: pd.Series,
            method: str,
            n_components: int,
            figsize: Tuple[int, int],
            alpha: float,
            point_size: int,
            cmap: str,
            title: Optional[str],
            show_legend: bool) -> Any:
        """Create static plots using matplotlib."""

        fig = plt.figure(figsize=figsize)

        unique_batches = batch_labels.drop_duplicates()
        n_batches = len(unique_batches)

        if n_batches <= 10:
            colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_batches))
        else:
            colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_batches))

        if n_components == 2:
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
        else:
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

        for i, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            if n_components == 2:
                ax1.scatter(
                    X_orig[mask, 0], X_orig[mask, 1],
                    c=[colors[i]],
                    s=point_size,
                    alpha=alpha,
                    label=f'Batch {batch}',
                    edgecolors='black',
                    linewidth=0.5
                )
            else:
                ax1.scatter(
                    X_orig[mask, 0], X_orig[mask, 1], X_orig[mask, 2],
                    c=[colors[i]],
                    s=point_size,
                    alpha=alpha,
                    label=f'Batch {batch}',
                    edgecolors='black',
                    linewidth=0.5
                )

        ax1.set_title(f'Before SiteWiseStandardScaler correction\n({method.upper()})')
        ax1.set_xlabel(f'{method.upper()}1')
        ax1.set_ylabel(f'{method.upper()}2')
        if n_components == 3:
            ax1.set_zlabel(f'{method.upper()}3')

        for i, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            if n_components == 2:
                ax2.scatter(
                    X_trans[mask, 0], X_trans[mask, 1],
                    c=[colors[i]],
                    s=point_size,
                    alpha=alpha,
                    label=f'Batch {batch}',
                    edgecolors='black',
                    linewidth=0.5
                )
            else:
                ax2.scatter(
                    X_trans[mask, 0], X_trans[mask, 1], X_trans[mask, 2],
                    c=[colors[i]],
                    s=point_size,
                    alpha=alpha,
                    label=f'Batch {batch}',
                    edgecolors='black',
                    linewidth=0.5
                )

        ax2.set_title(f'After SiteWiseStandardScaler correction\n({method.upper()})')
        ax2.set_xlabel(f'{method.upper()}1')
        ax2.set_ylabel(f'{method.upper()}2')
        if n_components == 3:
            ax2.set_zlabel(f'{method.upper()}3')

        if show_legend and n_batches <= 20:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if title is None:
            title = f'SiteWiseStandardScaler correction effect visualized with {method.upper()}'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig


    def _create_interactive_plot(
            self,
            X_orig: FloatArray,
            X_trans: FloatArray,
            batch_labels: pd.Series,
            method: str,
            n_components: int,
            cmap: str,
            title: Optional[str],
            show_legend: bool) -> Any:
        """Create interactive plots using plotly."""
        if n_components == 2:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    f'Before SiteWiseStandardScaler correction ({method.upper()})',
                    f'After SiteWiseStandardScaler correction ({method.upper()})'
                )
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=(
                    f'Before SiteWiseStandardScaler correction ({method.upper()})',
                    f'After SiteWiseStandardScaler correction ({method.upper()})'
                )
            )

        unique_batches = batch_labels.drop_duplicates()

        n_batches = len(unique_batches)
        cmap_func = plt.cm.get_cmap(cmap)
        color_list = [mcolors.to_hex(cmap_func(i / max(n_batches - 1, 1))) for i in range(n_batches)]

        batch_to_color = dict(zip(unique_batches, color_list))

        for batch in unique_batches:
            mask = batch_labels == batch

            if n_components == 2:
                fig.add_trace(
                    go.Scatter(
                        x=X_orig[mask, 0], y=X_orig[mask, 1],
                        mode='markers',
                        name=f'Batch {batch}',
                        marker=dict(
                            size=8,
                            color=batch_to_color[batch],
                            line=dict(width=1, color='black')
                        ),
                        showlegend=False),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=X_trans[mask, 0], y=X_trans[mask, 1],
                        mode='markers',
                        name=f'Batch {batch}',
                        marker=dict(
                            size=8,
                            color=batch_to_color[batch],
                            line=dict(width=1, color='black')
                        ),
                        showlegend=show_legend),
                    row=1, col=2
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=X_orig[mask, 0], y=X_orig[mask, 1], z=X_orig[mask, 2],
                        mode='markers',
                        name=f'Batch {batch}',
                        marker=dict(
                            size=5,
                            color=batch_to_color[batch],
                            line=dict(width=0.5, color='black')
                        ),
                        showlegend=False),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=X_trans[mask, 0], y=X_trans[mask, 1], z=X_trans[mask, 2],
                        mode='markers',
                        name=f'Batch {batch}',
                        marker=dict(
                            size=5,
                            color=batch_to_color[batch],
                            line=dict(width=0.5, color='black')
                        ),
                        showlegend=show_legend),
                    row=1, col=2
                )

        if title is None:
            title = f'SiteWiseStandardScaler correction effect visualized with {method.upper()}'

        fig.update_layout(
            title=title,
            title_font_size=16,
            height=600,
            showlegend=show_legend,
            hovermode='closest'
        )

        axis_labels = [f'{method.upper()}{i + 1}' for i in range(n_components)]

        if n_components == 2:
            fig.update_xaxes(title_text=axis_labels[0])
            fig.update_yaxes(title_text=axis_labels[1])
        else:
            fig.update_scenes(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2]
            )

        return fig