"""
Author: Aryan V S (https://github.com/a-r-r-o-w)
License: MIT License
Description: This file contains the class DataVisualizer and its derived subclasses,
which provides methods for visualizing data. The following visualization are supported:
    - Scatter Plot
    - Heatmap
    - PCA
    - t-SNE
    - UMAP
    - Scatter Plot Matrix
    - Parallel Coordinates Plot
    - Histogram
    - Andrews Curve
    - RadViz

This code was written with the help of OpenAI's ChatGPT, GitHub Copilot, StackOverflow,
and the following resources:
    - https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction
    - https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
    - https://www.analyticsvidhya.com/blog/2022/09/data-visualization-guide-for-multi-dimensional-data/
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import sklearn.decomposition
import sklearn.manifold
import umap


class DataVisualizer:
    """This class provides methods for visualizing data."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        plt.style.use("fivethirtyeight")

    def _check_dimensions(self, n_dimensions):
        if n_dimensions not in [2, 3]:
            raise ValueError("For visualization, n_dimensions must be 2 or 3.")

    def _scatter_plot(self, n_dimensions=2):
        self._check_dimensions(n_dimensions)

        if n_dimensions == 2:
            if self.data.shape[1] > 3:
                pca = sklearn.decomposition.PCA(n_components=2)
                data_2d = pca.fit_transform(self.data)
            else:
                data_2d = self.data
            fig = px.scatter(x=data_2d[:, 0], y=data_2d[:, 1], color=self.labels)
            fig.update_layout(
                title="2D Scatter Plot",
                xaxis_title="Component 1",
                yaxis_title="Component 2",
            )
            fig.show()
        else:
            if self.data.shape[1] > 3:
                pca = sklearn.decomposition.PCA(n_components=3)
                data_3d = pca.fit_transform(self.data)
            else:
                data_3d = self.data
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=data_3d[:, 0],
                        y=data_3d[:, 1],
                        z=data_3d[:, 2],
                        mode="markers",
                        marker=dict(color=self.labels, colorscale="Viridis"),
                    )
                ]
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3",
                ),
                title="3D Scatter Plot",
            )
            fig.show()

    def _heatmap(self):
        sns.heatmap(self.data, cmap="coolwarm", annot=False)
        plt.title("Heatmap")
        plt.show()

    def _pca(self, n_dimensions=2):
        self._check_dimensions(n_dimensions)

        pca = sklearn.decomposition.PCA(n_components=n_dimensions)
        data_reduced = pca.fit_transform(self.data)
        if n_dimensions == 2:
            fig = px.scatter(
                x=data_reduced[:, 0], y=data_reduced[:, 1], color=self.labels
            )
            fig.update_layout(
                title="2D PCA Visualization",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
            )
            fig.show()
        else:
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=data_reduced[:, 0],
                        y=data_reduced[:, 1],
                        z=data_reduced[:, 2],
                        mode="markers",
                        marker=dict(color=self.labels, colorscale="Viridis"),
                    )
                ]
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="Principal Component 1",
                    yaxis_title="Principal Component 2",
                    zaxis_title="Principal Component 3",
                ),
                title="3D PCA Visualization",
            )
            fig.show()

    def _tsne(self, n_dimensions=2):
        self._check_dimensions(n_dimensions)

        tsne = sklearn.manifold.TSNE(n_components=n_dimensions)
        data_reduced = tsne.fit_transform(self.data)
        if n_dimensions == 2:
            fig = px.scatter(
                x=data_reduced[:, 0], y=data_reduced[:, 1], color=self.labels
            )
            fig.update_layout(
                title="2D t-SNE Visualization",
                xaxis_title="t-SNE Component 1",
                yaxis_title="t-SNE Component 2",
            )
            fig.show()
        else:
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=data_reduced[:, 0],
                        y=data_reduced[:, 1],
                        z=data_reduced[:, 2],
                        mode="markers",
                        marker=dict(color=self.labels, colorscale="Viridis"),
                    )
                ]
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="t-SNE Component 1",
                    yaxis_title="t-SNE Component 2",
                    zaxis_title="t-SNE Component 3",
                ),
                title="3D t-SNE Visualization",
            )
            fig.show()

    def _umap(self, n_dimensions=2):
        self._check_dimensions(n_dimensions)

        reducer = umap.UMAP(n_components=n_dimensions)
        data_reduced = reducer.fit_transform(self.data)
        if n_dimensions == 2:
            fig = px.scatter(
                x=data_reduced[:, 0], y=data_reduced[:, 1], color=self.labels
            )
            fig.update_layout(
                title="2D UMAP Visualization",
                xaxis_title="UMAP Component 1",
                yaxis_title="UMAP Component 2",
            )
            fig.show()
        else:
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=data_reduced[:, 0],
                        y=data_reduced[:, 1],
                        z=data_reduced[:, 2],
                        mode="markers",
                        marker=dict(color=self.labels, colorscale="Viridis"),
                    )
                ]
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="UMAP Component 1",
                    yaxis_title="UMAP Component 2",
                    zaxis_title="UMAP Component 3",
                ),
                title="3D UMAP Visualization",
            )
            fig.show()

    def _splom(self):
        df = pd.DataFrame(
            self.data, columns=[f"Attribute_{i+1}" for i in range(self.data.shape[1])]
        )
        pd.plotting.scatter_matrix(df, c=self.labels, cmap="viridis")
        plt.title("Scatter Plot Matrix (SPLOM)")
        plt.show()

    def _parallel_coordinates(self):
        df = pd.DataFrame(
            self.data, columns=[f"Attribute_{i+1}" for i in range(self.data.shape[1])]
        )
        df["Labels"] = self.labels
        plt.figure()
        parallel_coordinates_plot = pd.plotting.parallel_coordinates(
            df, "Labels", colormap="viridis"
        )
        plt.title("Parallel Coordinates Plot")
        plt.show()

    def _histogram(self):
        for i in range(self.data.shape[1]):
            plt.hist(self.data[:, i], bins=20, alpha=0.5, label=f"Attribute_{i+1}")
        plt.xlabel("Attribute Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Histogram")
        plt.show()

    def _andrews_curves(self):
        df = pd.DataFrame(
            self.data, columns=[f"Attribute_{i+1}" for i in range(self.data.shape[1])]
        )
        df["Labels"] = self.labels
        pd.plotting.andrews_curves(df, "Labels", colormap="viridis")
        plt.title("Andrews Curves")
        plt.show()

    def _radviz(self):
        df = pd.DataFrame(
            self.data, columns=[f"Attribute_{i+1}" for i in range(self.data.shape[1])]
        )
        df["Labels"] = self.labels
        ax = plt.subplot(111)
        pd.plotting.radviz(df, "Labels", colormap="viridis", ax=ax)
        plt.title("RadViz")
        plt.show()

    def visualize(self, visualization_type, n_dimensions=2):
        if visualization_type == "scatter":
            self._scatter_plot(n_dimensions)
        elif visualization_type == "heatmap":
            self._heatmap()
        elif visualization_type == "pca":
            self._pca(n_dimensions)
        elif visualization_type == "tsne":
            self._tsne(n_dimensions)
        elif visualization_type == "umap":
            self._umap(n_dimensions)
        elif visualization_type == "splom":
            self._splom()
        elif visualization_type == "parallel_coordinates":
            self._parallel_coordinates()
        elif visualization_type == "histogram":
            self._histogram()
        elif visualization_type == "andrews_curves":
            self._andrews_curves()
        elif visualization_type == "radviz":
            self._radviz()
        else:
            raise ValueError(
                "Invalid visualization_type. Choose from 'scatter', 'heatmap', 'pca', 'tsne', 'umap', 'splom', 'parallel_coordinates', 'histogram', 'andrews_curves', or 'radviz'."
            )


class ScatterPlotVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self, n_dimensions=2):
        return super()._scatter_plot(n_dimensions)


class HeatmapVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self):
        return super()._heatmap()


class PCAVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self, n_dimensions=2):
        return super()._pca(n_dimensions)


class TSNEVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self, n_dimensions=2):
        return super()._tsne(n_dimensions)


class UMAPVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self, n_dimensions=2):
        return super()._umap(n_dimensions)


class SPLOMVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self):
        return super()._splom()


class ParallelCoordinatesVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self):
        return super()._parallel_coordinates()


class HistogramVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self):
        return super()._histogram()


class AndrewsCurvesVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self):
        return super()._andrews_curves()


class RadVizVisualizer(DataVisualizer):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def visualize(self):
        return super()._radviz()
