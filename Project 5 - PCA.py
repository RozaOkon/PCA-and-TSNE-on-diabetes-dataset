import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn import datasets
np.random.seed(203)

diabetes = datasets.load_diabetes()
n_components = 3
X = diabetes["data"]
y = diabetes["target"]

pca = PCA(n_components=n_components)
pca_2d = pca.fit_transform(X)

fig = px.scatter_matrix(pca_2d, color=y, dimensions=range(n_components))
fig.update_traces(diagonal_visible=False)
fig.show()
