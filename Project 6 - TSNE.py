import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import plotly.express as px
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.manifold import TSNE

diabetes = datasets.load_diabetes()
X = diabetes["data"]
y = diabetes["target"]
n_components = 3


tsne = TSNE(n_components=n_components)
pca_2d = tsne.fit_transform(X)

fig = px.scatter_matrix(pca_2d, color=y, dimensions=range(n_components))
fig.update_traces(diagonal_visible=False)
fig.show()
