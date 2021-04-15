import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import plot_confusion_matrix

# sns.set_theme(color_codes=True)
diabetes = datasets.load_diabetes()
X = diabetes["data"][:, :4]

g = sns.clustermap(X)
print(X.shape)

plt.show()
