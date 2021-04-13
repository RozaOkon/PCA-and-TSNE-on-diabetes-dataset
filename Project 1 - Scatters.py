from sklearn import datasets
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
X = diabetes["data"]
y = diabetes["target"]

for i in range (1, 11):
    plt.subplot(3, 4, i)
    plt.scatter(X[:, i - 1], y, c=y)
    plt.ylim(0, 400)

plt.show()