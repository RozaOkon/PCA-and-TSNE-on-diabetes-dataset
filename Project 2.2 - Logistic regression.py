import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

diabetes = datasets.load_diabetes()
X = diabetes["data"].reshape(-1, 10)
y = diabetes["target"]

lin_reg = LogisticRegression()
X_new = np.linspace(20, 350, 1000).reshape(-1, 1)

clf3 = lin_reg.fit(X[:, 2].reshape(-1, 1), y)
y_proba3 = lin_reg.predict_proba(X_new)

clf4 = lin_reg.fit(X[:, 3].reshape(-1, 1), y)
y_proba4 = lin_reg.predict_proba(X_new)

clf9 = lin_reg.fit(X[:, 8].reshape(-1, 1), y)
y_proba9 = lin_reg.predict_proba(X_new)

clf10 = lin_reg.fit(X[:, 9].reshape(-1, 1), y)
y_proba10 = lin_reg.predict_proba(X_new)

clf7 = lin_reg.fit(X[:, 6].reshape(-1, 1), y)
y_proba7 = lin_reg.predict_proba(X_new)

clf1 = lin_reg.fit(X[:, 0].reshape(-1, 1), y)
y_proba1 = lin_reg.predict_proba(X_new)

clf5 = lin_reg.fit(X[:, 4].reshape(-1, 1), y)
y_proba5 = lin_reg.predict_proba(X_new)

clf6 = lin_reg.fit(X[:, 5].reshape(-1, 1), y)
y_proba6 = lin_reg.predict_proba(X_new)

clf8 = lin_reg.fit(X[:, 7].reshape(-1, 1), y)
y_proba8 = lin_reg.predict_proba(X_new)

plt.subplot(3, 3, 2)
plt.plot(X_new, y_proba3, "g-", linewidth=2)
plt.title("BMI")

plt.subplot(3, 3, 3)
plt.plot(X_new, y_proba4, "r-", linewidth=2)
plt.title("Blood pressure")

plt.subplot(3, 3, 8)
plt.plot(X_new, y_proba9, "y-", linewidth=2)
plt.title("Lamotrigine level")

plt.subplot(3, 3, 9)
plt.plot(X_new, y_proba10, "b-", linewidth=2)
plt.title("Glucose level")

plt.subplot(3, 3, 6)
plt.plot(X_new, y_proba7, "k-", linewidth=2)
plt.title("High-density lipoproteins")

plt.subplot(3, 3, 1)
plt.plot(X_new, y_proba1, "g-", linewidth=2)
plt.title("Age")

plt.subplot(3, 3, 4)
plt.plot(X_new, y_proba5, "y-", linewidth=2)
plt.title("T-Cells  level")

plt.subplot(3, 3, 5)
plt.plot(X_new, y_proba6, "b-", linewidth=2)
plt.title("Low-density lipoproteins")

plt.subplot(3, 3, 7)
plt.plot(X_new, y_proba8, "k-", linewidth=2)
plt.title("Thyroid stimulating hormone")
plt.show()

# for i in range(1, 11):
#     clf = lin_reg.fit(X[:, i - 1].reshape(-1, 1), y)
#     y_proba = lin_reg.predict(X_new)
#     plt.subplot(3, 4, i)
#     plt.plot(X_new, y_proba, "k-", linewidth=2)
# plt.show()