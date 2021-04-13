import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

diabetes = datasets.load_diabetes()
y = (diabetes["data"][:, 1] == diabetes["data"][1, 1]).astype(np.int)
X = diabetes["target"].reshape(-1, 1)

plt.hist(X[y==0], color = "green")
plt.show()
plt.hist(X[y==1], color = "red")
plt.show()

log_reg = LogisticRegression(random_state=42)
clf = log_reg.fit(X, y)
X_new = np.linspace(20, 350, 1000).reshape(-1, 1)

y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 0], "g-", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "b--", linewidth=2)
plt.show()

# y_proba = log_reg.predict(X_new)
# plt.plot(X_new, y_proba, "g-", linewidth=2)
# plt.plot(X_new, 1 - y_proba, "b--", linewidth=2)
# plt.show()