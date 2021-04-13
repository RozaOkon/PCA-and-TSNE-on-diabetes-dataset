from sklearn import datasets


diabetes = datasets.load_diabetes()

print(list(diabetes.keys()))
print(diabetes.data.shape)
print(len(diabetes.data))
print(diabetes.target.shape)
print(diabetes.feature_names)
print(diabetes.DESCR)