import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# print(iris_dataset['DESCR'][:] + "\n...")

# print("Target names: {}".format(iris_dataset['target_names']))

# SPLIT THE DATA COLLECTED TO TRAINING SET AND TESTING SET (75%, 25%)
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
# pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
# alpha=.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=1)  # assign algorithm.

# MAKING AND TRAINING THE MODEL
knn.fit(X_train, y_train)  # assign arguments: x_train of the data and y_train of the labels.

# MAKING PREDICTIONS
X_new = np.array([[5, 2.9, 1, 0.2]])
# print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
# print("\nPrediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# TESTING THE MODEL
y_pred = knn.predict(X_test)  # WE PASS THE 25% OF THE DATA TO PREDICT IT.
print("\nTest set predictions:\n {}".format(y_pred))  # WE GET LABELS OF EACH ONE.
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))  # WE TEST THE PREDICTED DATA AGAINST THE KNOWN LABELS
