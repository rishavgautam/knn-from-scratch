# Imports
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd


# Class
class k_nearest_neighbors:
    # Initialization
    def __init__(self, n_neighbors=5):  # default neighbors to be returned
        self.n_neighbors = n_neighbors

    # Euclidian Distance
    def euclidean_distance(self, a, b):
        eucl_distance = 0.0  # initializing eucl_distance at 0

        for index in range(len(a)):
            eucl_distance += (a[index] - b[index]) ** 2
            euclidian_distance = np.sqrt(eucl_distance)

        return euclidian_distance

    # Fit k Nearest Neighbors
    def fit_knn(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # Predict X for kNN
    def predict_knn(self, X):
        # initialize prediction_knn as empty list
        prediction_knn = []

        # # initialize euclidian_distances as empty list
        # euclidian_distances = []

        for index in range(len(X)):  # Main loop iterating through len(X)

            # initialize euclidian_distances as empty list
            euclidian_distances = []

            for row in self.X_train:
                # for every row in X_train, find eucl_distance to X using
                # euclidean_distance() and append to euclidian_distances list
                eucl_distance = self.euclidean_distance(row, X[index])
                euclidian_distances.append(eucl_distance)

            # sort euclidian_distances in ascending order, and retain only k
            # neighbors as specified in n_neighbors (n_neighbors = k)
            neighbors = np.array(euclidian_distances).argsort()[: self.n_neighbors]

            # initialize dict to count class occurrences in y_train
            count_neighbors = {}

            for val in neighbors:
                if self.y_train[val] in count_neighbors:
                    count_neighbors[self.y_train[val]] += 1
                else:
                    count_neighbors[self.y_train[val]] = 1

            # max count labels to prediction_knn
            prediction_knn.append(max(count_neighbors, key=count_neighbors.get))

        return prediction_knn

    # Print/display list of nearest_neighbors + corresponding euclidian
    # distance
    def display_knn(self, x):

        # initialize euclidian_distances as empty list
        euclidian_distances = []

        # for every row in X_train, find eucl_distance to x
        # using euclidean_distance() and append to euclidian_distances list
        for row in self.X_train:
            eucl_distance = self.euclidean_distance(row, x)
            euclidian_distances.append(eucl_distance)

        # sort euclidian_distances in ascending order, and retain only k
        # neighbors as specified in n_neighbors (n_neighbors = k)
        neighbors = np.array(euclidian_distances).argsort()[: self.n_neighbors]

        # initiate empty display_knn_values list
        display_knn_values = []

        for index in range(len(neighbors)):
            neighbor_index = neighbors[index]
            e_distances = euclidian_distances[index]
            display_knn_values.append(
                (neighbor_index, e_distances)
            )  # changed to list of tuples
        # print(display_knn_values)
        return display_knn_values


# Instantiate model

data = pd.read_csv('./hhh.csv')
# data = data.sample(frac=1).reset_index(drop=True)
data = data[0:1000]

df_train = data[0 : int(len(data)*(3/5))]
df_test = data[int(len(data)*(3/5))+1 : ]


X_train = df_train.to_numpy()[:, :-1]
y_train = df_train.to_numpy()[:, -1]

#Testing dataset

X_test = df_test.to_numpy()[:, :-1]
y_test = df_test.to_numpy()[:, -1]



classifier = k_nearest_neighbors(n_neighbors=1)
# Fit
classifier.fit_knn(X_train, y_train)
# Prediction
predict = classifier.predict_knn(X_test)
# Accuracy Score
print(f"Build k_nearest_neighbors model accuracy: {accuracy_score(y_test, predict)}")