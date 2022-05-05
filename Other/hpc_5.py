import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('hhh.csv')
# data = data.sample(frac=1).reset_index(drop=True)
data = data.sample(frac=1, random_state=1).reset_index()

data = data[0:1000]
data["key"] = (data["key"] / data["key"].max())
data["tempo"] = (data["tempo"] / data["tempo"].max())
data["loudness"] = (data["loudness"] / data["loudness"].min())
data["danceability"] = (data["danceability"] / data["danceability"].max())
data["energy"] = (data["energy"] / data["energy"].max())
data["speechiness"] = (data["speechiness"] / data["speechiness"].max())
data["acousticness"] = (data["acousticness"] / data["acousticness"].max())
data["instrumentalness"] = (data["instrumentalness"] / data["instrumentalness"].max())
data["liveness"] = (data["liveness"] / data["liveness"].max())
data["valence"] = (data["valence"] / data["valence"].max())

print(data.head())
X = data.drop('Genre', axis = 1)
Y = data['Genre']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)


#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=450)

#Train the model using the training sets
knn.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", accuracy_score(Y_test, y_pred))