import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pendulum import time
import seaborn as sns
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

data = pd.read_csv('hhh.csv')
data = data.sample(frac=1, random_state=1).reset_index(drop=True)
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


def euclidean_distance(pt1, pt2):
    distance = np.sqrt(np.sum((pt1-pt2)**2))
    return distance

def KNN(X_train, X_test, Y_train, Y_test, k_val):
    y_hat = []
    for test_pt in X_test.to_numpy():
        distances = []
        for i in range(len(X_train)):
            distances.append(euclidean_distance((np.array(X_train.iloc[i])), test_pt))
        
        distance_data = pd.DataFrame(data = distances, columns=['distance'], index = Y_train.index)
        k_nearest_neighbors_list = distance_data.sort_values(by=['distance'], axis=0)[:k_val]
        
        labels = Y_train.loc[k_nearest_neighbors_list.index]
        
        voting = mode(labels).mode[0]
        
        y_hat.append(voting)
    
    return y_hat

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
# y_hat_test = KNN(X_train, X_test, Y_train, Y_test,k_val=3)

accuracy_vals = []
time_taken = []

for i in range(1,10):
    start = time.time()
    y_hat_test = KNN(X_train, X_test, Y_train, Y_test, k_val=i)
    accuracy_vals.append(accuracy_score(Y_test, y_hat_test))
    elapsed_time = (time.time() - start) 
    time_taken.append(elapsed_time)

print(accuracy_vals)
print(time_taken)
# y_hat_test = KNN(X_train, X_test, Y_train, Y_test,k_val=7)
# print("The accuracy score is", accuracy_score(Y_test, y_hat_test))
