import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from scipy.stats import mode

from sklearn.model_selection import train_test_split

data = pd.read_csv('iphone_purchase_records.csv')
data.info()
data.head()

data = data.drop('Gender', axis=1)
data.head()

X = data.drop('Purchase Iphone', axis=1) 
Y = data['Purchase Iphone']

sns.displot(data, x = 'Salary', hue= 'Purchase Iphone')

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
        print(distances)
        k_nearest_neighbors_list = distance_data.sort_values(by=['distance'], axis=0)[:k_val]
        
        labels = Y_train.loc[k_nearest_neighbors_list.index]
        
        voting = mode(labels).mode[0]
        
        y_hat.append(voting)
    
    return y_hat

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.3 , random_state=42)
y_hat_test = KNN(X_train, X_test, Y_train, Y_test,k_val=3)

accuracy_vals = []

for i in range(1,2):
    y_hat_test = KNN(X_train, X_test, Y_train, Y_test, k_val=i)
    accuracy_vals.append(accuracy_score(Y_test, y_hat_test))

plt.plot(range(1,2), accuracy_vals, color='blue', linestyle= 'dashed', marker='x')

y_hat_test = KNN(X_train, X_test, Y_train, Y_test,k_val=7)
print(accuracy_score(Y_test, y_hat_test))

for i in range(len(y_hat_test)):
    if(y_hat_test[i] == 0):
        plt.scatter(X_test.iloc[i]['Age'], X_test.iloc[i]['Salary'], color='blue')
    if(y_hat_test[i] == 1):
        plt.scatter(X_test.iloc[i]['Age'], X_test.iloc[i]['Salary'], color='orange')
        
plt.style.use('ggplot')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('KNN Test Data')