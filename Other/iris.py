# Importing the necessary libraries.
import heapq
import pandas  as pd
import numpy as np
from sklearn import datasets
from scipy.spatial import distance


# Coding the class structure
class KNN:
    # Defining the data we will use in the class
    def __init__ (self, X, k):
        self.X = X[X.columns[:-1]].values
        self.y = X[X.columns[ -1]].values
        self.k = k
    
    # Application of Euclidean Distance formula to code
    def __distance(self, object_1, object_2):
        return distance.euclidean(object_1, object_2)
    
    def predict(self , data):
        self.data = data 
        distance = []
        counter_row = 0 
    
        # Measuring the Euclidean Distance 
        # and adding it to the list in order from least to most
        for i in self.X:
            dst = self.__distance(self.data, i)
            heapq.heappush(distance, (dst, self.y[counter_row])) 
            counter_row += 1 

        # To determine which element has more in K selected elements
        liste = [0,0,0]
        chech_list = distance[0 : self.k]
        self.MAX_ELEMENT = -1
        self.MAX_NUMBER  = -1
        for i in chech_list:
            if(i[1] == 0): 
                liste[0]+=1
                if(liste[0] > self.MAX_NUMBER):
                    self.MAX_NUMBER  = liste[0]
                    self.MAX_ELEMENT = 0
            elif(i[1] == 1): 
                liste[1]+=1
                if(liste[1] > self.MAX_NUMBER):
                    self.MAX_NUMBER = liste[0]
                    self.MAX_ELEMENT = 1
            elif(i[1] == 2): 
                liste[2]+=1
                if(liste[2] > self.MAX_NUMBER):
                    self.MAX_NUMBER = liste[0]
                    self.MAX_ELEMENT = 2
        print(liste)
        return self.MAX_ELEMENT




def main():
    iris =  datasets.load_iris()
    data = pd.DataFrame(iris.data)
    data["target"]= iris.target
    print(data["target"].unique())
    data = data.sample(frac=1).reset_index(drop=True)
    # Training dataset
    df = data[0 : int(len(data)*(3/5))]

    #Testing dataset
    df_test = data[int(len(data)*(3/5))+1 : ]

    # Splitting the x and y values in testing dataset
    df_test_x = df_test[df_test.columns[:-1]].values
    df_test_y = df_test[df_test.columns[ -1]].values

    for k in range(1, 10):
        model = KNN(df, k) 
        resultant = []
        counter = 0 
        for i in df_test_x:
            resultat = resultant.append(model.predict(i))
            counter +=1 

        #control df_test_y resultat
        controle = 0
        for i in range(len(resultant)):
            if (resultant[i] == df_test_y[i]):
                controle += 1
        print(controle, len(resultant))
        print(f"k = {k} , succes rate = {(controle/len(resultant))*100}")   



if __name__ == "__main__":
    main()