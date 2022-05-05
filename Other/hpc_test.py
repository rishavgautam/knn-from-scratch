# Importing the necessary libraries.
import heapq
import pandas  as pd
import numpy as np
pd.options.display.float_format = '{:.2f}'.format



# Coding the class structure
class KNN:
    # Defining the data we will use in the class
    def __init__ (self, X, k):
        self.X = X.to_numpy()[:, :-1]
        self.y = X.to_numpy()[:, -1]
        self.k = k
        

    
    # Applying Euclidean Distance to calculate neighbors
    def __distanceCalc(self, object_1, object_2):
        dist = np.sqrt(np.sum(np.square(object_1-object_2)))
        return dist
    
    def predict(self , data):
        self.data = data
        distance = []
        counter_row = 0 
    
        # Measuring the Euclidean Distance 
        # and adding it to the list in order from least to most
        for i in self.X:
            dst = self.__distanceCalc(self.data, i)
            heapq.heappush(distance, (dst, self.y[counter_row]))
            counter_row += 1 

        print(distance)
       
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
        return self.MAX_ELEMENT


def dataProcessing():
    spotify_data = pd.read_csv('./spotify_dataset.csv')
    data = pd.DataFrame(spotify_data)
    data["Genre"]= data["playlist_genre"]
    x = data["Genre"].unique().tolist()
    y = list(range(0, len(x)))
    res = dict(zip(x, y))
    data['Genre'] = np.where((data.Genre == 'rock'), 0, data.Genre)
    data['Genre'] = np.where((data.Genre == 'pop'), 1, data.Genre)
    data['Genre'] = np.where((data.Genre == 'r&b'), 2, data.Genre)
    data['Genre'] = np.where((data.Genre == 'latin'), 3, data.Genre)
    data['Genre'] = np.where((data.Genre == 'edm'), 4, data.Genre)
    data['Genre'] = np.where((data.Genre == 'rap'), 5, data.Genre)
    
    data = data[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness','liveness', 'valence', 'tempo', 'Genre']]
    return data
    # data.to_csv('./hhh.csv', index=False, float_format='{:f}'.format, encoding='utf-8')




def main():
    # spotify_data = pd.read_csv('./hhh.csv')
    data = dataProcessing()
    data = data.sample(frac=1).reset_index(drop=True)

    # # Training dataset
    df = data[0 : int(len(data)*(4/5))]

    #Testing dataset
    df_test = data[int(len(data)*(4/5))+1 : ]
    df = df[0:10]
    df_test = df_test[0:1]

    # Splitting the x and y values in testing dataset
    df_test_x = df_test.to_numpy()[:, :-1]
    df_test_y = df_test.to_numpy()[:, -1]

    df_test_data = df_test.to_numpy()

    for k in range(1, 2):
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

        print(f"k = {k} , succes rate = {(controle/len(resultant))*100}")   



if __name__ == "__main__":
    main()