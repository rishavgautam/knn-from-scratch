# Importing the necessary libraries.
import heapq
import pandas  as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class KNN:
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
        
        # To determine which element has more in K selected elements
        liste = [0,0,0,0,0,0]
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
            elif(i[1] == 3): 
                liste[3]+=1
                if(liste[2] > self.MAX_NUMBER):
                    self.MAX_NUMBER = liste[0]
                    self.MAX_ELEMENT = 3
            elif(i[1] == 4): 
                liste[4]+=1
                if(liste[2] > self.MAX_NUMBER):
                    self.MAX_NUMBER = liste[0]
                    self.MAX_ELEMENT = 4
            elif(i[1] == 5): 
                liste[5]+=1
                if(liste[2] > self.MAX_NUMBER):
                    self.MAX_NUMBER = liste[0]
                    self.MAX_ELEMENT = 5
        return self.MAX_ELEMENT



def plot_figures(csv_file, colour):
    """Takes data from an csv file and plots histogram in subplots"""

    # Read csv file and put into DataFrame

    df = pd.read_csv(csv_file, index_col=0)

    # Initiate plot
    
    plt.figure(1,figsize=(30,25))
    
    # Create subplots for each column in the csv file

    plt.subplot(4,3,1)
    sns.distplot(df["danceability"], color=colour)

    plt.subplot(4,3,2)
    sns.distplot(df["energy"], color=colour)

    plt.subplot(4,3,3)
    sns.distplot(df["tempo"], color=colour)

    plt.subplot(4,3,4)
    sns.distplot(df["loudness"], color=colour)

    plt.subplot(4,3,5)
    sns.distplot(df["speechiness"], color=colour)

    plt.subplot(4,3,6)
    sns.distplot(df["acousticness"], color=colour)

    plt.subplot(4,3,7)
    sns.distplot(df["instrumentalness"], color=colour)

    plt.subplot(4,3,8)
    sns.distplot(df["liveness"], color=colour)

    plt.subplot(4,3,9)
    sns.distplot(df["valence"], color=colour)

    plt.subplot(4,3,11)
    sns.distplot(df["key"], color=colour)

    plt.subplot(4,3,12)
    sns.distplot(df["mode"], color=colour)



def dataProcessing():
    spotify_data = pd.read_csv('spotify_dataset.csv')
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
    
    data = data[['danceability', 'mode', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 
        'instrumentalness','liveness', 'valence', 'tempo', 'Genre']]
    
    # plot_figures("./spotify_dataset.csv", colour="darkcyan")
    # plt.savefig('./jjj.png')
    data["key"] = (data["key"] / data["key"].max())
    data["tempo"] = (data["tempo"] / data["tempo"].max())
    data["loudness"] = (data["loudness"] / data["loudness"].min())
    return data
    # data.to_csv('./hhh.csv', index=False, float_format='{:f}'.format, encoding='utf-8')




def main():
    data = dataProcessing()
    data = data.sample(frac=1).reset_index(drop=True)

    # # Training dataset
    df = data[0 : int(len(data)*(3/5))]

    #Testing dataset
    df_test = data[int(len(data)*(3/5))+1 : ]
    
    df = df[0:150]
    df_test = df_test[0:60]

    # Splitting the x and y values in testing dataset
    df_test_x = df_test.to_numpy()[:, :-1]
    df_test_y = df_test.to_numpy()[:, -1]

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
        print(f"k = {k} , success rate = {(controle/len(resultant))*100}")   



if __name__ == "__main__":
    main()