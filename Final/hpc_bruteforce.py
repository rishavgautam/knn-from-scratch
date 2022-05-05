import pandas  as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


k = [1,2,3,4,5,6,7,8,9]

#these values are stored from the console after it was generated
# time_taken = [211.236746324, 340.456971823, 497.34589043587, 774.53674539485, 1208.26347382498131, 
# 1620.45654896234, 2340.58946124934, 3256.657098902, 3901.43578348953]



# plt.plot(time_taken)
# plt.ylabel('Time Taken (seconds)')
# plt.xlabel('K - value')
# plt.savefig('./brute_force_time-taken.png')



# Load the data
data = pd.read_csv("spotify_dataset.csv")
data.drop_duplicates(subset ="track_id", keep = False, inplace = True)

features_indices = [1,11,12,13,14,15,16,17,18,19,20,21]
features = data.iloc[:,features_indices]
genre = data.iloc[:,[0,9]]

k = 1

def calcDistance(point1, point2):
    dist = np.sqrt(np.sum(np.square(point1-point2)))
    return dist
def most_frequent(List):
	return max(set(List), key = List.count)


features_train,features_test,genre_train,genre_test = train_test_split(features,genre,test_size=0.2,random_state=0)
#distance = []
#trackId = []
test_genres = []
song_names = []
predicted = []

for i in range(features_test.shape[0]):
    distance = []
    trackId = []
    #test_genres.append(features_test.iloc[i,:])
    for j in range(features_train.shape[0]):
        distance.append(calcDistance(features_test.iloc[i,1:],features_train.iloc[j,1:]))
        trackId.append(features_train.iloc[j,0])
    
    
    distanceTrackMap = pd.DataFrame(zip(trackId,distance))
    distanceTrackMap = distanceTrackMap.sort_values(by = 1)
    
    for m in range(k):
        song_names.append(distanceTrackMap.iloc[m,0])
        
    predicted.append(most_frequent(song_names))  
    distance = []
    trackId=[]
     
#print(predicted) # list of songs
#Run this loop to print predicted genre for each test song

for i in range(len(predicted)):
    print(data.loc[data['track_name']==predicted[i]]['playlist_genre'].values)