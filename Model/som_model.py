# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:55:16 2022

@author: 91987
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:26:47 2022

@author: 91987
"""
import pandas as pd
import numpy as np
import math
import random
from datetime import datetime
import matplotlib.pyplot as plt

patterns = []
classes = []

# filename = 'C:\\Users\\91987\\Desktop\\data_hmm\\train.csv'
# file = open(filename,'r')

# for line in file.readlines():
#     row = line.strip().split(',')
#     patterns.append(row[0:4])
#     classes.append(row[4])
# print("Iris Data Loaded")


# file.close
df = pd.read_csv("C:\\Users\\91987\\Desktop\\PaperWork\\DriCon\\Model\\train.csv", sep=';')
df = df.drop(columns=['FileName','timestamp', 'City','stopsign','motorcycle','bicycle','timeofday'])

df.isnull().values.any()
df = df.dropna()
classes = df['Ground_Truth_Rating'].tolist()
df = df.drop(columns=['Ground_Truth_Rating'])
df.columns
# patterns = np.asarray(patterns[1:],dtype=np.float32)

# sample_no = np.random.randint(0,len(patterns[1:]))

patterns = np.asarray(df, dtype=np.float32)
sample_no = np.random.randint(0,len(patterns[:]))


print("Sample pattern: " + str(patterns[int(sample_no)]))
print("Class of the above pattern: " + str(classes[int(sample_no)]))

#A heuristic formula for calculating no. of map units
#source: https://stackoverflow.com/questions/19163214/kohonen-self-organizing-maps-determining-the-number-of-neurons-and-grid-size

def mapunits(input_len,size='small'):
    
    heuristic_map_units = 5*input_len**0.54321
     
    if size == 'big':
        heuristic_map_units = 4*(heuristic_map_units)
    else:
        heuristic_map_units = 0.25*(heuristic_map_units)
        
    return heuristic_map_units
        
        
map_units = mapunits(len(patterns),size='big')
print("Heuristically computed appropriate no. of map units: "+str(int(map_units)))

#For reference purpose only - however this function can be used to automatically calculate the SOM dimensions
#from data length. I will still be specifying the SOM dimensions manually, anyway.

import matplotlib.pyplot as plt
# %matplotlib inline

def Eucli_dists(MAP,x):
    x = x.reshape((1,1,-1))
    #print(x)
    Eucli_MAP = MAP - x
    Eucli_MAP = Eucli_MAP**2
    Eucli_MAP = np.sqrt(np.sum(Eucli_MAP,2))
    return Eucli_MAP
len(df.columns)
# input_dimensions = 4
input_dimensions = 21

map_width = 9
map_height = 5
MAP = np.random.uniform(size=(map_height,map_width,input_dimensions))
prev_MAP = np.zeros((map_height,map_width,input_dimensions))

radius0 = max(map_width,map_height)/2
learning_rate0 = 0.1

coordinate_map = np.zeros([map_height,map_width,2],dtype=np.int32)

for i in range(0,map_height):
    for j in range(0,map_width):
        coordinate_map[i][j] = [i,j]

epochs = 500
radius=radius0
learning_rate = learning_rate0
max_iterations = len(patterns)+1
too_many_iterations = 10*max_iterations

convergence = [1]

timestep=1
e=0.001 
flag=0

epoch=0
print(datetime.now())
while epoch<epochs:
    # print("epoch#", epoch)
    shuffle = np.random.randint(len(patterns), size=len(patterns))
    for i in range(len(patterns)):
        
        # difference between prev_MAP and MAP
        J = np.linalg.norm(MAP - prev_MAP)
        #print(J)
        # J = || euclidean distance between previous MAP and current MAP  ||

        if  J <= e: #if converged (convergence criteria)
            flag=1
            break
            
        else:
            
            #if timestep == max_iterations and timestep != too_many_iterations:
            #    epochs += 1
            #    max_iterations = epochs*len(patterns)
            
            pattern = patterns[shuffle[i]]
            pattern_ary = np.tile(pattern, (map_height, map_width, 1))
            Eucli_MAP = np.linalg.norm(pattern_ary - MAP, axis=2)
           
            
            # Get the best matching unit(BMU) which is the one with the smallest Euclidean distance
            BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
            #BMU[1] = np.argmin(Eucli_MAP, 1)[int(BMU[0])]
    
            #Eucli_from_BMU = Eucli_dists(coordinate_map,BMU)  
        
            prev_MAP = np.copy(MAP)
             
            for i in range(map_height):
                for j in range(map_width):
                    distance = np.linalg.norm([i - BMU[0], j - BMU[1]])
                    if distance <= radius:
                        #theta = math.exp(-(distance**2)/(2*(radius**2)))
                        MAP[i][j] = MAP[i][j] + learning_rate*(pattern-MAP[i][j])
            
            learning_rate = learning_rate0*(1-(epoch/epochs))
            #time_constant = max_iterations/math.log(radius) 
            radius = radius0*math.exp(-epoch/epochs)
            #print([learning_rate, radius])
            
            timestep+=1
    
    if J < min(convergence):
        print('Lower error found: %s' %str(J) + ' at epoch: %s' % str(epoch))
        print('\tLearning rate: ' + str(learning_rate))
        print('\tNeighbourhood radius: ' + str(radius))
        MAP_final = MAP
    convergence.append(J)
    
    if flag==1:
        break
    epoch+=1
    
print(datetime.now())    
# Show a plot of the error at each epoch to show convergence, but this is guaranteed in SOM
# due to the learning rate and neighbourhood decay

# Show a plot of the error at each epoch to show convergence, but this is guaranteed in SOM
# due to the learning rate and neighbourhood decay
plt.plot(convergence)
plt.ylabel('Error', size=20, weight='bold')
plt.xlabel('Epoch', size=20, weight='bold')
plt.xticks(size=15, weight='bold')
plt.yticks(size=15, weight='bold')
plt.grid(True)
plt.yscale('log')
plt.show()
print('Number of timesteps: ' + str(timestep))
print('Final error: ' + str(J))
# from scipy.misc import toimage
# from PIL import Image


BMU = np.zeros([2],dtype=np.int32)
result_map = np.zeros([map_height,map_width,3],dtype=np.float32)

i=0
for pattern in patterns:
    print(pattern)
    
    pattern_ary = np.tile(pattern, (map_height, map_width, 1))
    Eucli_MAP = np.linalg.norm(pattern_ary - MAP_final, axis=2)

    # Get the best matching unit(BMU) which is the one with the smallest Euclidean distance
    BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
    
    x = BMU[0]
    y = BMU[1]
    print(x,y) #bmu coordinate
    
    # if classes[i] == 'Iris-setosa':
    #     if result_map[x][y][0] <= 0.5:
    #         result_map[x][y] += np.asarray([0.5,0,0])
    # elif classes[i] == 'Iris-virginica':
    #     if result_map[x][y][1] <= 0.5:
    #         result_map[x][y] += np.asarray([0,0.5,0])
    # elif classes[i] == 'Iris-versicolor':
    #     if result_map[x][y][2] <= 0.5:
    #         result_map[x][y] += np.asarray([0,0,0.5])
    # i+=1
result_map = np.flip(result_map,0)
result_map[x][y]
print(MAP[x][y])  ## bmu weight 
print(MAP.shape)

MAP_reshape = MAP.reshape(MAP.shape[0], -1)
np.savetxt("C:\\Users\\91987\\Desktop\\PaperWork\\DriCon\\Model\\MAP.txt", MAP_reshape)

df_test = pd.read_csv("C:\\Users\\91987\\Desktop\\PaperWork\\DriCon\\Model\\test.csv", sep=';')
df_test = df_test.drop(columns=['FileName','timestamp', 'City','stopsign','motorcycle','bicycle','timeofday'])
df_test.isnull().values.any()
df_test = df_test.dropna()
classes = df_test['Ground_Truth_Rating'].tolist()
df_test = df_test.drop(columns=['Ground_Truth_Rating'])
df_test.columns
f_list = list(df_test.columns)
df_test[0:]
feature = []
klist = []
pattern = np.asarray(df_test, dtype=np.float32)
MAP_load = np.loadtxt("C:\\Users\\91987\\Desktop\\PaperWork\\DriCon\\Model\\MAP.txt")
MAP_loaded = MAP_load.reshape(MAP_load.shape[0], MAP_load.shape[1] // 21, 21) ## divide by # features
#print result_map
print(MAP_loaded== MAP)
for pat in pattern:
#    print("pattern=", pat)
    
    pattern_ary = np.tile(pat, (5, 9, 1))
    Eucli_MAP = np.linalg.norm(pattern_ary - MAP_loaded, axis=2)

    # Get the best matching unit(BMU) which is the one with the smallest Euclidean distance
    BMU = np.unravel_index(np.argmin(Eucli_MAP, axis=None), Eucli_MAP.shape)
    
    x = BMU[0]
    y = BMU[1]
    weight = MAP_loaded[x][y]
    feature = []
    k_list = []
    for k in weight:
        if(k>5):
            # if((k>6 and k<8)or(k>10 and k<20)):
            #     pass
            
            # else:
            ind = list(weight).index(k)
            feature.append(f_list[ind])
            k_list.append(k)
#    print(feature)
#    print(k_list)
    
#    print(x,y) #bmu coordinate
#    print(MAP_loaded[x][y])  ## bmu weight 

fp = open("C:\\Users\\91987\\Desktop\\PaperWork\\DriCon\\Model\\final"+".txt", 'w')
str_write = ""
if(feature==[]):
    print("No events occurred\n")
    fp.write("No events occurred\n")
else:
    for i in feature:
        if(i == 'weaving'):
            str_write+= "The target vehicle is weaving and "
            # print("The target vehicle is weaving and")
            # fp.write("The target vehicle is weaving and")
        if(i == 'swerving'):
            str_write+= "The target vehicle is swerving and "
            # print("The target vehicle is swerving and")
            # fp.write("The target vehicle is swerving and")
        if(i == 'sideslip'):
            str_write+= "The target vehicle is sideslipping and "
            # print("The target vehicle is sideslipping and")
            # fp.write("The target vehicle is sideslipping and")
        if(i == 'Stops_Y' or i == 'Stops_Z'):
            str_write+= "The target vehicle is stopping and "
            # print("The target vehicle is stopping and")
            # fp.write("The target vehicle is stopping and")
        if(i == 'Turns'):
            str_write+= "The target vehicle is turning and "
            # print("The target vehicle is turning and")
            # fp.write("The target vehicle is turning and")
        if(i == 'Jerkiness'):
            str_write+= "The target vehicle is jerking and "
            # print("The target vehicle is jerking and")
            # fp.write("The target vehicle is jerking and")
        
        if(i == 'peds_speed'):
            str_write+= "The pedestrian crossing and "
            # print("The pedestrian crossing and")
            # fp.write("The pedestrian crossing and")
        if(i == 'Congestion'):
            str_write+= "The road is congested and "
            # print("The road is congested and")
            # fp.write("The road is congested and")
        if(i == 'RelSpeed'):
            str_write+= "high speed variation and "
            # print("high speed variation and")
            # fp.write("high speed variation and")
        if(i == 'RelDist'):
            str_write+= "high distance variation and "
            # print("high distance variation and")
            # fp.write("high distance variation and")
        if(i == 'BrakeLight'):
            str_write+= "The preceding vehicle is braking and "
            # print("The preceding vehicle is braking and")
            # fp.write("The preceding vehicle is braking and")
        if(i == 'buses'):
            str_write+= "The buses are stopping and "
            # print("The buses are stopping and")
            # fp.write("The buses are stopping and")
        if(i == 'trafficlight' or i == 'trafficlight_color'):
            str_write+= "The trafficlight transited and "
            # print("The trafficlight transited and")
            # fp.write("The trafficlight transited and")
        if(i == 'truck'):
            str_write+= "The truck are stopping and "
            # print("The truck are stopping")
            # fp.write("The truck are stopping")
        if(i == 'car'):
            str_write+= "The road is congested and "
            # print("The car abruptly stopped")
            # fp.write("The car abruptly stopped")

str_write = str_write[:-4]
fp.write(str_write)    
    
fp.close()
       














        





            