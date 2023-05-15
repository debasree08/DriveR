import os
import requests
import sys
import time
import pandas as pd
import numpy as np

time_s = time.process_time()
url = "http://192.168.43.32:8000/process-video"
#url = "http://10.5.20.192:8000/process-video"
file = sys.argv[1]
videopath = os.getcwd() + '/' + file
#
files=[('file',('testvid.mp4',open(videopath,'rb'),'application/octet-stream'))]
#
response = requests.request("POST", url, headers={}, data={}, files=files)
##response = requests.request("GET", url, headers={}, data={})
#
print(response.text)
#
file1 = open("object.csv","w")
res = response.text.split('\r')
for temp in res:
    file1.writelines(temp)
file1.close()
print("Response Time", time.process_time()-time_s)

#from threading import Thread

#def func1():
#    print('Working')

#def func2():
#    print("Working")

#if __name__ == '__main__':
#    Thread(target = func1).start()
#    Thread(target = func2).start()
print(file[:-4])
df1 = pd.read_csv("object.csv", header=0)
df2 = pd.read_csv("man"+file[:-4]+".csv", header=0)

df3 = df1.merge(df2, left_on="Unnamed: 0", right_on="Unnamed: 0")
df4 = pd.read_csv("train.csv", header=0)
df4.columns
df3.columns
len_list = len(df3['Unnamed: 0'].tolist())
emp_list = []
for i in range(len_list):
    emp_list.append(0)
df5 = pd.DataFrame(columns=['peds_speed', 'weaving', 'swerving','sideslip','Congestion',
       'Jerkiness', 'RelSpeed', 'RelDist', 'BrakeLight','Turns', 'Stops_Y', 'Stops_Z','person',
       'car', 'buses', 'trafficlight', 'trafficlight_color','truck'])
df5['peds_speed'] = emp_list
df5['weaving'] = emp_list
df5['swerving'] = emp_list
df5['sideslip'] = emp_list
df5['Congestion'] = emp_list
df5['Jerkiness'] =  df3['jerkiness']
df5['RelSpeed'] = emp_list
df5['RelDist'] = emp_list
df5['BrakeLight'] = emp_list
df5['Turns'] = emp_list
df5['Stops_Y'] =  df3['stopy']
df5['Stops_Z'] =  df3['stopz']
df5['person'] = df3['Persons']
df5['car'] = df3['Cars']
df5['buses'] =  df3['Buses']
df5['trafficlight'] = df3['Traffic Lights']
df5['trafficlight_color'] = emp_list
df5['truck'] = df3['Trucks']
f_list = list(df5.columns)
feature = []
klist = []
pattern = np.asarray(df5, dtype=np.float32)
MAP_load = np.loadtxt("/home/pi/Downloads/MAP.txt")
MAP_loaded = MAP_load.reshape(MAP_load.shape[0], MAP_load.shape[1] // 18, 18)
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
            if((k>40 and k<50)or(k>10 and k<20)):
                pass
            
            else:
                ind = list(weight).index(k)
                feature.append(f_list[ind])
                k_list.append(k)
#    print(feature)
#    print(k_list)
    
#    print(x,y) #bmu coordinate
#    print(MAP_loaded[x][y])  ## bmu weight 
    fp = open("final"+file[:-4]+".txt", 'a')
    if(feature==[]):
        print("No events occurred\n")
        fp.write("No events occurred\n")
    else:
        for i in feature:
            if(i == 'weaving'):
                print("The target vehicle is weaving and")
                fp.write("The target vehicle is weaving and")
            if(i == 'swerving'):
                print("The target vehicle is swerving and")
                fp.write("The target vehicle is swerving and")
            if(i == 'sideslip'):
                print("The target vehicle is sideslipping and")
                fp.write("The target vehicle is sideslipping and")
            if(i == 'Stops_Y' or i == 'Stops_Z'):
                print("The target vehicle is stopping and")
                fp.write("The target vehicle is stopping and")
            if(i == 'Turns'):
                print("The target vehicle is turning and")
                fp.write("The target vehicle is turning and")
            if(i == 'Jerkiness'):
                print("The target vehicle is jerking and")
                fp.write("The target vehicle is jerking and")
            
            if(i == 'peds_speed'):
                print("The pedestrian crossing and")
                fp.write("The pedestrian crossing and")
            if(i == 'Congestion'):
                print("The road is congested and")
                fp.write("The road is congested and")
            if(i == 'RelSpeed'):
                print("high speed variation and")
                fp.write("high speed variation and")
            if(i == 'RelDist'):
                print("high distance variation and")
                fp.write("high distance variation and")
            if(i == 'BrakeLight'):
                print("The preceding vehicle is braking and")
                fp.write("The preceding vehicle is braking and")
            if(i == 'buses'):
                print("The buses are stopping and")
                fp.write("The buses are stopping and")
            if(i == 'trafficlight' or i == 'trafficlight_color'):
                print("The trafficlight transited and")
                fp.write("The trafficlight transited and")
            if(i == 'truck'):
                print("The truck are stopping")
                fp.write("The truck are stopping")
    fp.close()
   




