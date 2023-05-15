#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 20:00:10 2022

@author: user
"""


import sys
#import __init__
import numpy as np
from numpy import var, arange
import pandas as pd
from datetime import datetime
import time
import math

from scipy import linalg, signal
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
import glob, os, shutil


#limit = 3.0
#limit = 0.5
#ylimit = 5.0
#zlimit = 10.5
#tlimit = 20.0

limit = 3.0
#ylimit = 5.0
ylimit = 0.1
#zlimit = 10.5
zlimit = 0.05
tlimit = 20.0

lowess = sm.nonparametric.lowess

def lowess(x, y, f=2./3., iter=3):
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations."""
    n = len(x)
    r = int(np.ceil(f*n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                   [np.sum(weights*x), np.sum(weights*x*x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2

    return yest

def preprocessor(sensor):
    x = np.linspace(0,len(sensor),len(sensor))
    b,a = signal.butter(1,0.3)
    bs = signal.filtfilt(b,a,sensor)
    f = 0.2
    return lowess(x,bs,f,3)


def sec_to_hour(time):
    hh=int(time/3600)
    mm =int( ((time/float(3600)) - hh)*60)
    ss = int(((((time/float(3600)) - hh)*60) - mm)*60)
    return (str("{:02d}".format(hh))+':'+str("{:02d}".format(mm))+':'+str("{:02d}".format(ss)))

    


def sec_to_date_time(timestamp):
    #print("Helllo")
    #timestamp1 = 1503828254943
    #timestamp2 = 1503828295206
    timestamp1 = int(timestamp) ## do not divide by 10000
    print(timestamp1)
    #timestamp2 = int(timestamp2/1000)
    #print(timestamp2)
    dt_object = datetime.fromtimestamp(timestamp1)
    
    #print("dt_object =", dt_object)
    #print("type(dt_object) =", type(dt_object))
    #dt_object = datetime.fromtimestamp(timestamp2)
    
    print("dt_object =", dt_object)
    #print("type(dt_object) =", type(dt_object))
    sec = dt_object.strftime("%S")
    mint = dt_object.strftime("%M")
    print("//")
    return (int(sec)+(int(mint)*60))

def removNestings(l): 
    output=list()
    #print(l)
    for i in l: 
        if type(i) == list: 
            z=str(i).strip('[ ]')
            z=z.strip('\'')
            print(z)
            
            output.append(float(z[:]))
            
            
        else: 
            output.append(i) 
    #print(output)
    return output


#def sec_to_date_time(timestamp):
#    #print("Helllo")
#    #timestamp1 = 1503828254943
#    #timestamp2 = 1503828295206
#    # timestamp = 1657803728801660
#
#    timestamp1 = int(int(1662716303)/1000000)
##    print(timestamp1)
#    #timestamp2 = int(timestamp2/1000)
#    #print(timestamp2)
#    dt_object = datetime.fromtimestamp(timestamp1)
#    
#    #print("dt_object =", dt_object)
#    #print("type(dt_object) =", type(dt_object))
#    #dt_object = datetime.fromtimestamp(timestamp2)
#    
#    # print("dt_object =", dt_object)
#    #print("type(dt_object) =", type(dt_object))
#    sec = dt_object.strftime("%S")
#    mint = dt_object.strftime("%M")
#    return (int(sec)+(int(mint)*60))

    

        
################################################## detect jerkiness ########################################


def call_jerkiness(filepath, filename):
    # filepath = 'D:\\nexar_baddrive\\without_audio\\imu\\jerkiness\\20220714_183207A.csv'
    # filename = '20220714_183207A'
#    df=pd.read_csv(filepath, sep=';', header = 0)
    df = pd.read_csv(filepath, sep=",", header=None)
    df.columns = ['timestamp', 'accx', 'accy', 'accz']
#    print(df.head(10))
#    df = df.drop(columns=['gyroy', 'gyroz', 'gyrox'])
    #df.columns=['time','lat','long','speed']
    # df.columns=['time','valuex','valuey','valuez']
    print(df.head(10))
    time=df[['timestamp']].values
    time=time.tolist()
    ts=list()
    ts1=0
    time = removNestings(time)
    for i in time:
        temp= i
        ts1 = sec_to_date_time(i)
        ts.append(ts1)
    #for i in range(len(ts)):  
     #   print(ts[i])
    #print(len(ts))
    #print(time)
    sensorx = df[['accz']].values
    sensorx = sensorx.tolist()
   # print(sensorz)
    sensorx1 = removNestings(sensorx)
    
    
    dict1 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict1:
            dict1[ts[i]]= sensorx[i]
        else:
            dict1[ts[i]].append(sensorx[i])
    #print(len(dict1.keys()))
    length = len(dict1.keys())
    keys=sorted(dict1.keys())
    len(keys)
    listj = []
    i=0
    print('******************jerkiness Detection Results*************')
    
    while(i<length-1):
        #print(keys[i])
        #print(i, removNestings(dict1[i]))
        op1=removNestings(dict1[keys[i]])
        op2 = removNestings(dict1[keys[i+1]])
        op1 = np.mean(op1)
        op2 = np.mean(op2)
#        print(op1,op2)
#        print(((op2-op1)/(keys[i+1] - keys[i])))
        var_temp = ((op2-op1)/(keys[i+1] - keys[i]))
        i = i + 1
        print(i, '-->', var_temp)
        listj.append(var_temp)
    return listj
#        fp = open('D:\\nexar_baddrive\\without_audio\\imu\\jerkiness\\values\\'+ filename + '.csv', 'a')
#        fp.write(str(filename) + ',' + str(var_temp) + '\n')
#        fp.close()       
        

############################### at each 5 sec do average #####################################
#os.chdir("D:\\nexar_baddrive\\without_audio\\imu\\stopy\\") 
#arr = os.listdir()          
#for line in arr:
#    str1 = line.strip()
#    str1 = str1[:-4]
#    str2 = "D:\\nexar_baddrive\\without_audio\\imu\\jerkiness\\values\\" + str1 + ".csv"
#    # print(str1, '\n')
#    df = pd.read_csv(str2, header = None)
#    time1 = df[0]
#    rating = df[1]
#    length = len(time1)
#    count = 1
#    ratings = 0
#    #print(length)
#    if(length == 61):
#        for i in range(0,61):
#            ratings = ratings + rating[i] 
#            if(i==4 or i==9 or i==14 or i==19 or i==24 or i==29 or i==34 or i==39 or i==44
#               or i==49 or i==54 or i==60):
#                
#                if(i==60):
#                    ratings = ratings / 6
#                else:
#                    ratings = ratings / 5
#                    
#                print(ratings)
#                
#                # if(ratings>= 0.5):
#                #     print( 1)
#                # else:
#                #     print( 0)
#                    
#                ratings = 0      
#    elif(length == 60):
#        for i in range(0,60):
#            ratings = ratings + rating[i] 
#            if(i==4 or i==9 or i==14 or i==19 or i==24 or i==29 or i==34 or i==39 or i==44
#               or i==49 or i==54 or i==59):
#                
#                if(i==59):
#                    ratings = ratings / 5
#                else:
#                    ratings = ratings / 5
#                    
#                print(ratings)
#                
#                # if(ratings>= 0.5):
#                #     print( 1)
#                # else:
#                #     print( 0)
#                    
#                ratings = 0        

##################### stops data ###################################################


def detect_stopy(sensory, sensorz):
    return max(sensory) ## this line is for stops_y
    # fp.write(str(max(sensory)) + ',')
    # l = list((filter(lambda z: z <= zlimit, sensorz))) ## this line is for logging the stops_z
    # return len(l) ## this line is for logging stops_z
    # # fp.write(str(len(l)) + '\n')
#     if max(sensory) < ylimit:
# 		#if len(filter(lambda z: z >= zlimit, sensorz)) < 3:
#         l = list((filter(lambda z: z <= zlimit, sensorz)))
#         if(len(l) > 3):
#             #print(len(l))
#             return True
def detect_stopz(sensory, sensorz):
     l = list((filter(lambda z: z <= zlimit, sensorz))) ## this line is for logging the stops_z
     return len(l) ## this line is for logging stops_z      
def call_stopper(filepath, filename):
    df = pd.read_csv(filepath, sep=",", header=None)
    df.columns = ['timestamp', 'accx', 'accy', 'accz']
#    df=pd.read_csv(filepath, sep=',', header = 0)
    # print(df.head(10))
#    df = df.drop(columns=['gyroy', 'gyroz', 'gyrox'])
    #df.columns=['time','lat','long','speed']
    # df.columns=['time','valuex','valuey','valuez']
    # print(df.head(10))
    time=df[['timestamp']].values
    time=time.tolist()
    ts=list()
    ts1=0
    time = removNestings(time)
    for i in range(len(time)):
        temp= time[i]
        ts1 = sec_to_date_time(time[i])
        ts.append(ts1)
    #for i in range(len(ts)):  
     #   print(ts[i])
    #print(len(ts))
    #print(time)
    sensorz = df[['accx']].values
    sensorz = sensorz.tolist()
   # print(sensorz)
    sensorz1 =removNestings(sensorz)
    sensory = df[['accy']].values
    sensory = sensory.tolist()
    
    dict1 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict1:
            dict1[ts[i]]= sensorz[i]
        else:
            dict1[ts[i]].append(sensorz[i])
    #print(len(dict1.keys()))
    length = len(dict1.keys())
    keys=sorted(dict1.keys())
    dict2 = dict()
    for i in range(len(ts)):
        
        if ts[i] not in dict2:
            dict2[ts[i]]= sensory[i]
        else:
            dict2[ts[i]].append(sensory[i])
    #print(len(dict2.keys()))
    length = len(dict2.keys())
    keys=sorted(dict2.keys())
    #print(removNestings(dict2[keys[0]]))
    #print(removNestings(dict2[keys[1]]))
    
    #print(sensorz1)
    listy = []
    listz = []
    i=0
    print('******************Stop Detection Results*************')
    
    while(i!=length):
        #print(keys[i])
        #print(i, removNestings(dict1[i]))
        op1=removNestings(dict1[keys[i]])
        op1 = np.subtract(op1, 9.81)
#        print(op1)
        opy = removNestings(dict2[keys[i]])
        var_temp1 = detect_stopy(opy, op1)
        listy.append(var_temp1)
        var_temp2 = detect_stopz(opy, op1)
        listz.append(var_temp2)
#        print(i, "-->", var_temp)
#        fp = open('D:\\nexar_baddrive\\without_audio\\imu\\stopy\\'+ filename + '.csv', 'a')
#        fp.write(str(filename) + ',' + str(var_temp) + '\n')
#        fp.close()
        # fp.write(filename + ',')   
        # fp.write(str(i+1) + ',')
        #print("++",op1)
        # if(detect_stop(opy, op1)):
            
        #     print(i,'',keys[i],"At timestamp detected stop")
        # else:
            
            
        #     print(i,'',keys[i],"Stop not detected")
        i=i+1

    return [listy,listz]

#os.chdir("D:\\nexar_baddrive\\without_audio\\imu\\jerkiness\\values\\")
#arr = os.listdir()
#os.chdir("D:\\nexar_baddrive\\without_audio\\imu\\")
#for i in arr:
#    call_stopper('D:\\nexar_baddrive\\without_audio\\imu\\jerkiness\\'+i[:-4]+'.csv', i[:-4])
#    


########################### fetch turn data ################################
import time
def convert(seconds):
    # seconds = seconds % (3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)
def detect_turn(filepath, filename):

        df = pd.read_csv(filepath, header=1, sep=';')
        #print(df.columns)
        time = df['YYYY-MO-DD HH-MI-SS_SSS']
        hrlist = []
        for i in time.tolist():
            hrs = i.split(' ')
            minute = hrs[1].split(':')
        #    print(minute[0],':',minute[1],':',minute[2])
            hrs1 = int(minute[0])*3600 + int(minute[1])*60 + int(minute[2])
        #    print(hrs1)
            hrlist.append(hrs1)
        df['hrlist'] = hrlist
        
        dict1 = dict()
        latitude = df[['LOCATION Latitude : ']].values
        latitude = latitude.tolist()
        for i in range(len(hrlist)):
        #    print(hrlist[i])
            if hrlist[i] not in dict1:
                dict1[hrlist[i]] = latitude[i]
            else:
                dict1[hrlist[i]].append(latitude[i])
        
        dict2 = dict()
        longitude = df[['LOCATION Longitude : ']].values
        longitude = longitude.tolist()
        for i in range(len(hrlist)):
        #    print(hrlist[i])
            if hrlist[i] not in dict2:
                dict2[hrlist[i]] = longitude[i]
            else:
                dict2[hrlist[i]].append(longitude[i])
        lat_list = []
        long_list = []
        for i in dict1.keys():
            temp = dict1[i]
            temp = np.mean(removNestings(temp))
            dict1[i] = temp
            lat_list.append(temp)
            
        for i in dict2.keys():
            temp = dict2[i]
            temp = np.mean(removNestings(temp))
            dict2[i] = temp
            long_list.append(temp)
        
        
        x = list(dict1.keys())  
        y = lat_list
        z = long_list
        
        # y_filter = preprocessor(y)
        # z_filter = preprocessor(z)
        # s_filter = preprocessor(s)
        
        #plt.xlabel('Longitude',fontsize= 25)
        #plt.ylabel('Latitude',fontsize= 25)
        #plt.plot(y,z,linewidth=8,label='Turn')
        #plt.legend(prop={'size': 20})
        #plt.show()
        
        
        
        cnt=0
        turn_list = []
        
        for i in range(1,len(y)-1):
            phi_prev= y[i-1]
            lamb_prev= z[i-1]
            phi_cur = y[i]
            lamb_cur = z[i]
            phi_next= y[i+1]
            lamb_next= z[i+1]
        
            del_phi1 = phi_cur - phi_prev
            del_lamb1 = lamb_cur- lamb_prev
        
            del_phi2 = phi_next - phi_cur
            del_lamb2 = lamb_next- lamb_cur
            sin = 0.707
            cos = 0.707
            res = "No turn"
            val_res = 0
            # cnt=0
            if del_phi1!=0 and del_lamb1!=0 and del_phi2!=0 and del_lamb2!=0:
                cnt = cnt + 1
                m3 = del_phi1/del_lamb1
                m4 = del_phi2/del_lamb2
                theta2 = abs((m3-m4)/(1+m3*m4))
                angle2 = math.degrees(math.atan(theta2))
#                print('\nAngle> ',i, '>', angle2)
                turn_list.append(angle2)
                
       #fp = open('C:\\Users\\91987\\Desktop\\results_turnangles\\sanmateo\\'+ temp + '.csv', 'a')
#        fp = open('D:\\dataset\\AndroSensor\\turn_20220718_150621.csv', 'a')
#        fp.write(str(x[i]) + ',' + convert(x[i]) + ',' + str(angle2) + '\n')
        #fp.close()
#    else:
        #fp = open('C:\\Users\\91987\\Desktop\\results_turnangles\\uah\\d1\\'+ temp + '.csv', 'a')
#        fp = open('D:\\dataset\\AndroSensor\\turn_20220718_150621.csv', 'a')
#
#        fp.write(str(x[i]) + ',' + convert(x[i]) + ',' + str(0) + '\n')
        #fp.close()
#fp.close()
        return turn_list

###################### fetching turn at 5 sec ####################################################
#str2 = "D:\dataset\AndroSensor\\turn_20220716_131931.csv"
#    # print(str1, '\n')
#df = pd.read_csv(str2, header = None)
#time1 = df[0].tolist()
#rating = df[2].tolist()
#length = len(time1)
#count = 1
#ratings = 0
##print(length)
#start = time1.index(47520)
#end = time1.index(47939)
#time1 = time1[start:end+1]
#rating = rating[start:end+1]
#length = len(time1)
#sum1 = 0
#count = 0
#for i in range(length):
#    count = count + 1
#    sum1 = sum1 + rating[i]
#    if(count == 5):
#        sum1 = sum1/5
#        print(sum1)
#        sum1 = 0
#        count = 0
#if(length == 61):
#    for i in range(0,61):
#        ratings = ratings + rating[i] 
#        if(i==4 or i==9 or i==14 or i==19 or i==24 or i==29 or i==34 or i==39 or i==44
#            or i==49 or i==54 or i==60):
#            
#            if(i==60):
#                ratings = ratings / 6
#            else:
#                ratings = ratings / 5
#                
#            print(ratings)
#            
#            # if(ratings>= 0.5):
#            #     print( 1)
#            # else:
#            #     print( 0)
#                
#            ratings = 0      
#elif(length == 60):
#    for i in range(0,60):
#        ratings = ratings + rating[i] 
#        if(i==4 or i==9 or i==14 or i==19 or i==24 or i==29 or i==34 or i==39 or i==44
#            or i==49 or i==54 or i==59):
#            
#            if(i==59):
#                ratings = ratings / 5
#            else:
#                ratings = ratings / 5
#                
#            print(ratings)
#            
#            # if(ratings>= 0.5):
#            #     print( 1)
#            # else:
#            #     print( 0)
#                
#            ratings = 0        

##############################################
################################# processing nexar IMU data ##################
def con_to_csv():
        files = glob.iglob(os.path.join('/home/pi/Downloads/imu/', "./*A.dat")) ##copying the imu files only
        for file in files:
            if os.path.isfile(file):
                shutil.copy2(file, '/home/pi/Downloads/imu/imu_proc/')
                
        
        os.getcwd()
        os.chdir('/home/pi/Downloads/imu/imu_proc/')
        path = os.getcwd()
#        arr = os.listdir()
        for file in glob.glob("*A.dat"):
#        for file in arr:
            fp = open(file, 'r')
            name = file[:-4]
            fw = open(path+"/" + name+".csv", 'a')
            fw.write("time"+";"+"accely"+";"+"accelz"+";"+"accelx"+";"+"gyroy"+";"+"gyroz"+";"+"gyrox"+"\n")
            next(fp)
            for line in fp:
                time = line.split('|')
                timestamp = time[0]
                fw.write(str(timestamp)+";")
                imu = time[1]
                imu = imu.split(":")
                imu = imu[1]
                imu = imu.split(",")
                fw.write(str(imu[0])+";"+str(imu[1])+";"+str(imu[2])+";"+str(imu[4])+";"
                         +str(imu[5])+";"+str(imu[6])+"\n")
            fw.close()
            fp.close()        
                
        
def main():
#file_name = open("C:\\Users\\91987\\Desktop\\filenames.txt",'r') 1d3fad59-b4117158 0a715be4-0064c477
# 000d35d3-41990aa4
# 000f8d37-d4c09a0f
# 09cdd188 8f45bedc
#count_line = 1
#path = 'C:\\Users\\91987\\Desktop\\json_info\\accel-x-y-z\\' + '000f8d37-d4c09a0f' + '.txt' 
#plot_traj(path)
    
#    os.chdir('/home/user/Desktop/imu/')
#    con_to_csv()
#    filepath = '/home/pi/Downloads/imu/imu_proc/'
#    name = sys.argv[1]
    filename = sys.argv[1]
#    os.chdir('/home/pi/Downloads/')
#    filepath = '/home/pi/Downloads/'+name+".csv"
    filepath = '/home/pi/Downloads/'+filename+".csv"
    
    df = pd.read_csv(filepath, sep=",", header=None)
    df.columns = ['timestamp', 'accx', 'accy', 'accz']
#    for file in glob.glob("*A.csv"):
#        print(file)
#        filepath = filepath
#        filename = file
    df1 = pd.DataFrame(columns=['jerkiness', 'stopy', 'stopz'])
    ls1 = call_jerkiness(filepath, filename)
    ls2 = call_stopper(filepath, filename)[0]
    ls3 = call_stopper(filepath, filename)[1]
    print(len(ls1), len(ls2), len(ls3))
    min_i = np.min([len(ls1), len(ls2), len(ls3)])
    df1['jerkiness'] = ls1[:min_i]  
    df1['stopy'] = ls2[:min_i]
    df1['stopz'] = ls3[:min_i]
#        filepath = '/home/user/Desktop/imu/'
#        filename = 'Sensor_record_20220718_150621_AndroSensor.csv'
#        detect_turn(filepath+filename, filename)
#        df['turn'] = detect_turn(filepath+filename, filename)[:60]

    df1.to_csv('/home/pi/Downloads/'+"man"+filename+".csv")
#sec_to_date_time(1503828254943)
#sec_to_date_time(1503828295206)
if __name__=='__main__':
    main()       
