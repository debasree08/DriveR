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
