import urllib.request
import cv2
import time
import sys
import math
import smbus  # import SMBus module of I2C
from time import sleep  # import
import datetime as dt
import threading
import multiprocessing

bus = smbus.SMBus(1) 	# or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x69
# some MPU6050 Registers and their Address
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
INT_ENABLE = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47

def cap_vid(name):
    file_name = name+".mp4"
    # Create an object to read
    # from camera
    #file_name = sys.argv[1]
# Create an object to read
# from camera
    video = cv2.VideoCapture(0)
    
    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False):
    	print("Error reading video file")
    
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    
    size = (frame_width, frame_height)
    
    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), 1, size)
    i=int(time.time())+61
    count = 1
    temp = time.time()
    while(time.time()<=i):
    	t1 = time.process_time()
    	ret, frame = video.read()
    	if time.time()-temp-count>=0:
    		count+=1
			
    		if ret == True:
    
    			# Write the frame into the
    			# file 'filename.avi'
    			result.write(frame)
    
    			# Display the frame
    			# saved in the file
    			#cv2.imshow('Frame', frame)
    
    			# Press S on keyboard
    			# to stop the process
    			if cv2.waitKey(1) & 0xFF == ord('s'):
    				break
    
    		# Break the loop
    		else:
    			
    			break
    	#print(time.process_time() - t1)
    	#count = count + 1
    # When everything done, release
    # the video capture and video
    # write objects
    video.release()
    result.release()
    	
    # Closes all the frames
    cv2.destroyAllWindows()
    print(count)
    print("The video was successfully saved")

    
    
    


def MPU_Init(val):
    # write to sample rate register
    # IO Erro
    global Device_Address
    Device_Address = val
    try:
        bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    except OSError as e:
        pass
    # Write to power management register
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)

    # Write to Configuration register
    bus.write_byte_data(Device_Address, CONFIG, 0)

    # Write to Gyro configuration register
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)

    # Write to interrupt enable register
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    # Accelero and Gyro value are 16-bit
    high = bus.read_byte_data(Device_Address, addr)
    low = bus.read_byte_data(Device_Address, addr + 1)

    # concatenate higher and lower value
    value = ((high << 8) | low)

    # to get signed value from mpu6050
    if value > 32768:
        value = value - 65536
    return value

def animate():
    # Add x and y to lists
    ##    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ##	#Read Accelerometer raw value
    acc_x = read_raw_data(ACCEL_XOUT_H)
    acc_y = read_raw_data(ACCEL_YOUT_H)
    acc_z = read_raw_data(ACCEL_ZOUT_H)
    
    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_YOUT_H)
    gyro_z = read_raw_data(GYRO_ZOUT_H)
    
    Ax = acc_x/16384.0
    Ay = acc_y/16384.0
    Az = acc_z/16384.0

    Gx = gyro_x / 131.0
    Gy = gyro_y / 131.0
    Gz = gyro_z / 131.0

    return Ax, Ay, Az, Gx, Gy, Gz
def read_acc(name):
   
    bus = smbus.SMBus(1)  # or bus = smbus.SMBus(0) for older version boards
    Device_Address = 0x68  # MPU6050 device address
    
    MPU_Init(Device_Address)
    
    print(" Reading Data of Gyroscope and Accelerometer")
    cur_time = int(time.time())+60
    #import os
    #os.getcwd()
    
    fp = open("/home/pi/Downloads/"+name+".csv", 'w')
    while (cur_time>=time.time()):
        
        timestamp = int(time.time())
    
        ax, ay, az, gx, gy, gz = animate()
        #print("inside 1")
        #print(ax, ay, az, gx, gy, gz)
        
        fp.write(str(timestamp)+","+str(ax)+","+str(ay)+","+str(az)+"\n")
    #    sleep(0.2)
    fp.close()
def read_gps(name):

	k = int(time.time())+61
	fp = open("/home/pi/Downloads/"+name+"gps.csv", 'w')
	while(time.time()<=k):
		req = urllib.request.Request('https://ipinfo.io/loc')
		with urllib.request.urlopen(req) as response:
		   the_page = response.read()
		the_page = the_page.strip()
		the_page = the_page.decode('UTF-8')
		lat, longi = the_page.split(',')
		print(lat, longi)
		fp.write(lat + "," + longi + "\n")
		
	fp.close()
def func1():
    while 1:
        print("inside 1")
def func2():
    while 1:
        print("inside 2")
def main():
    filename = sys.argv[1]
    Thread1 = threading.Thread(target=cap_vid, args=[filename])
#
##     Create another new thread
    Thread2 = threading.Thread(target=read_acc, args=[filename])

    #Thread3 = threading.Thread(target=read_gps, args=[filename])
#    
#    # Start the thread
    Thread1.start()
#    
#    # Start the thread
    Thread2.start()

    #Thread3.start()
#    
#    # Wait for the threads to finish
    Thread1.join()
    Thread2.join()
    #Thread3.join()
#
#print("Done!")
    #process1 = multiprocessing.Process(target=cap_vid, args=[filename])
    #process2 = multiprocessing.Process(target=read_acc, args=[filename])

    #process1.start()
    #process2.start()
    
    
if __name__=='__main__':
    main()  

    
