import boto3
import os
import glob
from secrets import Access_key , Secret_access_key
import datetime
import time
import traceback
import sys
import picamera
import json
from PIL import Image
import requests
import base64
import cv2
import concurrent.futures

count = 0

url = "https://rav1fwkidb.execute-api.us-east-1.amazonaws.com/face-recognition-api/recognize"
current_path = os.getcwd()
camera = picamera.PiCamera()
camera.resolution = (160,160)
camera.framerate = 15

def record_video(file_name):
    camera.start_recording(file_name)
    camera.wait_recording(0.5)
    camera.stop_recording()
    

def get_prediction_cv(file_name):
    t0 = time.time()
    video = cv2.VideoCapture(file_name)
    success, image = video.read()
    count=0
    base_name = file_name.split("/")[-1].split(".")[0]
    img_path = base_name + ".png"
    if success:        
        cv2.imwrite(current_path+"/images/"+img_path, image)
    
    f1 = current_path+"/images/"+img_path
    
    with open(f1 , "rb") as img:
        encode_img = base64.b64encode(img.read())
        d = {"image":encode_img.decode("utf-8")}
        json_obj = json.dumps(d)
        #print(file_name, "Post_request_sent" , start_time)
        x = requests.post(url , data = json_obj)
        latency = time.time()-t0
        
        data_json = json.loads(x.text)
        person = base_name.split("-")[1]
        name = data_json["name"]
        course = data_json["major"]
        year = data_json["year"]
        print("The "+person+" person recognized: ," +name+ " ," + course + " ," + year)
        print("Latency: {:2f} seconds.".format(latency))
    
    os.remove(f1)


def upload_to_s3(file_name):
    time.sleep(3)
    s3_name = file_name.split("/")[-1]
    s3 = boto3.client('s3' , region_name = "us-east-1", aws_access_key_id = Access_key , aws_secret_access_key = Secret_access_key)
    s3.upload_file(file_name ,"cc-project-video-2",s3_name)
    os.remove(file_name)
        
    
process_count = 0
if __name__ == "__main__":
    i = time.time()+ 300
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while(int(time.time()) <= i):    
            file_name ="/home/pi/Desktop/cc_project/videos/v-"+str(process_count)+".h264"
            record_video(file_name)
            f = executor.submit( get_prediction_cv, file_name)
            f2 = executor.submit(upload_to_s3 , file_name)
            process_count +=1
        camera.close()
        executor.shutdown(wait=True)
     
    print(process_count)


