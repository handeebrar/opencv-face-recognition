import cv2
import os
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
id = raw_input("enter user id")
sampleNumber = 0
open("users.txt", "a")
if (os.stat("users.txt").st_size == 0):
    print("lel")
    user_id = 1
    with open("users.txt", "a") as myfile:
        myfile.write(str(user_id)+" " +str(id))
                    
else:
    with open("users.txt", "r") as myfile:
        lines = myfile.read().splitlines()
        last_line = lines[-1]
        my_list = last_line.split(" ")
        user_id = int(my_list[0])+1
    with open("users.txt", "a") as myfile: 
        myfile.write("\n"+str(user_id)+" " +str(id))
while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces :
        sampleNumber = sampleNumber+1;
        cv2.imwrite("dataSet/"+str(user_id)+"_"+str(id)+"_"+str(sampleNumber)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100);
    cv2.imshow("Face",img);
    cv2.waitKey(1);
    if(sampleNumber>50):
        break;


cam.release()
cv2.destroyAllWindows()





