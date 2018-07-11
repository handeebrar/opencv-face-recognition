import cv2
import numpy as np

rec = cv2.face.LBPHFaceRecognizer_create()   
rec.read("recognizer\\trainingData.yml")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
#faceDetector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0);
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0, 255, 0)
#id=0
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        conf=100-float(conf)
        print(str(id) + " "  + str(conf))
        if(conf > 40):
            fp = open('users.txt') # Open file on read mode
            lines = fp.read().split("\n") # Create a list containing all lines
            fp.close() # Close file
            user_name_without_parse = lines[id-1]
            user_name_array = user_name_without_parse.split(" ")
            user_name = user_name_array[1]
            #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
            cv2.putText(img,str(user_name), (x,y+h),font, 1,fontcolor)
        else:
            cv2.putText(img,"UNKNOWN!!!", (x,y+h),font, 1,fontcolor)
            
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
    
cam.release()
cv2.destroyAllWindows()
