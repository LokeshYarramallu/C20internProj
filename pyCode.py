# if you not have a downloaded model, it automatically downloads the first time you run the code
# Define variables

voice_id = 0                        # index of the desired voice
voice_rate = 100                    # rate of the speech
voice_vol = 5                       # volume of the speech
WebCam_id = 1                       # index of the webcam device
Frame_interval = 8                 # interval for displaying the frames
requiredClss = ["chair","couch","bed","toilet","sink"] 
                                    # list of required classes to detect

# Import required libraries
from ultralytics import YOLO        # YOLO object detection model
import cv2                          # OpenCV for image and video processing
import math                         # for math.ceil() function
import pyttsx3                      # text-to-speech library
import time                         # for frame rate control


# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate',voice_rate)
engine.setProperty('volume',voice_vol)
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[voice_id].id)

# Define a function to draw a rectangle with corners and text
def cornerRect(img, x,y,x1,y1,text,textScale=1, textThickness=2, l=30, t=3, rt=1, colorR=(100, 0, 100), colorC=(50, 100, 200), colorT=(10,10,10),font=cv2.FONT_HERSHEY_SIMPLEX):
    if rt != 0 : cv2.rectangle(img, (x,y),(x1,y1), colorR, rt)
    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv2.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + l, y1), colorC, t)
    cv2.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    cv2.rectangle(img, (x1, y1), (x1+15, y1+15), colorR, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, textScale, colorT, textThickness)

    return img

# Initialize the webcam
cap = cv2.VideoCapture(WebCam_id)

# Load the YOLO model
model = YOLO("../models/yolov8m.pt")
prevTime = time.time()

while True :
    #reading the frames
    ret, frame = cap.read()
    detected=[]

    results = model(frame, stream=True)
    for res in results :
        boxes = res.boxes
        for box in boxes :
            if (model.names[int(box.cls[0])]) in requiredClss :
                detected.append(str(model.names[int(box.cls[0])]))

                x1 , y1 , x2 , y2 = box.xyxy[0]                                             # bounding box coordinates
                x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)

                Conf = math.ceil(box.conf[0] * 100)/100                                     # confidence
                frame = cornerRect(frame,x1,y1,x2,y2,str(model.names[int(box.cls[0])]))     # adding detections rectangle anf names of class
                response =  str(model.names[int(box.cls[0])]) + "is found"                  # forming a string for voice command

    cv2.imshow("Cam",frame)                                         # frame visualization
    if time.time() - prevTime >= Frame_interval:
        prevTime = time.time()
        opt = ""
        if len(detected) != 0:
            for thing in detected : opt = opt + "," + str(thing)
            opt += "found in the frame."
            engine.say(opt)                                         # Creating a Voice response
            engine.runAndWait()
    if cv2.waitKey(1) == ord('q'):break                             # loop break


cap.release()
cv2.destroyAllWindows()             # ending all created windows


