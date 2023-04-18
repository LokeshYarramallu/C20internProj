# Voices can be change using voice id
# Voice speed and volume can be change using SetProperty Commands
# Time loop can be changed when for the time intervals for readin frames
# Confidence and Overlap Can be changed in Model Prediction

voice_id = 0
voice_rate = 100 
voice_vol = 5

import cv2                             # For accessing Webcam
import time
from roboflow import Roboflow          
import pyttsx3                         # For audio Commands
import pickle                          # For extracting the model file saved in binary


# Importing the PreTrained locally saved Model
# Model is trained using RoboFlow and saved in the Local Memory
with open("model","rb") as file :
    model = pickle.load(file)

# Loading a variable using pytextToSpeechX3 library for Voice output
engine = pyttsx3.init()
engine.setProperty('rate',voice_rate)
engine.setProperty('volume',5)
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[voice_id].id) 

# Making a Video Capture using OpenCV
cap = cv2.VideoCapture(1)

# For Making Time Loop 
prev_time = time.time()


while True:
    # Extracting the frames from WebCam
    ret, frame = cap.read()

    # Creating to read the frames for every nth second
    if time.time() - prev_time >= 8:
        prev_time = time.time()
        detect = frame.copy()
        detections = model.predict(frame, confidence=40, overlap=30).json()

        for detection in detections["predictions"]:
            if detection["class"]!='1' and '2' :
                Class = detection["class"]
                x1 = detection['x'] - detection['width'] / 2
                x2 = detection['x'] + detection['width'] / 2
                y1 = detection['y'] - detection['height'] / 2
                y2 = detection['y'] + detection['height'] / 2

                # Creating a bounding box in the copy frame
                cv2.rectangle(detect,(int(x1),int(y1)),(int(x2),int(y2)),(255,125,0),3)
                w,h,c = detect.shape
                response = detection["class"] + "is found"

            cv2.imshow("detected",detect)
            engine.say(response)   # Creating a Voice response
            engine.runAndWait()


    cv2.imshow("webCam", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Closing the Opened Windows
cap.release()
cv2.destroyAllWindows()