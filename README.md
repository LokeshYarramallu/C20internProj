# C20internProj
Detection of key objects for helping visually disabled people

This project uses Yolov8 model , trained yolov8 and OpenCV to perform real-time object detection using a webcam.

Getting Started , 
To get started with this project, follow these steps:

Clone the repository to your local machine.

Create a Python virtual environment using venv or conda.

Requirements  ,
This project requires the following libraries to be installed:

### --> pip install opencv-python pyttsx3 ultralytics

# How to Use
1.Connect a webcam to your computer.

2.Download the script and the YOLOv8 model.

3.Run the script in a Python environment with the required libraries installed.

4.The script will open a live video stream from your webcam and start detecting the specified classes of objects. When an object is detected, the script will draw a rectangle around it and speak out its name using text-to-speech.

5.Press "q" to exit the program.

Note: The script assumes that the YOLOv8 model file is saved in a directory named "models" in the same directory as the script. If the model file is saved in a different directory, the path in the script should be modified accordingly.
