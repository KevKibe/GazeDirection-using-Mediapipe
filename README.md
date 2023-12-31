## Description
- This project is a Python program that uses the mediapipe and opencv libraries to track the Iris movement of a subject in real-time webcam video stream and calculate the gaze direction of the user.

## Screenshots
| Left                                       | Right                                      | Center                                     |
| ------------------------------------------ | ------------------------------------------ | ------------------------------------------ |
| ![Image 1](https://github.com/KevKibe/GazeDirection-using-Mediapipe/assets/86055894/9f359af2-7d9e-44b3-b227-fc88d223fcfc) | ![Image 2](https://github.com/KevKibe/GazeDirection-using-Mediapipe/assets/86055894/975590fe-4a2d-49f5-b076-5bd1122c91ec) | ![Image 3](https://github.com/KevKibe/GazeDirection-using-Mediapipe/assets/86055894/a9a062d7-02ff-46c0-bb87-2e142a493e0a) |


## Files in the Repository
- **[iris_tracker.py](https://github.com/KevKibe/GazeDirection-using-Mediapipe/blob/main/iris_tracker.py)** - contains the script to get the right and left iris location using Mediapipe Iris and Computer Vision.
- **[calibration.py](https://github.com/KevKibe/GazeDirection-using-Mediapipe/blob/main/calibration.py)** - contains the script that calibrates and undistorts the frame
- **[gaze_tracker.py](https://github.com/KevKibe/GazeDirection-using-Mediapipe/blob/main/gaze_tracker.py)** - contains the implementation of the iris tracker and calibration and calculation for getting the direction of gaze.
- **[requirements.txt](https://github.com/KevKibe/GazeDirection-using-Mediapipe/blob/main/requirements.txt)** - the libraries and dependencies used in the scripts.

## Installation
- Clone the repository: `git clone https://github.com/KevKibe/GazeDirection-using-Mediapipe.git`
- Install dependencies: `pip install -r requirements.txt`

## Usage
- Run the application by running the command `python gaze_tracker.py` in the terminal.
- To close the application press the Ctrl + C.<br>

## Limitations 
- The application works best with the subject closer to the webcam.
- The applications works best in a well lit environment with a clear webcam camera.
- Horizontal and vertical ratio may be different in each run because the mediapipe model like any other model doesnt produce the same prediction over and over so try playing around with the 'calculate_gaze direction' method to get better results.

**:zap: I'm currently open for roles in Data Science, Machine Learning, NLP and Computer Vision.**
