import cv2 as cv
import numpy as np
import mediapipe as mp
from calibration import Calibration
from iris_tracker import IrisTracker


class DirectionTracker:
    def __init__(self, cap):
        self.iris_tracker = IrisTracker()
        self.calibration = Calibration()
        self.cap = cap

    def get_vertical_ratio(self,left_eye_coords,right_eye_coords):
        left_vertical_ratio = left_eye_coords[1] / (left_eye_coords[0] * 2 - 10)
        right_vertical_ratio = right_eye_coords[1] / (right_eye_coords[0] * 2 - 10)
        return (left_vertical_ratio + right_vertical_ratio)/2

    def get_horizontal_ratio(self,left_eye_coords,right_eye_coords):
        left_horizontal_ratio = left_eye_coords[0] / (left_eye_coords[1] * 2 - 10)
        right_horizontal_ratio = right_eye_coords[0] / (right_eye_coords[1] * 2 - 10)
        return (left_horizontal_ratio + right_horizontal_ratio)/2
    

        
    def calculate_gaze_direction(self,get_horizontal_ratio,get_vertical_ratio):
        if get_horizontal_ratio <= 0.60:
            gaze_direction = 'left'
        elif get_horizontal_ratio >= 0.62:  #mid = 0.54  left=<0.50 right 0.5
            gaze_direction = 'right'#vert cent = 0.46
        else:
            gaze_direction = 'center'

        # if get_vertical_ratio >= 0.65:
        #     gaze_direction = 'up'
        # elif get_vertical_ratio <= 0.35:
        #     gaze_direction = 'down'
        # else:
        #     gaze_direction = 'center'
        return gaze_direction

    def run(self):
        # Calibrate the camera
        ret, frame = self.cap.read()
        if not ret:
          print("Ignoring empty camera frame.")
          return
        self.calibration.calibrate_camera(frame)
    
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break

            # Undistort the frame
            frame = self.calibration.undistort_frame(frame)

            left_eye_center, right_eye_center, frame = self.iris_tracker.run(frame)
        
            if left_eye_center is not None and right_eye_center is not None:
                horizontal_ratio = self.get_horizontal_ratio(left_eye_center, right_eye_center)
                vertical_ratio = self.get_vertical_ratio(left_eye_center, right_eye_center)
                print(horizontal_ratio, vertical_ratio)
                gaze_direction = self.calculate_gaze_direction(horizontal_ratio, vertical_ratio)
            else:
                gaze_direction = 'unknown'
         
            # print(f"Gaze direction: {gaze_direction}")

            cv.putText(frame, f"Gaze Direction: {gaze_direction}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Webcam', frame)

            if cv.waitKey(1) & 0xFF == 27:
                break

        
             
             
  


if __name__ == '__main__':


    cap = cv.VideoCapture(0)
    direction_tracker = DirectionTracker(cap)
    direction_tracker.run()
    direction_tracker.cap.release()
    cv.destroyAllWindows()




