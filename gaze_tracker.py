import cv2 as cv
import numpy as np
import mediapipe as mp
from calibration import CameraCalibration
from iris_tracker import IrisTracker


class DirectionTracker:
    def __init__(self, cap):
        self.iris_tracker = IrisTracker()
        self.calibrator = CameraCalibration((24, 17), (25, 18))
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
        if get_horizontal_ratio <= 0.49:
            gaze_direction = 'left'
        elif get_horizontal_ratio >= 0.53: 
            gaze_direction = 'right'
        else:
            gaze_direction = 'center'

        # if get_vertical_ratio <= 0.50:
        #     gaze_direction = 'up'
        # elif get_vertical_ratio >= 0.51:
        #     gaze_direction = 'down'
        # else:
        #     gaze_direction = 'center'
        return gaze_direction

    def run(self):
        ret, frame = self.cap.read()
        if not ret:
          print("Ignoring empty camera frame.")
          return
        print("Frame shape:", frame.shape)
        self.calibrator.calibrate_camera() 

        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break

            frame = self.calibrator.calibrate_frame(frame) 
            

            left_eye_center, right_eye_center, frame = self.iris_tracker.run(frame)
        
            if left_eye_center is not None and right_eye_center is not None:
                horizontal_ratio = self.get_horizontal_ratio(left_eye_center, right_eye_center)
                vertical_ratio = self.get_vertical_ratio(left_eye_center, right_eye_center)
                # print(horizontal_ratio, vertical_ratio)
                gaze_direction = self.calculate_gaze_direction(horizontal_ratio, vertical_ratio)
            else:
                gaze_direction = 'unknown'
         
            # print(f"Gaze direction: {gaze_direction}")

            cv.putText(frame, f"Gaze Direction: {left_eye_center, right_eye_center}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Webcam', frame)

            if cv.waitKey(1) & 0xFF == 27:
                break

        
             
             
  


if __name__ == '__main__':


    cap = cv.VideoCapture(0)
    direction_tracker = DirectionTracker(cap)
    direction_tracker.run()
    direction_tracker.cap.release()
    cv.destroyAllWindows()




