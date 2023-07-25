import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

class IrisTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv.VideoCapture(0)
        self.RIGHT_IRIS = RIGHT_IRIS
        self.LEFT_IRIS = LEFT_IRIS




    def find_iris_centers(self, landmarks):
        (cx, cy), radius = cv.minEnclosingCircle(landmarks)
        center = np.array([cx, cy], dtype=np.int32)
        return center, radius

    def run(self, frame):
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            left_eye_center, left_eye_radius = self.find_iris_centers(mesh_points[self.LEFT_IRIS])
            right_eye_center, right_eye_radius = self.find_iris_centers(mesh_points[self.RIGHT_IRIS])
            cv.circle(frame, tuple(left_eye_center), int(left_eye_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, tuple(right_eye_center), int(right_eye_radius), (255, 0, 255), 1, cv.LINE_AA)

            # Return the coordinates of the left and right irises
            return left_eye_center, right_eye_center, frame
        else:
            print('No Iris Detected')
            return None, None, frame


class EyeTracker:
    def __init__(self, cap):
        self.iris_tracker = IrisTracker()
        self.cap = cap

    def get_vertical_ratio(self,left_eye_coords,right_eye_coords):
        left_vertical_ratio = left_eye_coords[1] / left_eye_coords[0]
        right_vertical_ratio = right_eye_coords[1] / right_eye_coords[0]
        return (left_vertical_ratio + right_vertical_ratio)/2

    def get_horizontal_ratio(self,left_eye_coords,right_eye_coords):
        left_horizontal_ratio = left_eye_coords[0] / left_eye_coords[1]
        right_horizontal_ratio = right_eye_coords[0] / right_eye_coords[1]
        return (left_horizontal_ratio + right_horizontal_ratio)/2
    
    def is_top(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the top"""
        if pupil_left and pupil_right:
            return self.vertical_ratio(pupil_left, pupil_right) <= 0.35
        
    def calculate_gaze_direction(self,get_horizontal_ratio,get_vertical_ratio):
        if get_horizontal_ratio <= 0.65:
            horizontal_direction = 'left'
        elif get_horizontal_ratio >= 1.35:
            horizontal_direction = 'right'
        else:
            horizontal_direction = 'center'

        if get_vertical_ratio <= 0.65:
            vertical_direction = 'up'
        elif get_vertical_ratio >= 1.35:
            vertical_direction = 'down'
        else:
            vertical_direction = 'center'

        return horizontal_direction, vertical_direction

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame.")

                break
            
            left_eye_center,right_eye_center, frame=self.iris_tracker.run(frame)
            if left_eye_center is not None and right_eye_center is not None:
               horizontal_ratio = self.get_horizontal_ratio(left_eye_center, right_eye_center)
               vertical_ratio = self.get_vertical_ratio(left_eye_center, right_eye_center)
               horizontal_direction, vertical_direction = self.calculate_gaze_direction(horizontal_ratio, vertical_ratio)
            else:
               horizontal_direction = 'unknown'
               vertical_direction = 'unknown'
                
            print(f"Gaze direction: {horizontal_direction}, {vertical_direction}")

            cv.putText(frame, f"Gaze Direction: {horizontal_direction}, {vertical_direction}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Calibrated Webcam', frame)

            if cv.waitKey(1) & 0xFF == 27:
               break
  


if __name__ == '__main__':


    cap = cv.VideoCapture(0)
    eye_tracker = EyeTracker(cap)
    eye_tracker.run()
    eye_tracker.cap.release()
    cv.destroyAllWindows()


# # Camera calibration parameters
# mtx = np.array([[1.0, 0.0, 0.0],
#                 [0.0, 1.0, 0.0],
#                 [0.0, 0.0, 1.0]])
# dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

