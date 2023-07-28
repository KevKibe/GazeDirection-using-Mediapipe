import cv2 as cv
import numpy as np
import mediapipe as mp


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
        
        return center, radius/2

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
