import cv2
import mediapipe as mp
import numpy as np
from calibration import Calibration


import cv2
import mediapipe as mp

class FaceMesh:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.FACE_CONNECTIONS = [[33, 246], [133, 159]]

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        return results

    def draw_landmarks(self, frame, face_landmarks):
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        
    def extract_eye_coordinates(self, face_landmarks, frame_shape):
        eye_coordinates = {}

        if face_landmarks:
            for i, (start, end) in enumerate(self.FACE_CONNECTIONS):
                start_landmark = face_landmarks.landmark[start]
                end_landmark = face_landmarks.landmark[end]

                image_height, image_width, _ = frame_shape
                start_x, start_y = int(start_landmark.x * image_width), int(start_landmark.y * image_height)
                end_x, end_y = int(end_landmark.x * image_width), int(end_landmark.y * image_height)

                eye_coordinates[f'eye_{i+1}'] = {
                    'start': (start_x, start_y),
                    'end': (end_x, end_y)
                }

        return eye_coordinates
    
class EyeTracker:
    def __init__(self, cap):
        self.face_mesh = FaceMesh()
        self.eye_coordinates = None 
        self.cap = cap

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            image_undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
            image_undistorted = cv2.flip(image_undistorted, 1)

            results = self.face_mesh.process_frame(image_undistorted)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.face_mesh.draw_landmarks(image_undistorted, face_landmarks)

                    self.eye_coordinates = self.face_mesh.extract_eye_coordinates(face_landmarks, image_undistorted.shape)

            cv2.imshow('Calibrated Webcam', image_undistorted)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        
        return self.eye_coordinates
    
# Camera calibration parameters
mtx = np.array([[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]])
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    eye_tracker = EyeTracker(cap)
    eye_tracker.run()