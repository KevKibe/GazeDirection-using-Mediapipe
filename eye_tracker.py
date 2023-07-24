import cv2
import mediapipe as mp
import numpy as np
from calibration import Calibration


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
        self.calibration = Calibration()

    def get_left_eye_coordinates(eye_coordinates):
        """Returns the coordinates of the left eye."""
        left_eye_coordinates = eye_coordinates['eye_1']
        return left_eye_coordinates['start'], left_eye_coordinates['end']

    def get_right_eye_coordinates(eye_coordinates):
        """Returns the coordinates of the right eye."""
        right_eye_coordinates = eye_coordinates['eye_2']
        return right_eye_coordinates['start'], right_eye_coordinates['end']

    
    
    def horizontal_ratio(self, left_eye_coords, right_eye_coords, image_undistorted):
        left_start_x, left_start_y = left_eye_coords['start']
        left_end_x, left_end_y = left_eye_coords['end']

        right_start_x, right_start_y = right_eye_coords['start']
        right_end_x, right_end_y = right_eye_coords['end']

        left_center_x = (left_start_x + left_end_x) // 2
        right_center_x = (right_start_x + right_end_x) // 2

        image_width = image_undistorted.shape[1]

        left_ratio = left_center_x / image_width
        right_ratio = right_center_x / image_width

        return (left_ratio + right_ratio) / 2    

    def vertical_ratio(self, left_eye_coords, right_eye_coords, image_undistorted):
        left_start_x, left_start_y = left_eye_coords['start']
        left_end_x, left_end_y = left_eye_coords['end']

        right_start_x, right_start_y = right_eye_coords['start']
        right_end_x, right_end_y = right_eye_coords['end']

        left_center_y = (left_start_y + left_end_y) // 2
        right_center_y = (right_start_y + right_end_y) // 2

        image_height = image_undistorted.shape[0]

        left_ratio = left_center_y / image_height
        right_ratio = right_center_y / image_height

        return (left_ratio + right_ratio) / 2

    def calculate_gaze_direction(self,horizontal_ratio, vertical_ratio,left_eye_coords, right_eye_coords):
        if  horizontal_ratio(left_eye_coords,right_eye_coords) <= 0.65:
            horizontal_direction = 'left'
        elif horizontal_ratio>= 0.35:
            horizontal_direction = 'right'
        else:
            horizontal_direction = 'center'

        if vertical_ratio<= 0.35:
            vertical_direction = 'up'
        elif vertical_ratio >= 0.65:
            vertical_direction = 'down'
        else:
            vertical_direction = 'center'
  
        return horizontal_direction, vertical_direction

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

                    eye_coordinates = self.face_mesh.extract_eye_coordinates(face_landmarks, image_undistorted.shape)
                    left_eye_coords = eye_coordinates['eye_1']
                    print("Left Eye Coordinates:", left_eye_coords)

                    # Print the right eye coordinates
                    if 'eye_2' in eye_coordinates:
                        right_eye_coords = eye_coordinates['eye_2']
                        print("Right Eye Coordinates:", right_eye_coords)

                    horizontal_ratio = self.horizontal_ratio(eye_coordinates['eye_1'], right_eye_coords, image_undistorted)
                    vertical_ratio = self.vertical_ratio(eye_coordinates['eye_1'], right_eye_coords, image_undistorted)
                    direction = self.calculate_gaze_direction(horizontal_ratio, vertical_ratio,left_eye_coords, right_eye_coords)
                    print("Gaze Direction:", direction)
                    # Display the gaze direction on the video feed
                    cv2.putText(image_undistorted, f"Gaze Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            cv2.imshow('Calibrated Webcam', image_undistorted)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Camera calibration parameters
mtx = np.array([[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]])
dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    eye_tracker = EyeTracker(cap)
    eye_tracker.run()
