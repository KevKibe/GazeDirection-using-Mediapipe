import cv2
import mediapipe as mp
import numpy as np

class Calibration:
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """Returns the threshold value for the given eye.

        Argument:
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            if len(self.thresholds_right) == 0:
                return 0
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

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

    # def horizontal_ratio(self, left_eye_coords, right_eye_coords, image_undistorted):
    #     left_start_x, left_start_y = left_eye_coords['start']
    #     left_end_x, left_end_y = left_eye_coords['end']

    #     right_start_x, right_start_y = right_eye_coords['start']
    #     right_end_x, right_end_y = right_eye_coords['end']

    #     left_center_x = (left_start_x + left_end_x) // 2
    #     right_center_x = (right_start_x + right_end_x) // 2

    #     image_width = image_undistorted.shape[1]

    #     left_ratio = left_center_x / image_width
    #     right_ratio = right_center_x / image_width

    #     return (left_ratio + right_ratio) / 2    

    # def vertical_ratio(self, left_eye_coords, right_eye_coords, image_undistorted):
    #     left_start_x, left_start_y = left_eye_coords['start']
    #     left_end_x, left_end_y = left_eye_coords['end']

    #     right_start_x, right_start_y = right_eye_coords['start']
    #     right_end_x, right_end_y = right_eye_coords['end']

    #     left_center_y = (left_start_y + left_end_y) // 2
    #     right_center_y = (right_start_y + right_end_y) // 2

    #     image_height = image_undistorted.shape[0]

    #     left_ratio = left_center_y / image_height
    #     right_ratio = right_center_y / image_height

    #     return (left_ratio + right_ratio) / 2

    # def determine_gaze_direction(self, horizontal_ratio, vertical_ratio):
    #     # ... (same as before)

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
                    print("Left Eye Coordinates:", eye_coordinates['eye_1'])

                    # Print the right eye coordinates
                    if 'eye_2' in eye_coordinates:
                        right_eye_coords = eye_coordinates['eye_2']
                        print("Right Eye Coordinates:", right_eye_coords)

                    if self.calibration.is_complete():
                        horizontal_ratio = self.horizontal_ratio(eye_coordinates['eye_1'], right_eye_coords, image_undistorted)
                        vertical_ratio = self.vertical_ratio(eye_coordinates['eye_1'], right_eye_coords, image_undistorted)
                        direction = self.determine_gaze_direction(horizontal_ratio, vertical_ratio)
                        print("Gaze Direction:", direction)
                    else:
                        threshold = self.calibration.threshold(0 if face_landmarks.landmark[33].visibility > 0.5 else 1)
                        cv2.putText(image_undistorted, f"Threshold: {threshold}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Calibrated Webcam', image_undistorted)

            if cv2.waitKey(5) & 0xFF == ord('q'):
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
