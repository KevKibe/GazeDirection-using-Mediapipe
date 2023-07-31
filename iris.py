import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

class IrisTracker:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv.VideoCapture(0)

    def find_iris_centers(self, landmarks):
        (cx, cy), radius = cv.minEnclosingCircle(landmarks)
        center = np.array([cx, cy], dtype=np.int32)
        return center, radius

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                left_eye_center, left_eye_radius = self.find_iris_centers(mesh_points[LEFT_IRIS])
                right_eye_center, right_eye_radius = self.find_iris_centers(mesh_points[RIGHT_IRIS])
                cv.circle(frame, tuple(left_eye_center), int(left_eye_radius), (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, tuple(right_eye_center), int(right_eye_radius), (255, 0, 255), 1, cv.LINE_AA)

                print(left_eye_center, right_eye_center)

            cv.imshow('Iris_coordinate_detection', frame)

            if cv.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    iris_tracker = IrisTracker()
    iris_tracker.run()
