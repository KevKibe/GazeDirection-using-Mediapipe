import cv2 as cv
import numpy as np

class Calibration:
    def __init__(self):
        self.mtx = None
        self.dist = None

    def calibrate_camera(self, frame):
        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
        # Detect the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    
        # If the corners are found, calibrate the camera
        if ret:
            # Define the object points and image points
            objp = np.zeros((6 * 7, 3), np.float32)
            objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
            objpoints = [objp]
            imgpoints = [corners]
        
            # Calibrate the camera
            ret, self.mtx, self.dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def undistort_frame(self, frame):
        if self.mtx is not None and self.dist is not None:
            # Undistort the frame
            frame = cv.undistort(frame, self.mtx, self.dist, None, self.mtx)
        return frame
