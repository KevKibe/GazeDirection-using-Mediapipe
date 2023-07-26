import cv2 as cv
import numpy as np

class WebcamCalibration:
    def __init__(self, checkerboard_size=(7, 7)):
        # Define the dimensions of the checkerboard
        self.checkerboard_size = checkerboard_size

        # Define the criteria for termination of the iterative process of cornerSubPix
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Create vectors to store the 3D points and 2D image points
        self.objpoints = []
        self.imgpoints = []

        # Define the 3D points in real world coordinates
        objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

    def calibrate(self):
        # Read the checkerboard images from a file
        images = []
        for i in range(10):
            filename = f'checkerboard{i:02d}.jpg'
            image = cv.imread(filename)
            images.append(image)

        # Find the corners of the checkerboard in the images
        for image in images:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, self.checkerboard_size, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

            # If the corners were found, add the object points and image points
            if ret:
                self.objpoints.append(self.objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                self.imgpoints.append(corners2)

        # Calculate the camera calibration parameters
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, images[0].shape[::-1], None, None)

        # Return the camera matrix and distortion coefficients
        return mtx, dist
