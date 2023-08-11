import cv2
import numpy as np
import glob 

class CameraCalibration:
    def __init__(self, pattern_size, checkerboard_size):
        self.pattern_size = pattern_size
        self.checkerboard_size = checkerboard_size
        self.file_names = glob.glob("../calibration/Image*.png")
        self.q = []
        self.Q = []
        self.K = np.eye(3, dtype=np.float32)
        self.k = np.zeros(5, dtype=np.float32)
        self.mapX = None
        self.mapY = None

    def find_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pattern_found, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )
        return pattern_found, corners

    def calibrate_camera(self):
        rvecs = []
        tvecs = []
        flags = (
            cv2.CALIB_FIX_ASPECT_RATIO
            + cv2.CALIB_FIX_K3
            + cv2.CALIB_ZERO_TANGENT_DIST
            + cv2.CALIB_FIX_PRINCIPAL_POINT
        )
        frame_size = (480, 640)

        error, self.K, self.k, rvecs, tvecs = cv2.calibrateCamera(
            self.Q, self.q, frame_size, self.K, self.k, rvecs, tvecs, flags=flags
        )

    def compute_undistortion_maps(self):
        self.mapX, self.mapY = cv2.initUndistortRectifyMap(
            self.K, self.k, np.eye(3, dtype=np.float32), self.K, (480, 640), cv2.CV_32FC1
        )

    def undistort_frame(self, frame):
        if self.mapX is not None and self.mapY is not None:
            return cv2.remap(
                frame,
                self.mapX,
                self.mapY,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        return frame

    def calibrate_frame(self, frame):
        self.calibrate_camera()
        self.compute_undistortion_maps()
        calibrated_frame = self.undistort_frame(frame)
        return calibrated_frame

    def generate_world_coordinates(self):
        objp = np.zeros(
            ((self.checkerboard_size[0] - 1) * (self.checkerboard_size[1] - 1), 3)
        )
        objp[:, :2] = np.mgrid[
            1 : self.checkerboard_size[0], 1 : self.checkerboard_size[1]
        ].T.reshape(-1, 2)
        return objp
