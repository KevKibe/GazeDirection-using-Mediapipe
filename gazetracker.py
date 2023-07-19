import cv2
from eye_tracker import EyeTracker

class GazeDirection:
    def __init__(self):
        cap = cv2.VideoCapture(0)
        self.eye_tracker = EyeTracker(cap)
        
    def refresh(self, frame):
        # Perform gaze tracking using the eye tracker
        eye_coordinates = self.eye_tracker.run()
        pupil_left = eye_coordinates.get('eye_1', {}).get('start', None)
        pupil_right = eye_coordinates.get('eye_2', {}).get('start', None)
        return pupil_left, pupil_right
    
    def horizontal_ratio(self, pupil_left, pupil_right):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5, and the extreme left is 1.0.
        """
        if pupil_left and pupil_right:
            # Calculate horizontal ratio based on pupil coordinates
            return (pupil_left[0] + pupil_right[0]) / 2

    def vertical_ratio(self, pupil_left, pupil_right):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5, and the extreme bottom is 1.0.
        """
        if pupil_left and pupil_right:
            # Calculate vertical ratio based on pupil coordinates
            return (pupil_left[1] + pupil_right[1]) / 2
    
    def is_top(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the top"""
        if pupil_left and pupil_right:
            return self.vertical_ratio(pupil_left, pupil_right) <= 0.35

    def is_bottom(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the bottom"""
        if pupil_left and pupil_right:
            return self.vertical_ratio(pupil_left, pupil_right) >= 0.65

    def is_left(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the left"""
        if pupil_left and pupil_right:
            return self.horizontal_ratio(pupil_left, pupil_right) >= 0.65

    def is_right(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the right"""
        if pupil_left and pupil_right:
            return self.horizontal_ratio(pupil_left, pupil_right) <= 0.35

    def is_top_right(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the top right"""
        if pupil_left and pupil_right:
            return (
                self.horizontal_ratio(pupil_left, pupil_right) <= 0.35 and
                self.vertical_ratio(pupil_left, pupil_right) <= 0.35
            )

    def is_top_left(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the top left"""
        if pupil_left and pupil_right:
            return (
                self.horizontal_ratio(pupil_left, pupil_right) >= 0.65 and
                self.vertical_ratio(pupil_left, pupil_right) <= 0.35
            )

    def is_bottom_right(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the bottom right"""
        if pupil_left and pupil_right:
            return (
                self.horizontal_ratio(pupil_left, pupil_right) <= 0.35 and
                self.vertical_ratio(pupil_left, pupil_right) >= 0.65
            )

    def is_bottom_left(self, pupil_left, pupil_right):
        """Returns true if the user is looking towards the bottom left"""
        if pupil_left and pupil_right:
            return (
                self.horizontal_ratio(pupil_left, pupil_right) >= 0.65 and
                self.vertical_ratio(pupil_left, pupil_right) >= 0.65
            )

    def is_blinking(self, pupil_left, pupil_right):
        """Returns true if the user closes his eyes"""
        if pupil_left and pupil_right:
            # Calculate the average eye opening ratio
            eye_opening_ratio = (pupil_left[1] + pupil_right[1]) / 2
            return eye_opening_ratio > 3.8

    def annotated_frame(self, frame, pupil_left, pupil_right):
        """Returns the main frame with pupils highlighted"""
        if pupil_left and pupil_right:
            annotated_frame = frame.copy()
            color = (0, 255, 0)
            x_left, y_left = pupil_left
            x_right, y_right = pupil_right
            cv2.line(annotated_frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(annotated_frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(annotated_frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(annotated_frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
            return annotated_frame
        else:
            return frame
