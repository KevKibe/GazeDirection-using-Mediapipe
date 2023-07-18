from gazetracker import GazeDirection
import cv2

gaze = GazeDirection(cap)

# Webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform gaze tracking
    pupil_left, pupil_right = gaze.refresh(frame)

    # Check gaze direction
    if gaze.is_top_right(pupil_left, pupil_right):
        text = "Looking top right"
    elif gaze.is_top_left(pupil_left, pupil_right):
        text = "Looking top left"
    elif gaze.is_bottom_right(pupil_left, pupil_right):
        text = "Looking bottom right"
    elif gaze.is_bottom_left(pupil_left, pupil_right):
        text = "Looking bottom left"   
    elif gaze.is_top(pupil_left, pupil_right):
        text = "Looking top"
    elif gaze.is_bottom(pupil_left, pupil_right):
        text = "Looking bottom"
    elif gaze.is_left(pupil_left, pupil_right):
        text = "Looking left"
    elif gaze.is_right(pupil_left, pupil_right):
        text = "Looking right"
    else:
        text = "Looking center"

    # Get annotated frame with highlighted pupils
    annotated_frame = gaze.annotated_frame()
    cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("Gaze Tracking", annotated_frame)

    # Exit the program by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
