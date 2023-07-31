from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2 as cv
from calibration import Calibration
from iris_tracker import IrisTracker
from gaze_tracker import DirectionTracker

app = Flask(__name__)
socketio = SocketIO(app)


@socketio.on('video_feed')
def video_feed(message):
    cap = cv.VideoCapture(0)
    tracker = DirectionTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gaze_direction = tracker.get_gaze_direction(frame)

        emit('gaze_direction', {'gaze_direction': gaze_direction})

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
