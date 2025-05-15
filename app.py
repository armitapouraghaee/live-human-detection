from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'yolov5s', source='local')  # Using local repo
model.classes = [0]  # Only detect persons (class 0)

# Initialize camera
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated_frame = results.render()[0]
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
