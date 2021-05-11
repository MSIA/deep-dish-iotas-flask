import os

from flask import Flask, render_template, request, flash, redirect, url_for, Response
from werkzeug.utils import secure_filename
import cv2

import config
from src.image_transfer import transfer_image, transfer_video_frame

# Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = config.OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.secret_key = config.SECRET_KEY
camera = cv2.VideoCapture(config.CAMERA_DEVICE)


def _allowed_file(filename):
    """Check if file format is correct."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def _gen_frames():
    """Camera live stream."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Apply style transformation
            frame = transfer_video_frame(frame)

            # Convert image into buffer of bytes for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # concat frame one by one and show result
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/image_upload')
def image_upload():
    return render_template('image_upload.html')


@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('image_upload.html')

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and _allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_transferred = transfer_image(file)

        # Save transferred file
        new_filename = 'transferred_' + filename
        file_transferred.save(os.path.join(app.config['OUTPUT_FOLDER'], new_filename))

        # Show transferred file on /result
        return render_template('result.html', filename=new_filename)

    else:
        flash('Allowed image types are: {}'.format(', '.join(config.ALLOWED_EXTENSIONS)))
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='transferred/' + filename), code=301)


@app.route('/video_feed')
def video_feed():
    return Response(_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.PORT)
