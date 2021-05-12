import os

import cv2
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    Response,
    send_from_directory,
    url_for
)
from werkzeug.utils import secure_filename

import config
from src.style_transfer import transfer_image, transfer_video, transfer_webcam

# Configuration
app = Flask(__name__)
app.config.from_pyfile('config.py')
camera = cv2.VideoCapture(config.CAMERA_DEVICE)


def _allowed_file(filename):
    """Check if file format is correct."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )


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
        # Parse name of style from dropdown menu
        style = request.form.get('style', 'Grayscale')

        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_filepath)

        # Perform style transfer
        new_filename = 'transferred_' + filename
        save_path = os.path.join(app.config['OUTPUT_FOLDER'], new_filename)

        # Handle videos or images
        if new_filename[-3:] == 'mp4':
            transfer_video(upload_filepath, style, save_path)
            return redirect(url_for('static', filename='transferred/' + new_filename), code=301)
        else:
            transfer_image(file, style, save_path)

        # Show transferred file on /result
        return render_template('result.html', filename=new_filename)

    else:
        flash('Allowed image types are: {}'.format(', '.join(config.ALLOWED_EXTENSIONS)))
        return redirect(request.url)


@app.route('/video_settings')
def video_settings():
    return render_template('video_settings.html')


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='transferred/' + filename), code=301)


@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    # Parse name of style from dropdown menu
    style = request.form.get('style', 'Grayscale')

    return Response(transfer_webcam(style), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)
