import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

# Local util files
import config
sys.path.insert(0, "src")
import transform
import utils


def style_image(image_in, save_path, saved_model, device_t="/cpu:0", batch_size=1, save=True):
    """Apply model style to an image."""
    img_shape = image_in.shape

    g = tf.Graph()
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size, ) + img_shape
        img_placeholder = tf.compat.v1.placeholder(
            tf.float32, shape=batch_shape, name="img_placeholder"
        )

        preds = transform.net(img_placeholder)
        
        # Load pretrained model
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(saved_model):
            ckpt = tf.train.get_checkpoint_state(saved_model)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found.")
        else:
            saver.restore(sess, saved_model)

        # Apply new style (batch is only one image)
        X = np.zeros(batch_shape, dtype=np.float32)
        X[0] = image_in
        result_batch = sess.run(preds, feed_dict={img_placeholder: X})

        if save:
            utils.save_img(save_path, result_batch[0])
        else:
            return result_batch[0]


def style_video(video_in, save_path, saved_model, device_t="/cpu:0", batch_size=4):
    """Apply model style to a video."""
    video_clip = VideoFileClip(video_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(
        save_path,
        video_clip.size,
        video_clip.fps,
        codec="libx264",
        preset="medium",
        bitrate="2000k",
        audiofile=None,
        threads=None,
        ffmpeg_params=None,
    )

    g = tf.Graph()
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), tf.compat.v1.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.compat.v1.placeholder(
            tf.float32, shape=batch_shape, name="img_placeholder"
        )
        preds = transform.net(img_placeholder)

        # Load pretrained model
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(saved_model):
            ckpt = tf.train.get_checkpoint_state(saved_model)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, saved_model)

        X = np.zeros(batch_shape, dtype=np.float32)

        def style_and_write(count):
            """Copy the model style and write frames to file."""
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                video_writer.write_frame(np.clip(_preds[i], 0, 255).astype(np.uint8))

        frame_count = 0  # The frame count that written to X
        for frame in video_clip.iter_frames():
            X[frame_count] = frame
            frame_count += 1
            if frame_count == batch_size:
                style_and_write(frame_count)
                frame_count = 0

        if frame_count != 0:
            style_and_write(frame_count)

        video_writer.close()


def style_webcam(saved_model, device_t="/cpu:0", batch_size=1):
    """Apply model style to a video stream."""
    # Setup
    camera = cv2.VideoCapture(config.CAMERA_DEVICE)
    g = tf.Graph()
    soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), g.device(device_t), tf.compat.v1.Session(config=soft_config) as sess:
        # Build TF variables
        _, frame = camera.read()
        batch_shape = (batch_size,) + frame.shape
        img_placeholder = tf.compat.v1.placeholder(
            tf.float32, shape=batch_shape, name="img_placeholder"
        )
        preds = transform.net(img_placeholder)

        # Load pretrained model
        saver = tf.compat.v1.train.Saver()
        if os.path.isdir(saved_model):
            ckpt = tf.train.get_checkpoint_state(saved_model)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found.")
        else:
            saver.restore(sess, saved_model)

        # Apply new style
        while True:
            success, frame = camera.read()

            if not success:
                break
            else:
                # Apply style transformation
                X = np.zeros(batch_shape, dtype=np.float32)
                X[0] = frame
                result_batch = sess.run(preds, feed_dict={img_placeholder: X})
                frame_out = result_batch[0]

                # Convert image into buffer of bytes for streaming
                ret, buffer = cv2.imencode('.jpg', frame_out)
                frame = buffer.tobytes()

                # concat frame one by one and show result
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
