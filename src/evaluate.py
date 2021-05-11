import os
import sys

import numpy as np
import tensorflow as tf

# Local util files
sys.path.insert(0, "src")
import transform
import utils


def ffwd(image_in, save_path, saved_model, device_t="/cpu:0", batch_size=1):
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
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, saved_model)

        # Apply new style (batch is only one image)
        X = np.zeros(batch_shape, dtype=np.float32)
        X[0] = image_in
        _preds = sess.run(preds, feed_dict={img_placeholder: X})
        utils.save_img(save_path, _preds[0])
