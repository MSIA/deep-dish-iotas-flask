import glob
import os
import sys

import numpy as np
import tensorflow as tf


class PersistentModel(Object):

    def __init__(self, saved_model):
        self.session = tf.Session()

        # Load pretrained model
        tf.compat.v1.saved_model.loader.load(
            self.session,
            [tf.saved_model.tag_constants.SERVING],
            saved_model
        )

        self.predict_tensor = self.session.graph.get_tensor_by_name("final_ops/softmax:0")

    def predict(self, images):
        predictions = self.session.run(self.predict, feed_dict={"Placeholder:0": images})
        return predictions
