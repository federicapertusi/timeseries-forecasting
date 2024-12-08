import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):
        
        # Note: this is just an example.
        # Here the model.predict is called
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        out = self.model.predict(X)  # Shape [BSx9] for Phase 1 and [BSx18] for Phase 2
        out = np.reshape(out, (out.shape[0], out.shape[1]))

        return out
