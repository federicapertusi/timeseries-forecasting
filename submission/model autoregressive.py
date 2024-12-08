import os
import tensorflow as tf
import numpy as np

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):
        
        # Note: this is just an example.
        # Here the model.predict is called
        reg_predictions = np.array([])
        telescope = 18
        autoregressive_telescope = 9
        X_temp = np.reshape(X, (X.shape[0], X.shape[1], 1))
        for reg in range(0,telescope,autoregressive_telescope):
            pred_temp = self.model.predict(X_temp)
            pred_temp= np.reshape(pred_temp, (pred_temp.shape[0],pred_temp.shape[1],1))
            if(len(reg_predictions)==0):
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)
            X_temp = np.concatenate((X_temp[:,autoregressive_telescope:],pred_temp), axis=1)


        out = reg_predictions  # Shape [BSx9] for Phase 1 and [BSx18] for Phase 2
        out = np.reshape(out, (out.shape[0], out.shape[1]))

        return out
