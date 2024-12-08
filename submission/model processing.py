import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):
        robust_scaler = RobustScaler()
        scaled_data = robust_scaler.fit_transform(np.transpose(X))
        minmax_scaler = MinMaxScaler()
        scaled_data = minmax_scaler.fit_transform(scaled_data)
        X = np.transpose(scaled_data)
        

        reg_predictions = np.array([])
        telescope = 9 #then 18
        autoregressive_telescope = 1 #or 1 or 2
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

        inverse = minmax_scaler.inverse_transform(np.transpose(out))
        inverse = robust_scaler.inverse_transform(inverse)
        out = np.transpose(inverse)

        return out
