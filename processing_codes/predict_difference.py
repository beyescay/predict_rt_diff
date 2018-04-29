from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, save_npz, load_npz
import pickle
import sklearn.metrics as M
import time as T
import numpy as np
import glob
import os

class PredictDifference:
    def __init__(self):

        start_0 = T.clock()

        start = T.clock()
        print("Loading the model...")
        self.models = []
        self.model_paths = glob.glob(os.path.join(os.getcwd(), "models", "*.sav"))
        for model in self.model_paths:
            self.models.append(pickle.load(open(model, 'rb')))
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        start = T.clock()
        print("Loading the npz files of training data set...")
        self.x_test_features_matrix = load_npz("./test_data/x_test.npz")
        self.y_test_column_matrix = load_npz("./test_data/y_test.npz")
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        start = T.clock()
        print("Converting the test features matrix into sparse matrix...")
        self.x_test_features_matrix = csr_matrix(self.x_test_features_matrix.todense(), dtype=np.float64)
        self.y_test_column_matrix = self.y_test_column_matrix.todense()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        start = T.clock()
        print("Predicting the test data set error...\n")
        self.predict_test_data()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Process done.\nTotal Time taken: {}\n\n".format(T.clock()-start_0))

    def predict_test_data(self):

        for idx, model in enumerate(self.models):
            y_test_predicted_column_matrix = model.predict(self.x_test_features_matrix)

            print("Test mean absolute error for model {}: {}".format(os.path.basename(self.model_paths[idx]), M.mean_absolute_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))
            print("Test median absolute error for model {}: {}".format(os.path.basename(self.model_paths[idx]), M.median_absolute_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))
            print("Test mean squared error for model {}: {}\n".format(os.path.basename(self.model_paths[idx]), M.mean_squared_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))

        """
        for i in range(len(y_test_predicted_column_matrix)):
            print("\n\nRow: {}".format(i+1))
            print("Prediction: {}".format(y_test_predicted_column_matrix[i]))
            print("Actual:{}".format(self.y_test_column_matrix[i]))
        """