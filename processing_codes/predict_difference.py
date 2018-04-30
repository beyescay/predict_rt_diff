from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, save_npz, load_npz
import pickle
import sklearn.metrics as M
import numpy as np
import glob
import os
import csv
import sys
sys.path.append("../")


class PredictDifference:
    def __init__(self):

        self.output_csv = "../data/test_data/predicted_vs_actual.csv"

        print("Loading the model...")
        self.models = []
        self.stack_models = []
        self.model_paths = sorted(glob.glob(os.path.join(os.getcwd(), "../data/models", "model_*.sav")))

        print("Total models loaded: {}".format(len(self.model_paths)))
        for model in self.model_paths:
            self.models.append(pickle.load(open(model, 'rb')))

        self.stack_model_paths = sorted(glob.glob(os.path.join(os.getcwd(), "../data/models", "stack_model_*.sav")))
        for model in self.stack_model_paths:
            self.stack_models.append(pickle.load(open(model, 'rb')))

        print("Loading the npz files of the data set...")
        self.x_test_features_matrix = load_npz("../data/npz_arrays/X.npz")
        self.y_test_column_matrix = load_npz("../data/npz_arrays/y.npz")

        """
        if self.x_test_features_matrix.shape[0] == 223:
            self.x_test_features_matrix = self.x_test_features_matrix.T
        """

        if self.y_test_column_matrix.shape[0] == 1:
            self.y_test_column_matrix = self.y_test_column_matrix.T

        print("Converting the test features matrix into sparse matrix...")
        self.x_test_features_matrix = csr_matrix(self.x_test_features_matrix.todense(), dtype=np.float64)
        self.y_test_column_matrix = self.y_test_column_matrix.todense()

        print("Predicting the test data set error...\n")
        self.predict_test_data()

    def predict_test_data(self):

        meta_features = np.zeros((self.x_test_features_matrix.shape[0], len(self.model_paths)), dtype=np.float64)

        for idx, model in enumerate(self.models):
            y_test_predicted_column_matrix = model.predict(self.x_test_features_matrix)

            for i in range(len(y_test_predicted_column_matrix)):
                meta_features[i, idx] = y_test_predicted_column_matrix[i]

            print("Test mean absolute error for model {}: {}".format(os.path.basename(self.model_paths[idx]), M.mean_absolute_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))
            print("Test median absolute error for model {}: {}".format(os.path.basename(self.model_paths[idx]), M.median_absolute_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))
            print("Test mean squared error for model {}: {}\n".format(os.path.basename(self.model_paths[idx]), M.mean_squared_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))

        for idx, stack_model in enumerate(self.stack_models):
            final_y_test_predicted_column_matrix = stack_model.predict(meta_features)
            print("Final Test mean absolute error for stacked model {}: {}".format(os.path.basename(self.stack_model_paths[idx]), M.mean_absolute_error(self.y_test_column_matrix, final_y_test_predicted_column_matrix)))
            print("Final Test median absolute error for stacked model {}: {}".format(os.path.basename(self.stack_model_paths[idx]), M.median_absolute_error(self.y_test_column_matrix, final_y_test_predicted_column_matrix)))
            print("Final Test mean squared error for stacked model {}: {}\n".format(os.path.basename(self.stack_model_paths[idx]), M.mean_squared_error(self.y_test_column_matrix, final_y_test_predicted_column_matrix)))

        with open(self.output_csv, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["predicted", "actual"])

            for i in range(len(final_y_test_predicted_column_matrix)):
                actual = float(self.y_test_column_matrix.A[i][0])
                csv_writer.writerow([final_y_test_predicted_column_matrix[i], actual])



