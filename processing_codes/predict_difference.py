from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, save_npz, load_npz
import pickle
import sklearn.metrics as M
import time as T
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
        self.model_paths = glob.glob(os.path.join(os.getcwd(), "../data/models", "*.sav"))
        for model in self.model_paths:
            self.models.append(pickle.load(open(model, 'rb')))

        print("Loading the npz files of the data set...")
        self.x_test_features_matrix = load_npz("../data/npz_arrays/X.npz")
        self.y_test_column_matrix = load_npz("../data/npz_arrays/y.npz")

        if self.x_test_features_matrix.shape[0] == 223:
            self.x_test_features_matrix = self.x_test_features_matrix.T

        if self.y_test_column_matrix.shape[0] == 1:
            self.y_test_column_matrix = self.y_test_column_matrix.T

        print("Converting the test features matrix into sparse matrix...")
        self.x_test_features_matrix = csr_matrix(self.x_test_features_matrix.todense(), dtype=np.float64)
        self.y_test_column_matrix = self.y_test_column_matrix.todense()

        print("Predicting the test data set error...\n")
        self.predict_test_data()

    def predict_test_data(self):

        for idx, model in enumerate(self.models):
            y_test_predicted_column_matrix = model.predict(self.x_test_features_matrix)

            print("Test mean absolute error for model {}: {}".format(os.path.basename(self.model_paths[idx]), M.mean_absolute_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))
            print("Test median absolute error for model {}: {}".format(os.path.basename(self.model_paths[idx]), M.median_absolute_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))
            print("Test mean squared error for model {}: {}\n".format(os.path.basename(self.model_paths[idx]), M.mean_squared_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))

        with open(self.output_csv, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["predicted", "actual"])

            print(self.y_test_column_matrix.shape)
            for i in range(len(y_test_predicted_column_matrix)):
                actual = float(self.y_test_column_matrix.A[i][0])
                csv_writer.writerow([y_test_predicted_column_matrix[i], actual])



