"""
A simple script that demonstrates how we classify textual data with sklearn.

"""

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

import csv
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, save_npz, load_npz

from sklearn import svm
import string
import os
from io import open
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LinearRegression
import sklearn.metrics as M
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time as T


class BuildAndPredict:

    def __init__(self):

        self.x_train_features_matrix = load_npz("x_train.npz")
        self.y_train_column_matrix = load_npz("y_train.npz")
        self.x_test_features_matrix = load_npz("x_test.npz")
        self.y_test_column_matrix = load_npz("y_test.npz")

        self.x_train_features_matrix = csr_matrix(self.x_train_features_matrix.todense(), dtype=np.float64)
        self.x_test_features_matrix = csr_matrix(self.x_test_features_matrix.todense(), dtype=np.float64)
        self.y_train_column_matrix = self.y_train_column_matrix.todense()
        self.y_test_column_matrix = self.y_test_column_matrix.todense()

        start_0 = T.clock()

        start = T.clock()
        self.model = self.build_model()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Predicting on the test data set...")
        start = T.clock()
        self.predict_test_data()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Process done.\nTotal Time taken: {}\n\n".format(T.clock()-start_0))

    def build_model(self):
        model_1 = LinearRegression()

        model_1.fit(self.x_train_features_matrix, self.y_train_column_matrix)
        #model_1.fit(self.x_test_features_matrix, self.y_test_column_matrix)

        return model_1

    def predict_test_data(self):

        y_train_predicted_column_matrix = self.model.predict(self.x_train_features_matrix)
        print("\n")
        print("Train mean absolute error: {}".format(M.mean_absolute_error(self.y_train_column_matrix, y_train_predicted_column_matrix)))
        print("Train mean squared error: {}".format(M.mean_squared_error(self.y_train_column_matrix, y_train_predicted_column_matrix)))

        """
        for i in range(len(y_train_predicted_column_matrix)):
            print("\n\nRow: {}".format(i+1))
            print("Prediction: {}".format(y_train_predicted_column_matrix[i]))
            print("Actual:{}".format(self.y_train_column_matrix[i]))
        """

        y_test_predicted_column_matrix = self.model.predict(self.x_test_features_matrix)

        print("Test mean absolute error: {}".format(M.mean_absolute_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))
        print("Test mean squared error: {}".format(M.mean_squared_error(self.y_test_column_matrix, y_test_predicted_column_matrix)))
        print("\n")



        """
        for i in range(len(y_test_predicted_column_matrix)):
            print("\n\nRow: {}".format(i+1))
            print("Prediction: {}".format(y_test_predicted_column_matrix[i]))
            print("Actual:{}".format(self.y_test_column_matrix[i]))
        """


if __name__ == "__main__":

    print("\n\nBuilding the model...")
    BuildAndPredict()