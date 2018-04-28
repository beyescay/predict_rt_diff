"""
A simple script that demonstrates how we classify textual data with sklearn.

"""

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

import csv
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle
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


class BuildAndTrainModel:

    def __init__(self):

        start_0 = T.clock()

        start = T.clock()
        print("Loading the npz files of training data set...")
        self.x_train_features_matrix = load_npz("./training_data/x_train.npz")
        self.y_train_column_matrix = load_npz("./training_data/y_train.npz")
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        start = T.clock()
        print("Converting the features matrix to sparse matrix...")
        self.x_train_features_matrix = csr_matrix(self.x_train_features_matrix.todense(), dtype=np.float64)
        self.y_train_column_matrix = self.y_train_column_matrix.todense()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        start = T.clock()
        print("Building the model...")
        self.model = self.build_model()
        pickle.dump(self.model, open("./models/model.sav", 'wb'))
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Predicting the training data set error...\n")
        start = T.clock()
        self.predict_training_error()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Process done.\nTotal Time taken: {}\n\n".format(T.clock()-start_0))

    def build_model(self):
        model_1 = LinearRegression()

        model_1.fit(self.x_train_features_matrix, self.y_train_column_matrix)
        #model_1.fit(self.x_test_features_matrix, self.y_test_column_matrix)

        return model_1

    def predict_training_error(self):

        y_train_predicted_column_matrix = self.model.predict(self.x_train_features_matrix)
        print("Training mean absolute error: {}".format(M.mean_absolute_error(self.y_train_column_matrix, y_train_predicted_column_matrix)))
        print("Training mean squared error: {}\n".format(M.mean_squared_error(self.y_train_column_matrix, y_train_predicted_column_matrix)))

        """
        for i in range(len(y_train_predicted_column_matrix)):
            print("\n\nRow: {}".format(i+1))
            print("Prediction: {}".format(y_train_predicted_column_matrix[i]))
            print("Actual:{}".format(self.y_train_column_matrix[i]))
        """


if __name__ == "__main__":

    print("\n\nBuilding and training the model...")
    BuildAndTrainModel()