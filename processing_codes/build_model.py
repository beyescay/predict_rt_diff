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
from sklearn.decomposition import PCA
import numpy as np
import time as T
from sklearn.model_selection import KFold
import statistics as stat


"""Modes to try"""
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from sklearn.svm import libsvm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import sklearn.linear_model as LM

import sys
sys.path.append("../")

class BuildAndTrainModel:

    def __init__(self):

        self.csv = "../data/models/model_comparison.csv"
        do_pca = False

        print("Loading the npz files of training data set...")
        self.x_train_features_matrix = load_npz("../data/npz_arrays/X.npz")
        self.y_train_column_matrix = load_npz("../data/npz_arrays/y.npz")

        print("Converting the features matrix to sparse matrix...")
        self.x_train_features_matrix = csr_matrix(self.x_train_features_matrix.todense(), dtype=np.float64)

        self.y_train_column_matrix = self.y_train_column_matrix.todense()

        if self.y_train_column_matrix.shape[0] == 1:
            self.y_train_column_matrix = self.y_train_column_matrix.T


        if do_pca:
            start = T.clock()
            print("Extracting best features...")
            self.feature_extraction()
            print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Building the model...")
        self.models = self.build_model()
        for model_num, model in enumerate(self.models):
            pickle.dump(model, open("../data/models/model_{}.sav".format(model_num), 'wb'))


        """
        print("Predicting the training data set error...\n")
        start = T.clock()
        self.predict_training_error()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))
        """

    def feature_extraction(self):
        pca = PCA()
        print("Total features before PCA: {}".format(self.x_train_features_matrix.shape[1]))
        pca.fit_transform(self.x_train_features_matrix.todense())
        print("Total features after PCA: {}".format(self.x_train_features_matrix.shape[1]))
        #print(pca.explained_variance_ratio_)

    def build_model(self):

        k_fold = KFold(n_splits=5)
        model_names = ["Adaboost", "Bagging", "ExtraTrees", "GradientBoost",
                       "RandomForest"]

        models = [AdaBoostRegressor(), BaggingRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(),
                  RandomForestRegressor(), LM.ARDRegression(), LM.BayesianRidge(), LM.ElasticNet(), LM.ElasticNetCV(),
                  LM.HuberRegressor(), LM.LarsCV(), LM.LassoCV(), LM.LassoLarsCV(), LM.LassoLarsIC(), LM.LogisticRegression(),
                  LM.MultiTaskLasso(), LM.MultiTaskElasticNetCV(), LM.PassiveAggressiveRegressor(),
                  LM.RidgeCV(), LM.SGDRegressor()]

        final_model_names = ["BayesianRidge", "ElasticNet", "ElasticNetCV",
                       "HuberRegressor", "LarsCV", "LassoCV", "LassoLarsCV", "LassoLarsIC", "LogisticRegression",
                       "MultiTaskLasso", "MultiTaskElasticNetCV", "PassiveAggressiveRegressor",
                       "RidgeCV", "SGDRegressor"]
        final_models = [AdaBoostRegressor(), BaggingRegressor(), ExtraTreesRegressor(), GradientBoostingRegressor(),
                  RandomForestRegressor(), LM.ARDRegression(), LM.BayesianRidge(), LM.ElasticNet(), LM.ElasticNetCV(),
                  LM.HuberRegressor(), LM.LarsCV(), LM.LassoCV(), LM.LassoLarsCV(), LM.LassoLarsIC(), LM.LogisticRegression(),
                  LM.MultiTaskLasso(), LM.MultiTaskElasticNetCV(), LM.PassiveAggressiveRegressor(),
                  LM.RidgeCV(), LM.SGDRegressor()]


        final_models = [GradientBoostingRegressor()]
        final_model_names = ["GradientBoost"]

        meta_features = np.zeros((self.x_train_features_matrix.shape[0], len(final_models)), dtype=np.float64)
        stacked_model = []
        trained_models = []
        with open(self.csv, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)

            for idx, model in enumerate(final_models):
                cross_validation_mae_error = []
                cross_validation_mse_error = []
                print(self.x_train_features_matrix.shape)
                print(self.y_train_column_matrix.shape)
                for train_index, test_index in k_fold.split(self.x_train_features_matrix):
                    model.fit(self.x_train_features_matrix[train_index, :].todense(), self.y_train_column_matrix[train_index])
                    y_train_predicted_column_matrix = model.predict(self.x_train_features_matrix[test_index, :])

                    for test_idx, index in enumerate(test_index):
                        meta_features[index, idx] = y_train_predicted_column_matrix[test_idx]

                    cross_validation_mae_error.append(M.mean_absolute_error(self.y_train_column_matrix[test_index], y_train_predicted_column_matrix))
                    cross_validation_mse_error.append(M.mean_squared_error(self.y_train_column_matrix[test_index], y_train_predicted_column_matrix))

                cv_mae = stat.mean(cross_validation_mae_error)
                cv_mdae = stat.median(cross_validation_mae_error)
                cv_mse = stat.mean(cross_validation_mse_error)
                cv_mdse = stat.median(cross_validation_mse_error)

                print("\nCross-validated Mean Absolute Error for {}: {}".format(final_model_names[idx], cv_mae))
                print("Cross-validated Median Absolute Error for {}: {}\n".format(final_model_names[idx], cv_mdae))
                print("Cross-validated Mean Squared Error for {}: {}".format(final_model_names[idx], cv_mse))
                print("Cross-validated Median Squared Error for {}: {}".format(final_model_names[idx], cv_mdse))

                model.fit(self.x_train_features_matrix.todense(), self.y_train_column_matrix)
                trained_models.append(model)

            

                #csv_writer.writerow([idx+1, model_names[idx], cv_mae, cv_mdae, cv_mse, cv_mdae])

                """
                except:
                    print("{} threw an exception".format(model_names[idx]))
                    continue
                """

        #model_1.fit(self.x_test_features_matrix, self.y_test_column_matrix)

        return trained_models

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