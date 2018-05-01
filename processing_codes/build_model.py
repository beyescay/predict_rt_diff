import csv
import pickle
from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, save_npz, load_npz
from io import open
import sklearn.metrics as M
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import KFold
import statistics as stat
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import LinearSVR
import sklearn.linear_model as LM
from  sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append("../")


class BuildAndTrainModel:

    def __init__(self):
        print("Loading the npz files of training data set...")
        self.x_train_features_matrix = load_npz("../data/npz_arrays/X.npz")
        self.y_train_column_matrix = load_npz("../data/npz_arrays/y.npz")

        """Write the output to the below file"""
        self.output_individual_models = "../data/best_cv_error_individual_models.csv"

        print("Converting the features matrix to sparse matrix...")
        self.x_train_features_matrix = csr_matrix(self.x_train_features_matrix.todense(), dtype=np.float64)

        self.y_train_column_matrix = self.y_train_column_matrix.todense()

        if self.y_train_column_matrix.shape[0] == 1:
            self.y_train_column_matrix = self.y_train_column_matrix.T

        final_model_names = ["Extratrees",
                             "Bagging",
                             "GradientBoost",
                             "HuberRegressor",
                             "SVM"]
        self.models, self.stack_models = self.build_model()
        for model_num, model in enumerate(self.models):
            pickle.dump(model, open("../data/models/model_{}.sav".format(final_model_names[model_num]), 'wb'))

        stack_model_names = ["svm"]
        for model_num, model in enumerate(self.stack_models):
            pickle.dump(model, open("../data/models/stack_model_{}.sav".format(stack_model_names[model_num]), 'wb'))

    def build_model(self):

        k_fold = KFold(n_splits=10)

        final_models = [ExtraTreesRegressor(n_estimators=30),
                        BaggingRegressor(n_estimators=30),
                        GradientBoostingRegressor(),
                        LM.HuberRegressor(),
                        LinearSVR(),
                        ]

        final_model_names = ["Extratrees",
                             "Bagging",
                             "GradientBoost",
                             "HuberRegressor",
                             "SVM"]

        meta_features = np.zeros((self.x_train_features_matrix.shape[0], len(final_models)), dtype=np.float64)
        stacking_model = []
        trained_models = []

        with open(self.output_individual_models, 'w') as ind_csv_file:
            ind_csv_writer = csv.writer(ind_csv_file)
            ind_csv_writer.writerow(["Model", "Mean Absolute Error", "Mean Squared Error"])

            for idx, model in enumerate(final_models):
                print("\n\nBuilding the model - {}...".format(final_model_names[idx]))
                cross_validation_mae_error = []
                cross_validation_mse_error = []
                cross_validation_models = []

                for train_index, test_index in k_fold.split(self.x_train_features_matrix):

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(self.x_train_features_matrix[train_index, :].todense(), self.y_train_column_matrix[train_index])
                        y_train_predicted_column_matrix = model.predict(self.x_train_features_matrix[test_index, :])

                        for test_idx, index in enumerate(test_index):
                            meta_features[index, idx] = y_train_predicted_column_matrix[test_idx]

                        cross_validation_mae_error.append(M.mean_absolute_error(self.y_train_column_matrix[test_index], y_train_predicted_column_matrix))
                        cross_validation_mse_error.append(M.mean_squared_error(self.y_train_column_matrix[test_index], y_train_predicted_column_matrix))

                        cross_validation_models.append(model)

                best_model_mae = min(cross_validation_mae_error)
                best_model_mse = min(cross_validation_mse_error)

                ind_csv_writer.writerow([final_model_names[idx], best_model_mae, best_model_mse])

                #cv_mae = stat.mean(cross_validation_mae_error)
                #cv_mdae = stat.median(cross_validation_mae_error)
                #cv_mse = stat.mean(cross_validation_mse_error)
                #cv_mdse = stat.median(cross_validation_mse_error)

                print("\nCross-validated best Mean Absolute Error for {}: {}".format(final_model_names[idx], best_model_mae))
                #print("Cross-validated Median Absolute Error for {}: {}\n".format(final_model_names[idx], cv_mdae))
                print("Cross-validated best Mean Squared Error for {}: {}".format(final_model_names[idx], best_model_mse))
                #print("Cross-validated Median Squared Error for {}: {}\n".format(final_model_names[idx], cv_mdse))


            """ Train the best cv model on the entire data set"""
            #with warnings.catch_warnings():
            #    warnings.simplefilter("ignore")
            #    model.fit(self.x_train_features_matrix.todense(), self.y_train_column_matrix)
            #    trained_models.append(model)


        """Stack the models using SVM"""
        #svr = LinearSVR()
        #svr.fit(meta_features, self.y_train_column_matrix)
        #stacking_model.append(svr)

        return trained_models, stacking_model


if __name__ == "__main__":

    print("\n\nBuilding and training the model...")
    BuildAndTrainModel()