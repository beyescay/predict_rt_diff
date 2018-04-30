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
            print("Extracting best features...")
            self.feature_extraction()

        print("Building the model...")
        self.models, self.stack_models = self.build_model()
        for model_num, model in enumerate(self.models):
            pickle.dump(model, open("../data/models/model_{}.sav".format(model_num), 'wb'))

        for model_num, model in enumerate(self.stack_models):
            pickle.dump(model, open("../data/models/stack_model_{}.sav".format(model_num), 'wb'))


    def feature_extraction(self):
        pca = PCA()
        print("Total features before PCA: {}".format(self.x_train_features_matrix.shape[1]))
        pca.fit_transform(self.x_train_features_matrix.todense())
        print("Total features after PCA: {}".format(self.x_train_features_matrix.shape[1]))
        #print(pca.explained_variance_ratio_)

    def build_model(self):

        k_fold = KFold(n_splits=5)

        final_models = [BaggingRegressor(), GradientBoostingRegressor(), LM.HuberRegressor(), LM.RidgeCV(), LinearSVR(), MLPRegressor()]
        final_model_names = ["Bagging", "GradientBoost", "Huber", "RidgeCV", "LinearSVR", "NN"]

        meta_features = np.zeros((self.x_train_features_matrix.shape[0], len(final_models)), dtype=np.float64)
        stacking_model = []
        trained_models = []

        with open(self.csv, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)

            for idx, model in enumerate(final_models):
                cross_validation_mae_error = []
                cross_validation_mse_error = []

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

            svr = LinearSVR()
            svr.fit(meta_features, self.y_train_column_matrix)
            stacking_model.append(svr)

        return trained_models, stacking_model


if __name__ == "__main__":

    print("\n\nBuilding and training the model...")
    BuildAndTrainModel()