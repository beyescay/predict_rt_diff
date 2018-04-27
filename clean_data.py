import csv
from collections import namedtuple
from itertools import imap, groupby
from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, save_npz
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import datetime as DT
import numpy as np
import re
import cProfile, pstats, StringIO
import time as T
import pandas as PD
import os.path

class DataCleaner:

    def __init__(self, movie_info_txt_file):
        self.movie_info_txt_file = movie_info_txt_file
        self.csv_file = self.convert_txt_to_csv()
        self.list_of_movies = []
        self.header = None
        self.formatted_header = None
        self.num_samples = 0

        self.dict_of_string_features = {"actornames": [{}, [], []],
                                        "genre": [{}, [], []],
                                        "studio": [{}, [], []],
                                        "directedby": [{}, [], []],
                                        "rating": [{}, [], []],
                                        "writtenby": [{}, [], []]
                                        }

        self.delimiters_for_string_features = re.compile(",|&")

        self.dict_of_numeric_features = {"audiencescore": [[], []],
                                         "criticscore": [[], []],
                                         "runtime": [[], []]}

        self.dict_of_timestamp_features = {"intheaters": [[], []]}

        profiler = cProfile.Profile()
        profiler.enable()

        start_0 = T.clock()
        print("Parsing the csv file...")
        start = T.clock()
        self.create_list_of_movies()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Creating various features arrays...")
        start = T.clock()
        self.create_features_arrays()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Vectorizing string features arrays...")
        start = T.clock()
        self.vectorize_string_features()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Merging all the various features arrays...")
        start = T.clock()
        self.x_features_matrix = self.merge_features_arrays()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Creating the y label matrix [audience score - critic score]...")
        start = T.clock()
        self.y_column_matrix = self.create_y_column_matrix()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Binning the data according to time period...")
        start = T.clock()
        self.year_range_to_row_indices_list_dict = self.bin_data(10, self.dict_of_timestamp_features["intheaters"])
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        print("Splitting the data into training and testing data...")
        start = T.clock()
        self.split_data_into_training_and_testing()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

        profiler.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(profiler)
        ps.sort_stats(sortby)
        ps.dump_stats("profiler_stats.txt")

        print("Data cleaning process done.\nTotal Time taken: {}\n\n".format(T.clock()-start_0))
        #print s.getvalue()

    def convert_txt_to_csv(self):
        csv_file = os.path.splitext(self.movie_info_txt_file)[0]+ ".csv"
        with open(self.movie_info_txt_file, 'r') as txt_file_reader:
            lines = txt_file_reader.readlines()


        with open(csv_file,'w') as csv_file_w:
            csv_file_writer = csv.writer(csv_file_w)
            for line in lines:
                line = line.split('\t')
                csv_file_writer.writerow(line)

        return csv_file


    def create_list_of_movies(self):

        with open(self.csv_file, mode="rb") as infile:
            reader = csv.reader(infile)
            self.header = next(reader)
            self.formatted_header = self.format_header_line()

            movie_info = namedtuple("movie_info", self.formatted_header)

            print movie_info._fields
            for data in imap(movie_info._make, reader):
                self.list_of_movies.append(data)

    def format_header_line(self):
        """
        Clean the header line of the csv file by removing the special characters such that it contains only alphabets.
        This is necessary to parse the csv file rows into namedtuple that will be created based on the header.
        :return: Return the list of formatted column names.
        """
        formatted_header = []
        for column_name in self.header:
            column_name = ''.join(i for i in column_name if i.isalpha())
            column_name = column_name.lower()
            formatted_header.append(column_name)

        return formatted_header

    def create_features_arrays(self):

        for idx_1, movie in enumerate(self.list_of_movies):

            current_field_name_to_dict_to_append_dict = {}
            current_field_name_to_numeric_value_dict = {}
            current_field_name_to_num_days_before_dict = {}
            encountered_not_available_field_value = False

            for idx_2, field_name in enumerate(movie._fields):

                if str(field_name) in self.dict_of_string_features:
                    current_movies_string_field_value_to_counter_dict = self.create_features_from_strings(getattr(movie, field_name), self.dict_of_string_features[str(field_name)][0], str(field_name))

                    if len(current_movies_string_field_value_to_counter_dict) == 0:
                        #encountered_not_available_field_value = True
                        #break
                        string_feature = "na_" + str(field_name)

                        if string_feature not in self.dict_of_string_features[str(field_name)][0]:
                            self.dict_of_string_features[str(field_name)][0][string_feature] = len(self.dict_of_string_features[str(field_name)][0]) + 1

                        current_movies_string_field_value_to_counter_dict[string_feature] = self.dict_of_string_features[str(field_name)][0][string_feature]

                    current_field_name_to_dict_to_append_dict[str(field_name)] = current_movies_string_field_value_to_counter_dict

                elif str(field_name) in self.dict_of_numeric_features:
                    continue
                    numeric_value = self.create_features_from_numeric_values(getattr(movie, field_name))

                    if numeric_value is None:
                        encountered_not_available_field_value = True
                        break
                    else:
                        current_field_name_to_numeric_value_dict[str(field_name)] = numeric_value

                elif field_name in self.dict_of_timestamp_features:
                    continue
                    """
                    """
                    num_days_before = self.create_features_from_timestamp_values(getattr(movie, field_name))

                    if num_days_before is None:
                        encountered_not_available_field_value = True
                        break
                    else:
                        current_field_name_to_num_days_before_dict[str(field_name)] = num_days_before

                else:
                    continue

            if not encountered_not_available_field_value:
                self.num_samples += 1
                for field_name_str in current_field_name_to_dict_to_append_dict:
                    self.dict_of_string_features[field_name_str][1].append(current_field_name_to_dict_to_append_dict[field_name_str])
                    self.dict_of_string_features[field_name_str][2].append(movie.movieid)

                for field_name_str in current_field_name_to_numeric_value_dict:
                    self.dict_of_numeric_features[field_name_str][0].append(current_field_name_to_numeric_value_dict[field_name_str])
                    self.dict_of_numeric_features[field_name_str][1].append(movie.movieid)

                for field_name_str in current_field_name_to_num_days_before_dict:
                    self.dict_of_timestamp_features[field_name_str][0].append(current_field_name_to_num_days_before_dict[field_name_str])
                    self.dict_of_timestamp_features[field_name_str][1].append(movie.movieid)

    def type_1(self, current_string_feature, string_feature_dict, field_name_str):

        list_of_current_string_features = current_string_feature.split(',')
        string_counter = len(string_feature_dict)
        current_dict_of_string_features = {}

        for string_feature in list_of_current_string_features:
            string_feature = string_feature.strip()
            string_feature = string_feature.lower()
            string_feature = ''.join(i for i in string_feature if not i.isspace())

            if string_feature == "none":
                assert len(list_of_current_string_features) == 1
                string_feature = "none_" + field_name_str

            if string_feature not in string_feature_dict:
                string_feature_dict[string_feature] = str(string_counter)
                string_counter += 1

            current_dict_of_string_features[string_feature] = string_feature_dict[string_feature]

        return current_dict_of_string_features

    def create_features_from_strings(self, current_string_feature, string_feature_dict, field_name_str):

        string_counter = len(string_feature_dict)
        current_dict_of_string_features = {}

        if field_name_str == "actornames":

            current_dict_of_string_features = self.type_1(current_string_feature, string_feature_dict, field_name_str)

        elif field_name_str == "genre":
            list_of_current_string_features = re.split(self.delimiters_for_string_features, current_string_feature)
            current_dict_of_string_features = {}

            for string_feature in list_of_current_string_features:
                string_feature = string_feature.strip()
                string_feature = string_feature.lower()
                string_feature = ''.join(i for i in string_feature if not i.isspace())

                if string_feature == "none":
                    assert len(list_of_current_string_features) == 1
                    string_feature = "none_genre"

                if string_feature not in string_feature_dict:
                    string_feature_dict[string_feature] = str(string_counter)
                    string_counter += 1

                current_dict_of_string_features[string_feature] = string_feature_dict[string_feature]

        elif field_name_str == "studio":
            current_dict_of_string_features = self.type_1(current_string_feature, string_feature_dict, field_name_str)

        elif field_name_str == "directedby":
            current_dict_of_string_features = self.type_1(current_string_feature, string_feature_dict, field_name_str)

        elif field_name_str == "rating":
            string_feature = current_string_feature.split()[0]
            string_feature = string_feature.lower()
            string_feature = ''.join(i for i in string_feature if not i.isspace())

            if string_feature == "none":
                string_feature = "none_" + field_name_str

            if string_feature not in string_feature_dict:
                string_feature_dict[string_feature] = str(string_counter)

            string_counter += 1
            current_dict_of_string_features[string_feature] = string_feature_dict[string_feature]

            return current_dict_of_string_features

        elif field_name_str == "writtenby":
            pass

        else:
            pass

        return current_dict_of_string_features

    def create_features_from_numeric_values(self, numeric_feature_string):

        try:
            numeric_feature_string = numeric_feature_string.strip()
            numeric_feature_string = numeric_feature_string.split()[0]
            numeric_feature_string = numeric_feature_string.split('%')[0]
            numeric_feature_string = float(numeric_feature_string)
            return numeric_feature_string

        except:
            return None

    def create_features_from_timestamp_values(self, timestamp_feature_string):
        try:
            dt = DT.datetime.strptime(timestamp_feature_string, "%b %d, %Y")
            dt_future = DT.date(2018, 05, 01)
            return (dt_future - dt.date()).days
            #return dt.year
        except:
            return None

    def vectorize_string_features(self):

        for field_name_str, string_feature_list in self.dict_of_string_features.items():
            print("Vectorizing {}...".format(field_name_str))
            DV = DictVectorizer()
            string_feature_array_matrix = DV.fit_transform(string_feature_list[1])
            self.dict_of_string_features[field_name_str].append(string_feature_array_matrix)
            print("Total columns in array: {}".format(string_feature_array_matrix.toarray().shape[1]))
            print("Total column names: {}".format(len(DV.get_feature_names())))
            print("Done vectorizing. Converting to data frame...")
            df = PD.DataFrame(string_feature_array_matrix.toarray(), dtype=np.float64, index=self.dict_of_string_features[field_name_str][2], columns=DV.get_feature_names())
            print("Done creating data frame. Saving data frame to csv...")
            df.to_csv("{}.csv".format(field_name_str))
            print("Done saving to csv. Converting to sparse data frame...")
            df = df.to_sparse()
            print("Done saving to csv. Saving data frame to pickle...")
            df.to_pickle("{}.pkl".format(field_name_str))
            print("Done saving to pickle.\n\n")


    def merge_features_arrays(self):
        all_features_matrix = None

        for idx, column_name in enumerate(self.formatted_header):

            if column_name in self.dict_of_string_features:
                feature_matrix = self.dict_of_string_features[column_name][3]
            elif column_name == "runtime":
                feature_matrix = coo_matrix(np.asarray(self.dict_of_numeric_features[column_name][0]))
            elif column_name in self.dict_of_timestamp_features:
                feature_matrix = coo_matrix(np.asarray(self.dict_of_timestamp_features[column_name][0]))
            else:
                continue

            if all_features_matrix is None:
                all_features_matrix = feature_matrix
            else:

                if not feature_matrix.shape[0] == self.num_samples:
                    feature_matrix = feature_matrix.T

                assert feature_matrix.shape[0] == self.num_samples
                all_features_matrix = hstack([all_features_matrix, feature_matrix])

        return all_features_matrix

    def create_y_column_matrix(self):
        y_column_matrix = csr_matrix(self.dict_of_numeric_features["audiencescore"]) - csr_matrix(self.dict_of_numeric_features["criticscore"])

        if not y_column_matrix.shape[0] == self.num_samples:
            y_column_matrix = y_column_matrix.T
        return y_column_matrix

    def bin_data(self, bin_size, list_to_be_binned):

        key_fn = lambda (_, x): x/bin_size

        idx_year_pair_list = zip(range(len(list_to_be_binned)), list_to_be_binned)
        idx_year_pair_list.sort(key=lambda pair: pair[1])

        result = {}

        for item, grp in groupby(idx_year_pair_list, key_fn):
            list_of_indices_for_this_group, _ = (list(t) for t in zip(*grp))
            result[(item * bin_size, (item+1) * bin_size)] = list_of_indices_for_this_group

        return result

    def split_data_into_training_and_testing(self, test_size=0.20):

        x_train_features_matrix = None
        y_train_column_matrix = None

        x_test_features_matrix = None
        y_test_column_matrix = None

        for year_range, list_of_indices in self.year_range_to_row_indices_list_dict.items():

            x = list_of_indices

            x_train_indices, x_test_indices, _, _ = train_test_split(x, x, test_size=test_size, random_state=42)

            current_x_train = self.x_features_matrix.tocsr()[x_train_indices, :]

            current_y_train = self.y_column_matrix[x_train_indices]

            current_x_test = self.x_features_matrix.tocsr()[x_test_indices, :]
            current_y_test = self.y_column_matrix[x_test_indices]

            if x_train_features_matrix is None:
                x_train_features_matrix = current_x_train
                y_train_column_matrix = current_y_train
                x_test_features_matrix = current_x_test
                y_test_column_matrix = current_y_test
            else:
                x_train_features_matrix = vstack([x_train_features_matrix, current_x_train])
                y_train_column_matrix = vstack([y_train_column_matrix, current_y_train])
                x_test_features_matrix = vstack([x_test_features_matrix, current_x_test])
                y_test_column_matrix = vstack([y_test_column_matrix, current_y_test])

        print("Total training samples: {}".format(x_train_features_matrix.shape[0]))
        assert x_train_features_matrix.shape[0] == y_train_column_matrix.shape[0]
        print("Total test samples: {}".format(x_test_features_matrix.shape[0]))
        assert x_test_features_matrix.shape[0] == y_test_column_matrix.shape[0]

        print("\nSaving the training and testing data...")

        save_npz("x_train.npz", coo_matrix(x_train_features_matrix, dtype=np.float64))
        save_npz("y_train.npz", coo_matrix(y_train_column_matrix, dtype=np.float64))
        save_npz("x_test.npz", coo_matrix(x_test_features_matrix, dtype=np.float64))
        save_npz("y_test.npz", coo_matrix(y_test_column_matrix, dtype=np.float64))


if __name__ == "__main__":
    print("\n\nStarting the data cleaning process...\n\n")
    DataCleaner("8964_movies_data.txt")
