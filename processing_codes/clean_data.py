import csv
from collections import namedtuple
from itertools import groupby
from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, save_npz
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import datetime as DT
import numpy as np
import re
import time as T
import pandas as PD
import os.path

from studio_name_conversion import StudioConversion

class DataCleaner:

    def __init__(self, movie_info_txt_file, mode="train", to_csv=False, num_actors=2, max_num_samples=None):
        self.movie_info_txt_file = movie_info_txt_file
        self.mode = mode

        if to_csv:
            self.csv_file = self.convert_txt_to_csv()

        self.list_of_movies = []
        self.header = None
        self.formatted_header = None
        self.num_samples = 0
        self.num_actors = num_actors
        self.max_num_samples = max_num_samples

        self.dict_of_string_features = {"actornames": [{}, [], []],
                                        "genre": [{}, [], []],
                                        "studio": [{}, [], []],
                                        "directedby": [{}, [], []],
                                        "rating": [{}, [], []],
                                        "writtenby": [{}, [], []]
                                        }

        self.delimiters_for_string_features = re.compile(",")

        self.dict_of_numeric_features = {"audiencescore": [[], []],
                                         "criticscore": [[], []],
                                         "runtime": [[], []]}

        self.dict_of_timestamp_features = {"intheaters": [[], []],
                                           "intheaters_year": [[], []]}


        start_0 = T.clock()
        print("Parsing the txt file...")
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

        if self.mode == "train":
            print("Binning the data according to time period...")
            start = T.clock()
            self.year_range_to_row_indices_list_dict = self.bin_data(10, self.dict_of_timestamp_features["intheaters_year"][0])
            print("Done. Time taken: {}\n\n".format(T.clock()-start))

            print("Splitting the data into training and testing data...")
            start = T.clock()
            self.split_data_into_training_and_testing()
            print("Done. Time taken: {}\n\n".format(T.clock()-start))

        elif self.mode == "test":
            print("Saving the cleaned test data in model input format...")
            start = T.clock()
            self.save_data_in_test_format()
            print("Done. Time taken: {}\n\n".format(T.clock()-start))

            #TODO: load the dictionary objects for the string features.

        print("Data cleaning process done.\nTotal Time taken: {}\n\n".format(T.clock()-start_0))

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

        with open(self.movie_info_txt_file, mode="r") as infile:

            self.header = infile.readline().strip()
            self.header = self.header.split('\t')
            self.formatted_header = self.format_header_line()

            movie_info = namedtuple("movie_info", self.formatted_header)

            for line in infile:
                line = line.strip()
                line = line.split('\t')
                data = movie_info(*line)
                if data.audiencescore.upper() == "NONE" or data.criticscore.upper() == "NONE":
                    #print "No score available for this movie", data.movieid
                    continue
                self.list_of_movies.append(data)

                if self.max_num_samples and len(self.list_of_movies) > self.max_num_samples:
                    break

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
            current_field_name_to_year_dict = {}
            encountered_not_available_field_value = False

            for idx_2, field_name in enumerate(movie._fields):

                if str(field_name) in self.dict_of_string_features:
                    current_movies_string_field_value_to_counter_dict = self.create_features_from_strings(getattr(movie, field_name), self.dict_of_string_features[str(field_name)][0], str(field_name))
                    current_field_name_to_dict_to_append_dict[str(field_name)] = current_movies_string_field_value_to_counter_dict

                elif str(field_name) in self.dict_of_numeric_features:
                    numeric_value = self.create_features_from_numeric_values(getattr(movie, field_name))

                    if numeric_value is None:
                        #print("Encountered none in numeric values")
                        encountered_not_available_field_value = True
                        break
                    else:
                        current_field_name_to_numeric_value_dict[str(field_name)] = numeric_value

                elif field_name in self.dict_of_timestamp_features:
                    date_object = self.create_features_from_timestamp_values(getattr(movie, field_name))
                    if date_object is None:
                        #print("Encountered none in timestamp")
                        encountered_not_available_field_value = True
                        break
                    else:
                        [num_days_before, year] = date_object
                        current_field_name_to_num_days_before_dict[str(field_name)] = num_days_before
                        current_field_name_to_year_dict[str(field_name)] = year

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

                    self.dict_of_timestamp_features["intheaters_year"][0].append(current_field_name_to_year_dict[field_name_str])
                    self.dict_of_timestamp_features["intheaters_year"][1].append(movie.movieid)
            else:
                continue
        print("Total movies: {}".format(idx_1+1))
        print("Total movies with all values: {}".format(self.num_samples))

    def create_features_from_strings(self, current_string_feature, string_feature_dict, field_name_str):

        string_counter = len(string_feature_dict)
        current_dict_of_string_features = {}
        list_of_current_string_features = []

        if field_name_str in ["actornames", "studio", "directedby", "writtenby"]:
            list_of_current_string_features = current_string_feature.split(',')

            if self.num_actors and field_name_str == "actornames":
                list_of_current_string_features = list_of_current_string_features[0:self.num_actors]

        elif field_name_str == "genre":
            list_of_current_string_features = re.split(self.delimiters_for_string_features, current_string_feature)

        elif field_name_str == "rating":
            string_feature = current_string_feature.split('(')[0]
            list_of_current_string_features = [string_feature]

        else:
            pass

        for string_feature in list_of_current_string_features:

            string_feature = re.sub('[^a-zA-Z\d]', '', string_feature)
            string_feature = re.sub(' ', '' , string_feature)
            string_feature = string_feature.strip().lower()

            #string_feature = ''.join(i for i in string_feature if not i.isspace())

            if string_feature == "none":
                assert len(list_of_current_string_features) == 1
                string_feature = "none_" + field_name_str

            if string_feature not in string_feature_dict:
                string_feature_dict[string_feature] = str(string_counter)
                string_counter += 1

            current_dict_of_string_features[string_feature] = string_feature_dict[string_feature]

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

        timestamp_feature_string = timestamp_feature_string.strip()
        if timestamp_feature_string.find("wide") != -1:
            timestamp_feature_string = timestamp_feature_string[0:timestamp_feature_string.find("wide")-1]
        elif timestamp_feature_string.find("limited") != -1:
            timestamp_feature_string = timestamp_feature_string[0:timestamp_feature_string.find("limited")-1]
        else:
            #print(timestamp_feature_string)
            return None

        timestamp_feature_string = timestamp_feature_string.strip()
        #print(timestamp_feature_string)
        dt = DT.datetime.strptime(str(timestamp_feature_string), "%b %d, %Y")
        #print(dt)
        dt_future = DT.date(2018, 5, 1)
        num_days_before = (dt_future - dt.date()).days
        year_num = dt.year
        date_object = [num_days_before, year_num]
        #print(num_days_before)
        #print(year_num)
        return date_object
        #return dt.year


    def vectorize_string_features(self):

        for field_name_str, string_feature_list in self.dict_of_string_features.items():
            print("Vectorizing {}...".format(field_name_str))

            DV = DictVectorizer()
            string_feature_array_matrix = DV.fit_transform(string_feature_list[1])
            self.dict_of_string_features[field_name_str].append(string_feature_array_matrix)
            print("Total columns in array: {}".format(string_feature_array_matrix.toarray().shape[1]))
            print("Total column names: {}".format(len(DV.get_feature_names())))

            """
            if field_name_str == "rating":
                print(DV.get_feature_names())
                
            """
            #print("Done vectorizing. Converting to data frame...")
            print("Done vectorizing. ")
            #df = PD.DataFrame(string_feature_array_matrix.toarray(), dtype=np.float64, index=self.dict_of_string_features[field_name_str][2], columns=DV.get_feature_names())
            #print("Done creating data frame. Saving data frame to csv...")
            #df.to_csv("{}.csv".format(field_name_str))
            #print("Done saving to csv. Converting to sparse data frame...")
            #df = df.to_sparse()
            #print("Done saving to csv. Saving data frame to pickle...")
            #df.to_pickle("{}.pkl".format(field_name_str))
            #print("Done saving to pickle.\n\n")


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
        y_column_matrix = csr_matrix(self.dict_of_numeric_features["audiencescore"][0]) - csr_matrix(self.dict_of_numeric_features["criticscore"][0])

        if not y_column_matrix.shape[0] == self.num_samples:
            y_column_matrix = y_column_matrix.T
        return y_column_matrix

    def bin_data(self, bin_size, list_to_be_binned):

        key_fn = lambda _x: _x[1]/bin_size

        idx_year_pair_list = zip(range(len(list_to_be_binned)), list_to_be_binned)
        idx_year_pair_list = sorted(idx_year_pair_list, key=lambda pair: pair[1])

        result = {}

        for item, grp in groupby(idx_year_pair_list, key_fn):
            list_of_indices_for_this_group, _ = (list(t) for t in zip(*grp))
            result[(item * bin_size, (item+1) * bin_size)] = list_of_indices_for_this_group
        #print(result.keys())
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

        save_npz("./training_data/x_train.npz", coo_matrix(x_train_features_matrix, dtype=np.float64))
        save_npz("./training_data/y_train.npz", coo_matrix(y_train_column_matrix, dtype=np.float64))
        save_npz("./test_data/x_test.npz", coo_matrix(x_test_features_matrix, dtype=np.float64))
        save_npz("./test_data/y_test.npz", coo_matrix(y_test_column_matrix, dtype=np.float64))

    def save_data_in_test_format(self):
        x_test_features_matrix = self.x_features_matrix.tocsr()
        y_test_column_matrix = self.y_column_matrix

        save_npz("./test_data/x_test.npz", coo_matrix(x_test_features_matrix, dtype=np.float64))
        save_npz("./test_data/y_test.npz", coo_matrix(y_test_column_matrix, dtype=np.float64))


if __name__ == "__main__":
    print("\n\nStarting the data cleaning process...\n\n")
    DataCleaner("14642_movies_raw_data_prof_format.txt")
