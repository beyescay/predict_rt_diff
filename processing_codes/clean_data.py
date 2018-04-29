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
import pickle
import sys
sys.path.append("../")
from utility_codes.studio_name_conversion import StudioConversion

class DataCleaner:

    def __init__(self, movie_info_txt_file, num_actors=2, max_num_samples=None):
        self.movie_info_txt_file = movie_info_txt_file

        self.list_of_movies = []
        self.header = None
        self.formatted_header = None
        self.num_samples = 0
        self.num_actors = num_actors
        self.max_num_samples = max_num_samples

        self.score_based_features = ["actornames", "directedby", "writtenby"]

        self.one_hot_encoded_features = ["genre", "rating"]

        self.binned_one_hot_encoded_features = ["runtime", "intheaters"]

        self.feature_dict_objects = {"genre": {},
                                     "rating": {},
                                     "runtime": {},
                                     "intheaters": {},
                                     "release_type": {"wide": "0", "limited": "1"},
                                     "actornames": {},
                                     "directedby": {},
                                     "writtenby": {} }

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


        self.create_feature_dict_objects()

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

    def create_feature_dict_objects(self):

        print("Creating dict object for studio...")
        studio_feature_converter = StudioConversion(self.movie_info_txt_file)
        studio_dict = studio_feature_converter.get_studio_name_converter_dict()
        print("Saving dict object for studio...")
        self.save_feature_dict_objects(studio_dict, "studio")

        for idx_1, movie in enumerate(self.list_of_movies):

            for idx_2, field_name in enumerate(movie._fields):

                field_name_str = str(field_name)

                if field_name_str in self.feature_dict_objects:

                    current_feature = getattr(movie, field_name)
                    feature_dict_object = self.feature_dict_objects[str(field_name)]

                    if field_name_str == "genre":
                        list_of_current_string_features = re.split(self.delimiters_for_string_features, current_feature)
                        self.add_string_feature_to_dict_object(list_of_current_string_features, field_name_str, feature_dict_object)

                    elif field_name_str == "rating":
                        string_feature = current_feature.split('(')[0]
                        list_of_current_string_features = [string_feature]
                        self.add_string_feature_to_dict_object(list_of_current_string_features, field_name_str, feature_dict_object)

                    elif field_name_str == "runtime":
                        current_feature = current_feature.strip().lower()
                        current_feature = current_feature.split()[0]
                        try:
                            current_feature = int(current_feature)
                            current_feature = current_feature/15
                        except:
                            if not current_feature == "none":
                                print("Some unknown value in runtime: {}".format(current_feature))

                        self.add_binnable_feature_to_dict_object(current_feature, field_name_str, feature_dict_object)

                    elif field_name_str == "intheaters":
                        current_feature = current_feature.strip().lower()

                        if current_feature.find("wide") != -1:
                            current_feature = current_feature[0:current_feature.find("wide")-1]
                        elif current_feature.find("limited") != -1:
                            current_feature = current_feature[0:current_feature.find("limited")-1]

                        try:
                            dt = DT.datetime.strptime(str(current_feature), "%b %d, %Y")
                            year_num = dt.year
                            current_feature = year_num/10
                        except:
                            if not current_feature == "none":
                                print("Some unknown value in intheaters: {}".format(current_feature))

                        self.add_binnable_feature_to_dict_object(current_feature, field_name_str, feature_dict_object)


    def add_string_feature_to_dict_object(self, list_of_string_features, field_name_str, string_feature_dict):

        string_counter = len(string_feature_dict)

        for string_feature in list_of_string_features:
            string_feature = re.sub('[^a-zA-Z\d]', '', string_feature)
            string_feature = re.sub(' ', '', string_feature)
            string_feature = string_feature.strip().lower()

            if string_feature == "none":
                assert len(list_of_string_features) == 1
                string_feature = "none_" + field_name_str

            if string_feature not in string_feature_dict:
                string_feature_dict[string_feature] = str(string_counter)
                string_counter += 1


    def add_binnable_feature_to_dict_object(self, current_feature, field_name_str, feature_dict):

        counter = len(feature_dict)

        if current_feature == "none":
            current_feature = "none_" + field_name_str

        if current_feature not in feature_dict:
            feature_dict[current_feature] = str(counter)
            counter += 1







    def save_feature_dict_objects(self, obj, name):
        with open('../data/dictionary_objects/'+ name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    print("\n\nStarting the data cleaning process...\n\n")
    DataCleaner("../data/14642_movies_raw_data_prof_format.txt")
