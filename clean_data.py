import csv
from collections import namedtuple
from itertools import imap
import pandas as PD
from sklearn.feature_extraction import DictVectorizer
import datetime as DT
import numpy as np
import re


class DataCleaner:

    def __init__(self, movie_info_csv_file):
        self.csv_file = movie_info_csv_file
        self.list_of_movies = []

        with open(self.csv_file, mode="rb") as infile:
            reader = csv.reader(infile)
            self.header = next(reader)
            formatted_header = self.format_header_line()

            movie_info = namedtuple("movie_info", formatted_header)

            self.string_field_types_to_info_dict = {"cast": [{}, []],
                                                    "directedby": [{}, []],
                                                    "genre": [{}, []],
                                                    "rating": [{}, []],
                                                    "studio": [{}, []],
                                                    "writtenby": [{}, []]}

            self.delimiters_for_string_features = re.compile(",|&|\|")

            self.dict_of_numeric_features = {"audiencescore": [],
                                             "criticscore": [],
                                             "runtime": []}

            self.dict_of_timestamp_features = {"intheaters": [],
                                               "ondiscstreaming": {}}

            for data in imap(movie_info._make, reader):
                self.list_of_movies.append(data)

        self.make_features_array()
        #d = self.string_field_types_to_info_dict["cast"][0]
        #print(["{}: {}\n".format(k, d[k]) for k in sorted(d, key=d.get)])

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

    def make_features_array(self):

        for idx_1, movie in enumerate(self.list_of_movies):

            current_field_name_to_dict_to_append_dict = {}
            current_field_name_to_numeric_value_dict = {}
            current_field_name_to_num_days_before_dict = {}
            encountered_not_available_field_value = False

            for idx_2, field_name in enumerate(movie._fields):

                if str(field_name) in self.string_field_types_to_info_dict:
                    current_movies_string_field_value_to_counter_dict = self.make_features_from_strings(getattr(movie, field_name), self.string_field_types_to_info_dict[str(field_name)][0])

                    if current_movies_string_field_value_to_counter_dict is None:
                        encountered_not_available_field_value = True
                        break

                    current_field_name_to_dict_to_append_dict[str(field_name)] = current_movies_string_field_value_to_counter_dict

                elif str(field_name) in self.dict_of_numeric_features:
                    numeric_value = self.make_features_from_numeric_values(getattr(movie, field_name))

                    if numeric_value is None:
                        encountered_not_available_field_value = True
                        break
                    else:
                        current_field_name_to_numeric_value_dict[str(field_name)] = numeric_value

                elif field_name in self.dict_of_timestamp_features:
                    num_days_before = self.make_features_from_timestamp_values(getattr(movie, field_name))

                    if num_days_before is None:
                        encountered_not_available_field_value = True
                        break
                    else:
                        current_field_name_to_num_days_before_dict[str(field_name)] = num_days_before

                else:
                    continue

            if not encountered_not_available_field_value:
                for field_name_str in current_field_name_to_dict_to_append_dict:
                    self.string_field_types_to_info_dict[field_name_str][1].append(current_field_name_to_dict_to_append_dict[field_name_str])

                for field_name_str in current_field_name_to_numeric_value_dict:
                    self.dict_of_numeric_features[field_name_str] = current_field_name_to_numeric_value_dict[field_name_str]

                for field_name_str in current_field_name_to_num_days_before_dict:
                    self.dict_of_timestamp_features[field_name_str] = current_field_name_to_num_days_before_dict[field_name_str]

            """
            if idx == 0:
                num_features = feature_array.shape[1]
                features_array = feature_array
            else:
                assert feature_array.shape[1] == num_features
                np.vstack([features_array, feature_array])
            """
        #np.save("")

    def make_features_from_strings(self, current_string_feature, string_feature_dict):

        string_counter = len(string_feature_dict)
        list_of_current_string_features = re.split(self.delimiters_for_string_features, current_string_feature)

        current_dict_of_string_features = {}

        for string_feature in list_of_current_string_features:
            string_feature = string_feature.strip()
            string_feature = string_feature.lower()
            string_feature = ''.join(i for i in string_feature if i.isalpha())

            if string_feature == "na":
                return None

            if string_feature not in string_feature_dict:
                string_feature_dict[string_feature] = str(string_counter)
                string_counter += 1

            current_dict_of_string_features[string_feature] = string_feature_dict[string_feature]

        return current_dict_of_string_features

    def make_features_from_numeric_values(self, numeric_feature_string):

        try:
            numeric_feature_string = numeric_feature_string.strip()
            numeric_feature_string = numeric_feature_string.split()[0]
            numeric_feature_string = numeric_feature_string.split('%')[0]
            numeric_feature_string = float(numeric_feature_string)
            return numeric_feature_string

        except:
            return None

    def make_features_from_timestamp_values(self, timestamp_feature_string):
        try:
            dt = DT.datetime.strptime(timestamp_feature_string, "%b %d, %Y")
            dt_future = DT.date(2018, 05, 01)
            return (dt_future - dt.date()).days
        except:
            return None

    def bin_data(self):
        pass

    def split_data_into_training_and_testing(self):
        pass


if __name__ == "__main__":
    DataCleaner("movies_raw_data.csv")
