from collections import namedtuple
from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, save_npz
from sklearn.feature_extraction import DictVectorizer
import datetime as DT
import numpy as np
import re
import pickle
import statistics
import sys

sys.path.append("../")


class DataWrangler:

    def __init__(self, movie_info_txt_file, mode="train", num_actors=2):
        self.movie_info_txt_file = movie_info_txt_file
        self.list_of_movies = []
        self.header = None
        self.formatted_header = None
        self.num_samples = 0
        self.num_actors = num_actors
        self.y_column_matrix = []
        self.mode = mode

        self.feature_dict_objects = {"genre": {},
                                     "rating": {},
                                     "runtime": {},
                                     "intheaters": {},
                                     "release_type": {},
                                     "actornames": {},
                                     "directedby": {},
                                     "writtenby": {},
                                     "studio": {},
                                     "studio_mapper": {}
                                     }

        self.dict_of_score_based_features = {"actornames": [],
                                             "directedby": [],
                                             "writtenby": []
                                             }
        self.dict_of_one_hot_encodable_features = {"genre": [],
                                                   "studio": [],
                                                   "rating": [],
                                                   "runtime": [],
                                                   "intheaters": [],
                                                   "release_type": []
                                                    }

        self.dict_of_one_hot_encoded_matrix = {}

        self.delimiters_for_string_features = re.compile(",")

        self.abbreviations_set_for_names = {"jr", "jr.", "sr", "sr."}

        self.dict_of_numeric_features = {"audiencescore": [],
                                         "criticscore": []
                                         }

        print("Parsing the txt file...\n")
        self.create_list_of_movies()

        print("Loading the feature dicts...\n")
        self.load_feature_dict_objects()

        print("\nCreating various features arrays...\n")
        self.create_features_arrays()

        print("\nDict-Vectorizing string features arrays...")
        self.vectorize_string_features()

        print("\nMerging all the various features arrays...\n")
        self.x_features_matrix = self.merge_features_arrays()

        print("Saving the data...")
        self.save_npz_data()

        print("\nData mapping process done.\n")

    def load_feature_dict_objects(self):
        for dict_object_name in self.feature_dict_objects:
            print("Loading dict object for {}...".format(dict_object_name))
            with open('../data/dictionary_objects/' + dict_object_name + '.pkl', 'rb') as f:
                self.feature_dict_objects[dict_object_name] = pickle.load(f)

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

            audience_score = self.create_features_from_numeric_values(movie.audiencescore)
            if audience_score is None:
                print("Invalid numerical value given for audience score. So skipping the row")
                continue

            critic_score = self.create_features_from_numeric_values(movie.criticscore)
            if critic_score is None:
                print("Invalid numerical value given for critic score. So skipping the row")
                continue

            self.y_column_matrix.append(audience_score-critic_score)

            for idx_2, field_name in enumerate(movie._fields):

                field_name_str = str(field_name)

                if field_name_str in self.feature_dict_objects:
                    current_feature = getattr(movie, field_name)
                    feature_dict_object = self.feature_dict_objects[str(field_name)]

                    if field_name_str == "genre":
                        list_of_current_string_features = re.split(self.delimiters_for_string_features, current_feature)
                        genre_dict_for_current_movie = self.add_string_feature_to_dict_object(list_of_current_string_features, feature_dict_object)
                        self.dict_of_one_hot_encodable_features["genre"].append(genre_dict_for_current_movie)

                    elif field_name_str == "rating":
                        string_feature = current_feature.split('(')[0]
                        list_of_current_string_features = [string_feature]
                        rating_dict_for_current_movie = self.add_string_feature_to_dict_object(list_of_current_string_features, feature_dict_object)
                        if "none" in rating_dict_for_current_movie:
                            rating_dict_for_current_movie.pop("none", None)
                            rating_dict_for_current_movie["g"] = feature_dict_object["g"]
                        self.dict_of_one_hot_encodable_features["rating"].append(rating_dict_for_current_movie)

                    elif field_name_str == "runtime":
                        current_feature = current_feature.strip().lower()
                        current_feature = current_feature.split()[0]
                        runtime_dict = {}
                        try:
                            current_feature = int(current_feature)
                            current_feature = int(current_feature/15)
                            runtime_dict[current_feature] = feature_dict_object[current_feature]
                        except:
                            if current_feature == "none":
                                runtime_dict[current_feature] = feature_dict_object[current_feature]
                            else:
                                print("Some unknown value in runtime: {}".format(current_feature))
                                runtime_dict["none"] = feature_dict_object["none"]

                        self.dict_of_one_hot_encodable_features["runtime"].append(runtime_dict)

                    elif field_name_str == "intheaters":
                        intheaters_dict = {}
                        release_type_dict = {}

                        if current_feature.find("wide") != -1:
                            current_feature = current_feature[0:current_feature.find("wide")-1]
                            release_type_dict["wide"] = self.feature_dict_objects["release_type"]["wide"]
                        elif current_feature.find("limited") != -1:
                            current_feature = current_feature[0:current_feature.find("limited")-1]
                            release_type_dict["limited"] = self.feature_dict_objects["release_type"]["limited"]
                        else:
                            release_type_dict["none"] = self.feature_dict_objects["release_type"]["none"]

                        try:
                            dt = DT.datetime.strptime(str(current_feature), "%b %d, %Y")
                            year_num = dt.year
                            current_feature = int(year_num/10)
                            intheaters_dict[current_feature] = feature_dict_object[current_feature]
                        except:
                            current_feature = re.sub('[^a-zA-Z\d]', '', current_feature)
                            current_feature = re.sub(' ', '', current_feature)
                            current_feature = current_feature.strip().lower()
                            if current_feature == "none":
                               intheaters_dict[current_feature] = feature_dict_object[current_feature]
                            else:
                                print("Some unknown value in intheaters: {}".format(current_feature))
                                intheaters_dict["none"] = feature_dict_object["none"]

                        self.dict_of_one_hot_encodable_features["intheaters"].append(intheaters_dict)
                        self.dict_of_one_hot_encodable_features["release_type"].append(release_type_dict)

                    elif field_name_str == "actornames":
                        full_list_of_current_string_features = current_feature.split(',')
                        top_actors_list_count = 0
                        cast_weight_list = []

                        for current_string_feature in full_list_of_current_string_features:
                            current_string_feature = re.sub('[^a-zA-Z\d]', '', current_string_feature)
                            current_string_feature = re.sub(' ', '', current_string_feature)
                            current_string_feature = current_string_feature.strip().lower()
                            if current_string_feature not in self.abbreviations_set_for_names:
                                top_actors_list_count += 1

                                if top_actors_list_count > self.num_actors:
                                    break

                                if current_string_feature in self.feature_dict_objects["actornames"]:
                                    cast_weight_list.append(self.feature_dict_objects["actornames"][current_string_feature])
                                else:
                                    print("Unknown actor name in {}: {}".format(idx_1, current_string_feature))
                                    cast_weight_list.append(self.feature_dict_objects["actornames"]["none"])

                        self.dict_of_score_based_features["actornames"].append(statistics.mean(cast_weight_list))

                    elif field_name_str == "directedby":
                        list_of_current_string_features = current_feature.split(',')
                        director_weight_list = []

                        for current_string_feature in list_of_current_string_features:
                            current_string_feature = re.sub('[^a-zA-Z\d]', '', current_string_feature)
                            current_string_feature = re.sub(' ', '', current_string_feature)
                            current_string_feature = current_string_feature.strip().lower()
                            if current_string_feature not in self.abbreviations_set_for_names:

                                if current_string_feature in self.feature_dict_objects["directedby"]:
                                    director_weight_list.append(self.feature_dict_objects["directedby"][current_string_feature])
                                else:

                                    print("Unknown director name in {}: {}".format(idx_1, current_string_feature))
                                    director_weight_list.append(self.feature_dict_objects["directedby"]["none"])

                        self.dict_of_score_based_features["directedby"].append(statistics.mean(director_weight_list))

                    elif field_name_str == "writtenby":
                        list_of_current_string_features = current_feature.split(',')
                        writer_weight_list = []

                        for current_string_feature in list_of_current_string_features:
                            current_string_feature = re.sub('[^a-zA-Z\d]', '', current_string_feature)
                            current_string_feature = re.sub(' ', '', current_string_feature)
                            current_string_feature = current_string_feature.strip().lower()
                            if current_string_feature not in self.abbreviations_set_for_names:

                                if current_string_feature in self.feature_dict_objects["writtenby"]:
                                    writer_weight_list.append(self.feature_dict_objects["writtenby"][current_string_feature])
                                else:
                                    print("Unknown writer name in {}: {}".format(idx_1, current_string_feature))
                                    writer_weight_list.append(self.feature_dict_objects["writtenby"]["none"])

                        self.dict_of_score_based_features["writtenby"].append(statistics.mean(writer_weight_list))

                    elif field_name_str == "studio":
                        #list_of_current_string_features = current_feature.split(',')
                        list_of_current_string_features = [current_feature]
                        studio_dict = {}
                        for current_string_feature in list_of_current_string_features:
                            current_string_feature = re.sub('[^a-zA-Z\d]', '', current_string_feature)
                            current_string_feature = re.sub(' ', '', current_string_feature)
                            current_string_feature = current_string_feature.strip().lower()

                            if current_string_feature in self.feature_dict_objects["studio_mapper"]:
                                current_string_feature = self.feature_dict_objects["studio_mapper"][current_string_feature]
                                studio_dict[current_string_feature] = self.feature_dict_objects["studio"][current_string_feature]
                            else:
                                print("Unknown studio name in {}, {}".format(idx_1, current_string_feature))
                                studio_dict["others"] = self.feature_dict_objects["studio"]["others"]

                        self.dict_of_one_hot_encodable_features["studio"].append(studio_dict)

            self.num_samples += 1
        print("Total movies: {}".format(idx_1+1))
        print("Total movies with both scores available: {}".format(self.num_samples))

    def add_string_feature_to_dict_object(self, list_of_string_features, string_feature_dict):

        string_counter_value_dict = {}

        for string_feature in list_of_string_features:
            string_feature = re.sub('[^a-zA-Z\d]', '', string_feature)
            string_feature = re.sub(' ', '', string_feature)
            string_feature = string_feature.strip().lower()

            if string_feature not in string_feature_dict:
                string_counter_value_dict["none"] = string_feature_dict["none"]
            else:
                string_counter_value_dict[string_feature] = string_feature_dict[string_feature]

        return string_counter_value_dict

    def create_features_from_numeric_values(self, numeric_feature_string):

        try:
            numeric_feature_string = numeric_feature_string.strip()
            numeric_feature_string = numeric_feature_string.split()[0]
            numeric_feature_string = numeric_feature_string.split('%')[0]
            numeric_feature_string = float(numeric_feature_string)
            return numeric_feature_string

        except:
            return None

    def vectorize_string_features(self):

        for field_name_str, list_of_feature_dicts in self.dict_of_one_hot_encodable_features.items():
            print("Vectorizing {}...".format(field_name_str))

            if self.mode == "train":
                DV = DictVectorizer()
                feature_array_matrix = DV.fit_transform(list_of_feature_dicts)
                pickle.dump(DV, open("../data/models/vector_{}.vec".format(field_name_str), 'wb'))
                #print(DV.get_feature_names())

            elif self.mode == "test":
                with open("../data/models/vector_{}.vec".format(field_name_str), 'rb') as f:
                    DV = pickle.load(f)
                    feature_array_matrix = DV.transform(list_of_feature_dicts)
                    #print(DV.get_feature_names())
            self.dict_of_one_hot_encoded_matrix[field_name_str] = feature_array_matrix

        print("Done vectorizing. ")

    def merge_features_arrays(self):
        all_features_matrix = None

        for idx, column_name in enumerate(self.formatted_header):

            if column_name in self.dict_of_score_based_features:
                feature_matrix = coo_matrix(np.asarray(self.dict_of_score_based_features[column_name]))

            elif column_name in self.dict_of_one_hot_encodable_features:
                feature_matrix = self.dict_of_one_hot_encoded_matrix[column_name]

            else:
                continue

            if not feature_matrix.shape[0] == self.num_samples:
                feature_matrix = feature_matrix.T

            if all_features_matrix is None:
                all_features_matrix = feature_matrix
            else:
                if not feature_matrix.shape[0] == self.num_samples:
                    feature_matrix = feature_matrix.T

                assert feature_matrix.shape[0] == self.num_samples
                all_features_matrix = hstack([all_features_matrix, feature_matrix])

        feature_matrix = self.dict_of_one_hot_encoded_matrix["release_type"]
        if not feature_matrix.shape[0] == self.num_samples:
            feature_matrix = feature_matrix.T

        all_features_matrix = hstack([all_features_matrix, feature_matrix])

        return all_features_matrix

    def save_npz_data(self):
        print("\nSaving the npz arrays data...")
        save_npz("../data/npz_arrays/X.npz", coo_matrix(self.x_features_matrix, dtype=np.float64))
        save_npz("../data/npz_arrays/y.npz", coo_matrix(self.y_column_matrix, dtype=np.float64))


if __name__ == "__main__":
    print("\n\nStarting the data cleaning process...\n\n")
    DataWrangler("../data/training_data/11713_movies_raw_data_prof_format.txt")
