import re
import datetime as DT


class GenreRatingRuntimeIntheatersConverter:

    def __init__(self, list_of_movies, feature_dict_objects):
        self.list_of_movies = list_of_movies
        self.feature_dict_objects = feature_dict_objects
        self.delimiters_for_string_features = re.compile(",")
        self.convert()
        self.append_others_string_to_all_dicts()

    def get_feature_name_converter_dict(self):
        return self.feature_dict_objects

    def convert(self):
        for idx_1, movie in enumerate(self.list_of_movies):

            for idx_2, field_name in enumerate(movie._fields):

                field_name_str = str(field_name)

                if field_name_str in self.feature_dict_objects:
                    current_feature = getattr(movie, field_name)
                    feature_dict_object = self.feature_dict_objects[str(field_name)]

                    if field_name_str == "genre":
                        list_of_current_string_features = re.split(self.delimiters_for_string_features, current_feature)
                        self.add_string_feature_to_dict_object(list_of_current_string_features, feature_dict_object)

                    elif field_name_str == "rating":
                        string_feature = current_feature.split('(')[0]
                        list_of_current_string_features = [string_feature]
                        self.add_string_feature_to_dict_object(list_of_current_string_features, feature_dict_object)

                    elif field_name_str == "runtime":
                        current_feature = current_feature.strip().lower()
                        current_feature = current_feature.split()[0]
                        try:
                            current_feature = int(current_feature)
                            current_feature = int(current_feature/15)

                        except:
                            if not current_feature == "none":
                                print("Some unknown value in runtime: {}".format(current_feature))

                        self.add_binnable_feature_to_dict_object(current_feature, feature_dict_object)

                    elif field_name_str == "intheaters":
                        current_feature = current_feature.strip().lower()

                        if current_feature.find("wide") != -1:
                            current_feature = current_feature[0:current_feature.find("wide")-1]
                        elif current_feature.find("limited") != -1:
                            current_feature = current_feature[0:current_feature.find("limited")-1]

                        try:
                            dt = DT.datetime.strptime(str(current_feature), "%b %d, %Y")
                            year_num = dt.year
                            current_feature = int(year_num/10)
                        except:
                            if not current_feature == "none":
                                print("Some unknown value in intheaters: {}".format(current_feature))

                        self.add_binnable_feature_to_dict_object(current_feature, feature_dict_object)

    def append_others_string_to_all_dicts(self):

        for field_name_str, dict_object in self.feature_dict_objects.items():
            counter = len(dict_object)
            self.feature_dict_objects[field_name_str]["others"] = str(counter)

    def add_string_feature_to_dict_object(self, list_of_string_features, string_feature_dict):

        string_counter = len(string_feature_dict)

        for string_feature in list_of_string_features:
            string_feature = re.sub('[^a-zA-Z\d]', '', string_feature)
            string_feature = re.sub(' ', '', string_feature)
            string_feature = string_feature.strip().lower()

            if string_feature not in string_feature_dict:
                string_feature_dict[string_feature] = str(string_counter)
                string_counter += 1

    def add_binnable_feature_to_dict_object(self, current_feature, feature_dict):

        counter = len(feature_dict)

        if current_feature not in feature_dict:
            feature_dict[current_feature] = str(counter)
            counter += 1