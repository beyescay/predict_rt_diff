
from collections import namedtuple
import pickle
import sys

sys.path.append("../")
from utility_codes.studio_name_conversion import StudioConversion
from utility_codes.cast_directedby_writtenby_conversion import CastWrittenByDirectedByConversion
from utility_codes.genre_rating_runtime_intheaters_conversion import GenreRatingRuntimeIntheatersConverter


class DataCleaner:

    def __init__(self, movie_info_txt_file, num_actors=2):
        self.movie_info_txt_file = movie_info_txt_file
        self.list_of_movies = []
        self.header = None
        self.formatted_header = None
        self.num_actors = num_actors

        self.feature_dict_objects = {"genre": {},
                                     "rating": {},
                                     "runtime": {},
                                     "intheaters": {},
                                     "release_type": {"wide": "0", "limited": "1"},
                                    }

        self.create_list_of_movies()
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

    def create_feature_dict_objects(self):

        print("Creating dict object for studio...")
        studio_feature_converter = StudioConversion(self.movie_info_txt_file)
        studio_dict = studio_feature_converter.get_studio_name_converter_dict()
        print("Saving dict object for studio...")
        self.save_feature_dict_objects(studio_dict, "studio")

        print("Creating dict objects for actornames...")
        cast_feature_converter = CastWrittenByDirectedByConversion(self.movie_info_txt_file, "actornames", max_actors=2)
        cast_dict = cast_feature_converter.get_feature_name_converter_dict()
        print("Saving dict object for actornames...")
        self.save_feature_dict_objects(cast_dict, "actornames")

        print("Creating dict objects for directedby...")
        directedby_feature_converter = CastWrittenByDirectedByConversion(self.movie_info_txt_file, "directedby")
        directedby_dict = directedby_feature_converter.get_feature_name_converter_dict()
        print("Saving dict object for directedby...")
        self.save_feature_dict_objects(directedby_dict, "directedby")

        print("Creating dict objects for writtenby...")
        writtenby_feature_converter = CastWrittenByDirectedByConversion(self.movie_info_txt_file, "writtenby")
        writtenby_dict = writtenby_feature_converter.get_feature_name_converter_dict()
        print("Saving dict object for writtenby...")
        self.save_feature_dict_objects(writtenby_dict, "writtenby")
        
        print("Creating dicts objects for genre, rating, intheaters...")
        genre_rating_intheaters_converter = GenreRatingRuntimeIntheatersConverter(self.list_of_movies, self.feature_dict_objects)
        self.feature_dict_objects = genre_rating_intheaters_converter.get_feature_name_converter_dict()

        for dict_object_name in self.feature_dict_objects:
            print("Saving dict object for {}...".format(dict_object_name))
            self.save_feature_dict_objects(self.feature_dict_objects[dict_object_name], dict_object_name)

    @staticmethod
    def save_feature_dict_objects(obj, name):
        with open('../data/dictionary_objects/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    print("\n\nStarting the data cleaning process...\n\n")
    DataCleaner("../data/14642_movies_raw_data_prof_format.txt")
