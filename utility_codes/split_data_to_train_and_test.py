from sklearn.model_selection import train_test_split
import time as T
import os

import sys

sys.path.append("../")
sys.path.append("../data")

class TrainTestSplitter:

    def __init__(self, movie_info_txt_file):
        self.movie_info_txt_file = movie_info_txt_file

        print("Splitting the data into training and testing data...")
        start = T.clock()
        self.split_data_into_training_and_testing()
        print("Done. Time taken: {}\n\n".format(T.clock()-start))

    def split_data_into_training_and_testing(self, test_size=0.20):

        with open(self.movie_info_txt_file, mode="r") as infile:
            infile_lines = infile.readlines()
            total_lines = len(infile_lines)

        list_of_row_indices = range(1, total_lines)

        train_indices, test_indices, _, _ = train_test_split(list_of_row_indices, list_of_row_indices, test_size=test_size, random_state=42)

        print("\nSaving the training data...")
        with open("../data/training_data/{}_movies_raw_data_prof_format.txt".format(len(train_indices)), mode='w+') as train_outfile:
            train_outfile.write(infile_lines[0])

            for line_num in train_indices:
                train_outfile.write(infile_lines[line_num])

        print("\nSaving the testing data...")
        with open("../data/test_data/{}_movies_raw_data_prof_format.txt".format(len(test_indices)), mode='w+') as test_outfile:
            test_outfile.write(infile_lines[0])

            for line_num in test_indices:
                test_outfile.write(infile_lines[line_num])


if __name__ == "__main__":

    TrainTestSplitter("../data/14642_movies_raw_data_prof_format.txt")
