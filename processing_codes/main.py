import argparse as ARG
import sys

sys.path.append("../")

from wrangle_data import DataWrangler
from build_model import BuildAndTrainModel
from predict_difference import PredictDifference
import time as T

if __name__ == "__main__":

    parser = ARG.ArgumentParser()

    parser.add_argument("-i", "--input-text-file", default="../data/training_data/11713_movies_raw_data_prof_format.txt",
                        help="Input text file containing the details of the movies")
    parser.add_argument("-m", "--mode", type=str, choices=["train", "test"], default="train",
                        help="The mode in which this script has to run. Options are \"train\" and \"test\". By default "
                             "it is in train mode")
    parser.add_argument("--skip-cleaning", action="store_true", help="If provided will skip the data cleaning process,"
                                                                     "and use the pre-built feature arrays.")
    args = parser.parse_args()
    input_text_file = args.input_text_file
    mode = args.mode
    skip_cleaning = args.skip_cleaning

    start = T.clock()

    if mode == "train":
        print("\n\n~~~~~~~~~~~~~~~~~~Starting the data wrangling process in {} mode~~~~~~~~~~~~~~~~~~\n\n".format(mode))
        DataWrangler(input_text_file, mode="train")

        print("\n\n~~~~~~~~~~~~~~~~~~Building and training the model~~~~~~~~~~~~~~~~~~\n\n""")
        BuildAndTrainModel()

        print("\n\n~~~~~~~~~~~~~~~~~~Predicting on the segregated test data set~~~~~~~~~~~~~~~~~~\n\n")
        PredictDifference()

    elif mode == "test":
        print("\n\n~~~~~~~~~~~~~~~~~~Starting the data wrangling process in {} mode~~~~~~~~~~~~~~~~~~\n\n".format(mode))
        DataWrangler(input_text_file, mode="test")

        print("\n\n~~~~~~~~~~~~~~~~~~Predicting on {}~~~~~~~~~~~~~~~~~~\n\n".format(input_text_file))
        PredictDifference()

    else:
        print("\n\nInvalid mode. Options are \"train\" and \"test\".\n\n")

    print("\n\n~~~~~~~~~~~~~~~~~~All processes completed~~~~~~~~~~~~~~~~~~\n\n")
    print("End to End total time taken: {}\n\n".format(T.clock()-start))
    print("\n\n=============== That's all Folks! ===============\n\n")



