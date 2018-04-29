# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:11:42 2018

@author: Shivani
"""
import re
import statistics
import io
import csv

class CastWrittenByDirectedByConversion:
    
    def __init__(self, movie_raw_file, feature_name, max_actors=2):
        self.all_movies_raw_file = movie_raw_file
        self.feature_to_be_converted = feature_name
        self.max_actors_to_be_extracted = max_actors
        self.removal_set = set()
        self.removal_set.add('jr')
        self.removal_set.add('jr.')
        self.removal_set.add('sr')
        self.removal_set.add('sr.')
        self.unique_feature_value_set = set()
        self.feature_value_popularity_dict = {}
        self.total_feature_values_popularity_list = []
        self.feature_mean_value = 0
        self.feature_sd_value = 0
        
        if self.feature_to_be_converted == "actornames":
            self.feature_position = 3
        elif self.feature_to_be_converted == "directedby":
            self.feature_position = 9
        elif self.feature_to_be_converted == "writtenby":
            self.feature_position = 13
            
        self.get_unique_feature_value_popularity()
        
        self.get_total_feature_values_popularity_list()
                
        self.get_feature_mean_sd_value()
        
        self.assign_normalized_score()
        
        #self.write_to_csv()
                
    def get_feature_name_converter_dict(self):
        return self.feature_value_popularity_dict

    def get_unique_feature_value_popularity(self):
        feature_split_list = []
        feature_value_split_list = []

        with open(self.all_movies_raw_file,'r',encoding="utf8") as input:
            next(input)
            for line in csv.reader(input, dialect="excel-tab"):
                i=-1
                movie_info = line
                feature_value_list = movie_info[self.feature_position]
                feature_split_list = feature_value_list.split(',')

                for value in feature_split_list:
                    value = re.sub('[^a-zA-Z\d\.]','',value)
                    value = re.sub(' ','',value)
                    value = value.strip().lower()
                    if self.feature_to_be_converted != "actornames" and value not in self.removal_set:
                        feature_value_split_list.append(value)
                    if self.feature_to_be_converted == "actornames" and value not in self.removal_set:
                        i = i+1
                        if i < self.max_actors_to_be_extracted:
                            feature_value_split_list.append(value)
            
            for each_val in feature_value_split_list:
                        
                self.unique_feature_value_set.add(each_val)
                        
                if each_val != 'none' and each_val not in self.feature_value_popularity_dict.keys():
                    self.feature_value_popularity_dict[each_val] = 1
                elif each_val == 'none':
                    self.feature_value_popularity_dict[each_val] = 0
                else:
                    self.feature_value_popularity_dict[each_val] = self.feature_value_popularity_dict[each_val] + 1
                                   
    def get_total_feature_values_popularity_list(self):
        
        for key, value in self.feature_value_popularity_dict.items():
            self.total_feature_values_popularity_list.append(value) 
            


    def get_feature_mean_sd_value(self):
        
        self.feature_mean_value = statistics.mean(self.total_feature_values_popularity_list)
        self.feature_sd_value = statistics.pstdev(self.total_feature_values_popularity_list)
         
    def assign_normalized_score(self):
        for key,value in self.feature_value_popularity_dict.items():
            if key != 'none':
                self.feature_value_popularity_dict[key] = (value - self.feature_mean_value) / self.feature_sd_value
                
            else:
                self.feature_value_popularity_dict[key] = self.feature_mean_value
     
    def write_to_csv(self):
        feature_op_value = ''
        features_op_value = ''
        for key, value in self.feature_value_popularity_dict.items():
            feature_op_value = key + '\t' + str(value) + '\n'
            features_op_value = features_op_value + feature_op_value
            
        if self.feature_to_be_converted == "actornames":
            with io.open('cast_weighted_score.txt', 'w',encoding="utf8") as file:
                file.write(features_op_value)
        elif self.feature_to_be_converted == "directedby":
            with io.open('directed_by_weighted_score.txt', 'w',encoding="utf8") as file:
                file.write(features_op_value)
        elif self.feature_to_be_converted == "writtenby":
            with io.open('written_by_weighted_score.txt', 'w',encoding="utf8") as file:
                file.write(features_op_value)
        

if __name__ == "__main__":
    print("\n\n Starting the conversion process for cast/written by/directed by....")
    CastWrittenByDirectedByConversion("../data/14642_movies_raw_data_prof_format.txt", "actornames", 2)