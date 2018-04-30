# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:33:58 2018

@author: Shivani
"""
import csv
import re
from operator import itemgetter
import io
from difflib import SequenceMatcher

               
class StudioConversion:
    
    def __init__(self, movie_raw_file):
        self.all_movies_raw_file = movie_raw_file
        self.freq_studio_terms = ("pictures","films","entertainment","home","video","media","releasing","film","classics","classic","new"
                                  ,"features","cinema","productions","first","the","ltd","ltd.","films","films/","llc","llc.","company","studios"
                                  ,"international","independent","distribution","group","limited","corporation","factory","animation")
        self.studio_name_modification_dict = {}
        self.studio_string_mapping_dict = {}
        self.unique_studio_modified_values = set()
        self.studio_string_mapping_dict = {}
        self.studio_modified_name_counter_dict = {}
        
        self.remove_freq_name_from_studios()
        
        self.create_unique_studio_modified_values()
        
        self.studio_string_mapping_dict = self.string_matcher(limitControl=0.7)
        
        self.alter_originial_studio_value()
        
        #second iteration
        self.unique_studio_modified_values = set()
        self.create_unique_studio_modified_values()
        self.studio_string_mapping_dict = {}
        self.studio_string_mapping_dict = self.string_matcher(limitControl=0.6)
        self.alter_originial_studio_value()
        
        self.set_to_others()

        self.augment_studio_names_with_counter()
      #  self.write_to_output_file()
 
        print("\nCompleted the studio conversion process...\n")
    
    def get_studio_name_converter_dict(self):
        return self.studio_name_modification_dict, self.studio_modified_name_counter_dict

    def augment_studio_names_with_counter(self):
        counter = 0
        for studio_name in self.studio_name_modification_dict:
            studio_name_mapped_name = self.studio_name_modification_dict[studio_name]

            if studio_name_mapped_name not in self.studio_modified_name_counter_dict:
                self.studio_modified_name_counter_dict[studio_name_mapped_name] = str(counter)
                counter += 1

        self.studio_modified_name_counter_dict["others"] = str(counter)

    def remove_freq_name_from_studios(self):
        
        with open(self.all_movies_raw_file,'r',encoding="utf8") as input:
            
            next(input)
    
            for line in csv.reader(input, dialect="excel-tab"):
                movie_info = line
                studio_name = movie_info[8]
                
                if studio_name.find('/') > -1:
                    studio_name_modified = studio_name[:studio_name.find('/')].lower()
                else:    
                    studio_name_modified = studio_name.lower()
                
                studio_name_parts = studio_name_modified.split()
                
                for part in studio_name_parts:
                    if part in self.freq_studio_terms:
                        studio_name_modified = studio_name_modified.replace(part,"").strip()
                        studio_name_modified = re.sub('[^a-zA-Z\d]','',studio_name_modified)
                    else:
                        studio_name_modified = re.sub('[^a-zA-Z\d]','',studio_name_modified)
                
                studio_name = re.sub('[^a-zA-Z\d]', '', studio_name)
                studio_name = re.sub(' ', '', studio_name)
                studio_name = studio_name.strip().lower()
            
                if studio_name not in self.studio_name_modification_dict.keys():
                    self.studio_name_modification_dict[studio_name] = studio_name_modified
        
        
    def create_unique_studio_modified_values(self):
         for key, value in self.studio_name_modification_dict.items():
            if value != "":
                self.unique_studio_modified_values.add(value)
         
    def string_matcher(self, limitControl):
        studio_corr_list_1 = list(self.unique_studio_modified_values)
        studio_corr_list_2 = list(self.unique_studio_modified_values)


        s = SequenceMatcher(None)

        limit = limitControl

        string_mapping_dict = {}

        for interest in studio_corr_list_1:
            s.set_seq2(interest)
            for keyword in studio_corr_list_2:
                s.set_seq1(keyword)
                
                b = s.ratio()>=limit and len(s.get_matching_blocks())==2
             
                if b and s.ratio() != 1.0:
                    #print(interest,keyword,s.ratio(),'** MATCH **' if b else '')
                    if len(interest) < len(keyword):
                        string_mapping_dict[keyword] = interest
                        
        return string_mapping_dict
    
    def alter_originial_studio_value(self):
        for key, value in self.studio_name_modification_dict.items():
            if value in self.studio_string_mapping_dict.keys():
                self.studio_name_modification_dict[key] = self.studio_string_mapping_dict[value]
                
    def set_to_others(self):
        studio_value_dict = {}
        
        for key, value in self.studio_name_modification_dict.items():
            if key == 'none':
                self.studio_name_modification_dict[key] = 'none'
            elif key == 'mediahomeentertainment':
                self.studio_name_modification_dict[key] = 'mediahomeentertainment'
            elif key == 'newfilmscinema' or key == 'newfilmsinternational':
                self.studio_name_modification_dict[key] = 'newfilms'
            elif key == 'newvideo' or key == 'newvideogroup':
                self.studio_name_modification_dict[key] = 'newvideogroup'
            elif key == 'firstindependentpictures':
                self.studio_name_modification_dict[key] = 'firstindependentpictures'
            elif key == 'independentpictures' or key == 'independentpicturesmetrodomedist' or key == 'independentfilms' or key == 'independent':
                self.studio_name_modification_dict[key] = 'independent'
            elif key == 'classicmedia' or key == 'cinemagroup' or key == 'distributioncompany' or key == 'cinemaproductions' or key == 'cinemafactory' or key == 'newcinema':
                self.studio_name_modification_dict[key] = 'others'
         
            if value == '20thcenturyfox':
                self.studio_name_modification_dict[key] = 'fox'
            elif value == 'foxsearchlight':
                self.studio_name_modification_dict[key] = 'fox'
            elif value == 'hbodocumentary':
                self.studio_name_modification_dict[key] = 'hbo'
            elif value == 'ifcmidnight':
                self.studio_name_modification_dict[key] = 'ifc'
            elif value == 'millenium':
                self.studio_name_modification_dict[key] = 'millennium'
                
       
        with open(self.all_movies_raw_file,'r',encoding="utf8") as input:
            next(input)
            for line in csv.reader(input, dialect="excel-tab"):
                movie_info = line
                studio_name = movie_info[8]
                
                studio_name = re.sub('[^a-zA-Z\d]', '', studio_name)
                studio_name = re.sub(' ', '', studio_name)
                studio_name = studio_name.strip().lower()
                
                for key, value in self.studio_name_modification_dict.items():
                    if key == studio_name:
                        if value not in studio_value_dict.keys():
                            studio_value_dict[value] = 1
                        else:
                            studio_value_dict[value] = studio_value_dict[value] + 1
                                           
        for key, value in self.studio_name_modification_dict.items():
            if studio_value_dict[value] < 10:
                self.studio_name_modification_dict[key] = 'others'
        
    def write_to_output_file(self):
        studios_conversion = ''
          
        for key, value in self.studio_name_modification_dict.items():
            
            studio_conversion = key + '\t'+ value + '\n'    
            studios_conversion = studios_conversion + studio_conversion
    
    
        with io.open('studio_conversion.txt', 'w',encoding="utf8") as file:
                file.write(studios_conversion)
        
        
        
if __name__ == "__main__":
    print("\nStarting the studio conversion process...\n")
    StudioConversion("14642_movies_raw_data_prof_format.txt")

