# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:44:50 2018

@author: Shivani
"""

import os, sys, glob
import shutil as SH
import time

           
def data_consolidation():    
    
    path_to_movies_dir = os.path.join(os.getcwd(), "movie_details")
    
    if not (os.path.exists(os.path.join(os.getcwd(), "combined_movies_data"))):
        os.makedirs(os.path.join(os.getcwd(), "combined_movies_data"))
    
    combined_movies_data = os.path.join(os.getcwd(), "combined_movies_data")

    if not (os.path.isfile(os.path.join(combined_movies_data, 'combined_movies_file.txt'))):
        movie_directory = glob.glob(os.path.join(path_to_movies_dir, "*"))

        with open('combined_movies_file.txt', 'wb') as wfd:
            for data_file in movie_directory:
                with open(data_file,'rb') as fd:
                    SH.copyfileobj(fd,wfd)

        SH.move('combined_movies_file.txt', combined_movies_data)
    
    consolidated_movies_file = glob.glob(os.path.join(combined_movies_data, 'combined_movies_file.txt'))

    return consolidated_movies_file[0]

def extract_unique_cols(movies_file):
    
    if os.path.isfile(os.path.join(os.getcwd(), "combined_movies_data","column_list.txt")):
        os.remove(os.path.join(os.getcwd(), "combined_movies_data","column_list.txt"))
        
    column_set = set()
    fin=open(movies_file)
    time.sleep(15)
    for each_line in fin:
        colon_pos = each_line.find(':')
        column_set.add(each_line[:colon_pos].strip())
    
    col_list_file = open('column_list.txt', 'w')
    
    for each_col in  column_set:
        col_list_file.write(each_col+'\n')
    
    col_list_file.close()
    fin.close()
    
    SH.move('column_list.txt', os.path.join(os.getcwd(), "combined_movies_data"))
   
def extract_unique_censor_ratings(movies_file):
    
    if os.path.isfile(os.path.join(os.getcwd(), "combined_movies_data","censor_rating_list.txt")):
        os.remove(os.path.join(os.getcwd(), "combined_movies_data","censor_rating_list.txt"))
        
    censor_rating_set = set()
    
    fin=open(movies_file)
    time.sleep(15)
    
    for each_line in fin:
        if (each_line[:(each_line.find(':'))]) == "Rating":
            colon_pos = each_line.find(':')
            entire_cr_line = each_line[colon_pos+1:].strip()
            credit_rating = entire_cr_line
            if entire_cr_line.find('(') > 0:
                credit_rating = entire_cr_line[:entire_cr_line.find('(')].strip()
            
            censor_rating_set.add(credit_rating)
    
    censor_rating_list_file = open('censor_rating_list.txt', 'w')
    
    for each_col in  censor_rating_set:
        censor_rating_list_file.write('Rating_'+each_col+'\n')
    
    censor_rating_list_file.close()
    
    fin.close()
    
    SH.move('censor_rating_list.txt', os.path.join(os.getcwd(), "combined_movies_data"))
    
def extract_unique_genre(movies_file):  
  
    if os.path.isfile(os.path.join(os.getcwd(), "combined_movies_data","genre_list.txt")):
        os.remove(os.path.join(os.getcwd(), "combined_movies_data","genre_list.txt"))
        
    genre_set = set()
    
    fin=open(movies_file)
    time.sleep(15)
    
    for each_line in fin:
        if (each_line[:(each_line.find(':'))]) == "Genre":
            colon_pos = each_line.find(':')
            entire_genre_line = each_line[colon_pos+1:].strip().split(",")
            
            for each_genre in entire_genre_line:
                each_genre = each_genre.strip()
                genre_set.add(each_genre)
                
    genre_list_file = open('genre_list.txt', 'w')
    
    for each_col in  genre_set:
        genre_list_file.write('Genre_'+each_col+'\n')
    
    genre_list_file.close()
    
    fin.close()
    
    SH.move('genre_list.txt', os.path.join(os.getcwd(), "combined_movies_data"))

def entire_attribute_list ():
    
    if os.path.isfile(os.path.join(os.getcwd(), "combined_movies_data","entire_attribute_list.txt")):
        os.remove(os.path.join(os.getcwd(), "combined_movies_data","entire_attribute_list.txt"))
    
    entire_attribute_file = os.path.join(os.getcwd(), "combined_movies_data","entire_attribute_list.txt")
    column_list_file = os.path.join(os.getcwd(), "combined_movies_data","column_list.txt")
    censore_rating_list_file = os.path.join(os.getcwd(), "combined_movies_data","censor_rating_list.txt")
    genre_list_file = os.path.join(os.getcwd(), "combined_movies_data","genre_list.txt")
    attribute_file_list = [column_list_file,censore_rating_list_file,genre_list_file]
    
    with open(entire_attribute_file, 'wb') as wfd:
        for col_file in attribute_file_list:
                with open(col_file,'rb') as fd:
                    SH.copyfileobj(fd,wfd)
    

if __name__ == "__main__":
    consolidated_movies_file = data_consolidation()
    extract_unique_cols(consolidated_movies_file)
    extract_unique_censor_ratings(consolidated_movies_file)
    extract_unique_genre(consolidated_movies_file)
    entire_attribute_list()
    
  