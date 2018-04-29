# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 01:44:38 2018

@author: Shivani
"""
import os

def movie_url_parser():

    movie_url_raw_file_name = os.path.join(os.getcwd(), "movie_urls_raw_file.txt")
    
    movies_url_raw_file = open(movie_url_raw_file_name,'r')
    
    movies_url_file = open('movies_url.txt','w')
    
    for raw_line in movies_url_raw_file:
        movie_name = raw_line[raw_line.find('"')+1:raw_line.find('": "https://')].strip() 
        movie_url = raw_line[raw_line.find('https://'):raw_line.find('",')].strip()
        if movie_url.find('null') == -1:
            movies_url_file.write(movie_name+'\t'+movie_url+'\n')
    
    movies_url_raw_file.close()
    movies_url_file.close()

if __name__ == "__main__":
    movie_url_parser()