# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 06:04:07 2018

@author: Shivakumar Barathi
"""
from bs4 import BeautifulSoup
from datetime import datetime
import re
import time
import requests
import os
import shutil as SH
import io

class ExtractMovieDetails:
    
    def __init__(self, movie_url_details_txt_file):
        self.txt_file = movie_url_details_txt_file
        self.movie_url_list = []
        self.iter = 0
        self.column_names = ["movie_id", "audience_score","critic_score","actor_names","actor_links",	
                             "synopsis","In Theaters","Genre","Studio","Directed By","Runtime","Box Office",	
                             "Rating","Written By"]
        
        self.movie_movie_details = {}

        self.get_movie_name_url()        
        self.get_movie_details()         
        self.write_movie_details_to_txt()
        
    def get_movie_name_url(self):
        if not (os.path.exists(os.path.join(os.getcwd(), "movie_details"))):
            os.makedirs(os.path.join(os.getcwd(), "movie_details"))
        
        movie_url_details = open(self.txt_file, 'r')
        
        for movie_name_url in movie_url_details:
            movie_name_url_details = movie_name_url.split('\t')
            movie_url = movie_name_url_details[1].strip()
            
            self.movie_url_list.append(movie_url.strip())
        
        movie_url_details.close()
        
    def get_movie_details(self):
        for movie_url in self.movie_url_list:
            self.movie_details = {key: 'NONE' for key in self.column_names}
            movie_name = movie_url[movie_url.find('/m/')+3:].strip().lower()
#            movie_name=re.sub('[^a-zA-Z\d]','',movie_name)
#            movie_name=re.sub(' +',' ',movie_name).strip().lower()
             
            self.movie_details["movie_id"] = movie_name
            movie_url = movie_url.strip()
            
            for i in range(5): 
                try:
                    #use the browser to access the url
                    response=requests.get(movie_url,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                    html=response.content # get the html
                    break # we got the file, break the loop
                except Exception as e:# browser.open() threw an exception, the attempt to get the response failed
                    print ('failed attempt',i)
                    time.sleep(2) # wait 2 secs
            
            if html:
                self.iter += 1
                print("Success: Retrieved the main html for movie: " + str(self.iter))
                
            self.soup = BeautifulSoup(html.decode('ascii', 'ignore'),'lxml') # parse the html
            
            self.get_audience_critic_score()            
            self.get_cast_details()            
            self.get_synopsis()            
            self.get_other_details()
            
            self.movie_movie_details[movie_name] = self.movie_details
    
    def get_audience_critic_score(self):  
        audience_score = 'NONE'
        critic_score = 'NONE'
        movie_rating_info_tags=self.soup.find('div',{'id':'scorePanel'})
        if movie_rating_info_tags:
            audience_score_meter_tag=movie_rating_info_tags.find('div',{'class':'audience-score meter'})
        
            if audience_score_meter_tag:
                audience_score_tag = audience_score_meter_tag.find('div',{'class':re.compile('meter-value')})
                if audience_score_tag:
                    audience_score = audience_score_tag.text.strip()        
        
        if movie_rating_info_tags:
            critic_score_meter_tag=movie_rating_info_tags.find('div',{'class':'critic-score meter'})
            if critic_score_meter_tag:
                critic_score_tag = critic_score_meter_tag.find('span',{'class':re.compile('meter-value')})
                if critic_score_tag:
                    critic_score = critic_score_tag.text.strip()
                   
        self.movie_details["audience_score"] = audience_score
        self.movie_details["critic_score"] = critic_score
                
    def get_cast_details(self):
        cast_details_list=[]
        cast_url_list=[]
         
        cast_details_tags=self.soup.findAll('div',{'class':re.compile('cast-item')})
         
        if cast_details_tags:
            for each_tag in cast_details_tags:
                cast_name_tag=each_tag.find('span')
                if cast_name_tag:
                    cast_details_list.append(cast_name_tag.text.strip())
                
                cast_url_tag=each_tag.find('a',{'href':re.compile('/celebrity/')})
                if cast_url_tag:
                    cast_url_list.append(cast_url_tag.get('href').strip())
        
        if cast_details_list:
            self.movie_details["actor_names"] = ','.join(cast_details_list) 
        else: self.movie_details["actor_names"] = 'NONE'
        
        if cast_url_list:
            self.movie_details["actor_links"] = ','.join(cast_url_list) 
        else: self.movie_details["actor_links"] = 'NONE'

    def get_synopsis(self):
        synopsis = 'NONE'
        synopsis_tag = self.soup.find('div',{'id':'movieSynopsis'})
        
        if synopsis_tag:
            synopsis = synopsis_tag.text.strip()
        
        self.movie_details["synopsis"] = synopsis
            
    def get_other_details(self):
        movie_info_tags=self.soup.findAll('li',{'class':re.compile('meta-row')})
    
        if movie_info_tags:
            for each_tag in movie_info_tags:
                key=each_tag.find('div',{'class':'meta-label subtle'}).text.strip()
                key=key[:key.find(':')].strip()
                value=each_tag.find('div',{'class':'meta-value'}).text.strip().replace('\n','')
                
                if key == 'Genre':
                    value = value.replace(' ','')
                if key in self.movie_details.keys():
                    self.movie_details[key] = value
            
    def write_movie_details_to_txt(self):
        movie_lines = "movie_id"+'\t'+"audience_score"+'\t'+"critic_score"+'\t'+"actor_names" \
                            +'\t'+"actor_links"+'\t'+"synopsis"+'\t'+"In Theaters" \
                            +'\t'+"Genre"+'\t'+"Studio"+'\t'+"Directed By" \
                            +'\t'+"Runtime"+'\t'+"Box Office"+'\t'+"Rating" \
                            +'\t'+"Written By"+'\n'
        for key, value in self.movie_movie_details.items():
            
            for inner_key, inner_value in value.items():
                if inner_key == 'movie_id':
                    movieName = inner_value
                if inner_key == 'audience_score':
                    audienceScore = inner_value
                if inner_key == 'critic_score':
                    criticScore = inner_value
                if inner_key == 'actor_names':
                    actorNames = inner_value
                if inner_key == 'actor_links':
                    actorLinks = inner_value
                if inner_key == 'synopsis':
                    movieSynopsis = inner_value
                if inner_key == 'In Theaters':
                    inTheatres = inner_value
                if inner_key == 'Genre':
                    movieGenre = inner_value
                if inner_key == 'Studio':
                    movieStudio = inner_value
                if inner_key == 'Directed By':
                    directedBy = inner_value
                if inner_key == 'Runtime':
                    movieRuntime = inner_value
                if inner_key == 'Box Office':
                    movieBoxoffice = inner_value
                if inner_key == 'Rating':
                    movieRating = inner_value
                if inner_key == 'Written By':
                    movieWrittenby = inner_value
                    
            movie_line = movieName+'\t'+audienceScore+'\t'+criticScore+'\t'+actorNames \
                            +'\t'+actorLinks+'\t'+movieSynopsis+'\t'+inTheatres \
                            +'\t'+movieGenre+'\t'+movieStudio+'\t'+directedBy \
                            +'\t'+movieRuntime+'\t'+movieBoxoffice+'\t'+movieRating \
                            +'\t'+movieWrittenby+'\n'
            
            
            movie_lines += movie_line
  
            with io.open('rottenTomatoes_raw_data.txt', 'w') as file:
                file.write(movie_lines)
                
 

if __name__ == "__main__":
    print ("\n\nStarting the extraction of movie details... \n\n")
    movie_name_url_details = os.path.join(os.getcwd(), "movies_url_5000_04_27.txt")
    ExtractMovieDetails(movie_name_url_details)
    

        
