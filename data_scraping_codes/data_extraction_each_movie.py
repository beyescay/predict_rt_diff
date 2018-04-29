# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 06:04:07 2018

@author: Shivakumar Barathi
"""
from bs4 import BeautifulSoup
import re
import time
import requests
import os
import shutil as SH

def movie_info(movie, soup):
    movie_details={}
    movie_details['1_Movie:'] = movie
    movie_info_tags=soup.findAll('li',{'class':re.compile('meta-row')})
    
    if movie_info_tags:
        for each_tag in movie_info_tags:
            key=each_tag.find('div',{'class':'meta-label subtle'}).text.strip().replace(' ','')
            value=each_tag.find('div',{'class':'meta-value'}).text.strip()
        
            movie_details[key] = value
    
    
    return movie_details

def movie_rating_info(soup):
    
    critic_score = 'N/A' 
    critic_avg_rating = 'N/A' 
    audience_score = 'N/A' 
    audience_avg_rating = 'N/A'
    
    movie_rating_info_tags=soup.find('div',{'id':'scorePanel'})
    if movie_rating_info_tags:
        critic_score_meter_tag=movie_rating_info_tags.find('div',{'class':'critic-score meter'})
        if critic_score_meter_tag:
            critic_score_tag = critic_score_meter_tag.find('span',{'class':re.compile('meter-value')})
            if critic_score_tag:
                critic_score = critic_score_tag.text.strip()
    
    critic_ratings_tag = movie_rating_info_tags.find('div',{'id':'scoreStats'})
    if critic_ratings_tag:
        critic_avg_ratings_tag = critic_ratings_tag.find('div',{'class':'superPageFontColor'})
        if critic_avg_ratings_tag:
            critic_avg_rating = critic_avg_ratings_tag.text.strip()
            critic_avg_rating = critic_avg_rating[critic_avg_rating.find(':')+1:].strip()
    
    audience_score_meter_tag=movie_rating_info_tags.find('div',{'class':'audience-score meter'})
    if audience_score_meter_tag:
        audience_score_tag = audience_score_meter_tag.find('div',{'class':re.compile('meter-value')})
        if audience_score_tag:
            audience_score = audience_score_tag.text.strip()
    
    audience_ratings_tag=movie_rating_info_tags.find('div',{'class':re.compile('audience-info')})
    if audience_ratings_tag:
        audience_avg_rating_tag=audience_ratings_tag.find('div')
        if audience_avg_rating_tag:
            audience_avg_rating = audience_avg_rating_tag.text.strip()
            audience_avg_rating = audience_avg_rating[audience_avg_rating.find(':')+1:].strip()

    return critic_score, critic_avg_rating, audience_score, audience_avg_rating



def cast_info(soup):
    cast_details_list=[]
    cast_details='N/A'
    
    cast_details_tags=soup.findAll('div',{'class':re.compile('cast-item')})
    
    if cast_details_tags:
        for each_tag in cast_details_tags:
            cast_name_tag=each_tag.find('span')
            cast_details_list.append(cast_name_tag.text.strip())
    
        cast_details = '|'.join(cast_details_list)
    
    return cast_details

def critic_review_count(soup):
    fresh_count = 0
    rotten_count = 0
    
    critic_review_tag = soup.find('p',{'id':'criticHeaders'})
    
    if critic_review_tag:
        critic_review_fresh=critic_review_tag.find('a',{'href':re.compile('fresh')})
        critic_review_rotten=critic_review_tag.find('a',{'href':re.compile('rotten')})
        if critic_review_fresh:
            fresh_start_pos = critic_review_fresh.text.find('(')
            fresh_finish_pos = critic_review_fresh.text.find(')')
            fresh_count = critic_review_fresh.text[fresh_start_pos+1:fresh_finish_pos]
        if critic_review_rotten:
            rotten_start_pos = critic_review_rotten.text.find('(')
            rotten_finish_pos = critic_review_rotten.text.find(')')
            rotten_count = critic_review_rotten.text[rotten_start_pos+1:rotten_finish_pos]
    
    
    
    return fresh_count, rotten_count   

    
def run(movie_url):
    movie_url = movie_url.strip()
    movie_name = movie_url[:movie_url.find('::')]
    
    if movie_name.find('(') > -1:
        movie_name = movie_name[:movie_name.find('(')-1].strip()
    
    if movie_name.find('\\') > -1:
        movie_name = movie_name[:movie_name.find('\\')-1].strip()
    
    if movie_name.find(':') > -1:
        movie_name = movie_name.replace(':','')
        
    mov_url = movie_url[movie_url.find('::')+2:]
    
    movie_text_file=movie_name+'.txt'

#    url='https://www.rottentomatoes.com/m/'+movie+'/'
    
      
    fw=open(movie_text_file,'w')
    
    for i in range(5): 
            try:
                #use the browser to access the url
                response=requests.get(mov_url,headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
                html=response.content # get the html
                break # we got the file, break the loop
            except Exception as e:# browser.open() threw an exception, the attempt to get the response failed
                print ('failed attempt',i)
                time.sleep(2) # wait 2 secs
            
    if html: print("Success: Retrieved the main html")    
    
    soup = BeautifulSoup(html.decode('ascii', 'ignore'),'lxml') # parse the html 
    
    movie_details = movie_info(movie_name, soup)    
    critic_score, critic_avg_rating, audience_score, audience_avg_rating = movie_rating_info(soup)
    (critic_fresh_count, critic_rotten_count) = critic_review_count(soup)
    movie_details['CriticScore:'] = critic_score
    movie_details['CriticAverageRating:'] = critic_avg_rating
    movie_details['CriticPositiveReviewsCount:'] = critic_fresh_count
    movie_details['CriticNegativeReviewsCount:'] = critic_rotten_count
    movie_details['AudienceScore:'] = audience_score
    movie_details['AudienceAverageRating:'] = audience_avg_rating
    movie_details['Cast:'] = cast_info(soup)
    
    for key,value in movie_details.items():
        if key=='Genre:': value = movie_details[key].replace(' ','').replace('\n','')
        if key=='InTheaters:': value = movie_details[key].replace(' ','').replace('\n','')
        fw.write(str(key)+'\t'+str(value)+'\n')
        
    
    fw.close()
    
    SH.move(movie_text_file, os.path.join(os.getcwd(), "movie_details"))
    


if __name__ == '__main__':
    if not (os.path.exists(os.path.join(os.getcwd(), "movie_details"))):
        os.makedirs(os.path.join(os.getcwd(), "movie_details"))
    movie_url_file_name = os.path.join(os.getcwd(), "movies_url.txt") 
    movies_url_list=[]
    movie_url_names = open(movie_url_file_name, 'r')
    
    for movie_name_url in movie_url_names:
        movie_name_url_details = movie_name_url.split('\t')
        n_url = movie_name_url_details[0].strip() + '::' + movie_name_url_details[1].strip()
        movies_url_list.append(n_url.strip())
    
    movie_url_names.close()
    
    movies = movies_url_list[:1000]
    
#    movies =['mad_max_fury_road','inside_out_2015','the_cabinet_of_dr_caligari','the_wizard_of_oz_1939','get_out','i_am_not_your_negro','citizen_kane','psycho','et_the_extraterrestrial','singin_in_the_rain','the_third_man','it_happened_one_night','wonder_woman_2017','bride_of_frankenstein','murderball','life_itself','treasure_of_the_sierra_madre']
    for movie_url in movies:        
        run(movie_url)
        
