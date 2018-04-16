# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:34:13 2018

@author: Shivani
"""
import os
import pandas
from pandas import set_option

def create_pandas_df():
    k=0
    path_to_consolidated_dir = os.path.join(os.getcwd(), "combined_movies_data")
    path_to_attribute_list = os.path.join(path_to_consolidated_dir, "entire_attribute_list.txt")
    
    att_file = open(path_to_attribute_list, 'r')
    att_list = []
    for line in att_file:
        line=line.strip()
        att_list.append(line)
    att_list = sorted(att_list)   
    
    num_of_att = len(att_list)    
    movies_df = pandas.DataFrame([range(num_of_att)])

    movies_df.columns = att_list
    
    att_file.close()
      
    path_to_movies_dir = os.path.join(os.getcwd(), "movie_details")
    path_to_file_col_list = os.path.join(path_to_consolidated_dir, "column_list.txt")
    
    movie_directory = glob.glob(os.path.join(path_to_movies_dir, "*"))
    col_vals = open(path_to_file_col_list, 'r')
    col_list = []
    
    for val in col_vals:
        col_list.append(val.strip())

    movie_att = dict()

    for movie in movie_directory:
        movie_file = open(movie, 'r')
        Movie = 'N/A'
        AudienceAverageRating = 'N/A'
        AudienceScore = 'N/A'
        BoxOffice = 'N/A'
        Cast = 'N/A'
        CriticAverageRating = 'N/A'
        CriticNegativeReviewsCount = 'N/A'
        CriticPositiveReviewsCount = 'N/A'
        CriticScore = 'N/A'
        DirectedBy = 'N/A'
        Genre = 'N/A'
        InTheaters = 'N/A'
        OnDisc_Streaming = 'N/A'        
        Rating = 'N/A'
        Runtime = 'N/A'
        Studio = 'N/A'
        WrittenBy = 'N/A'
        Genre_Action_Adventure = 0
        Genre_Animation = 0
        Genre_Anime_Manga = 0
        Genre_ArtHouse_International = 0
        Genre_Classics = 0
        Genre_Comedy = 0
        Genre_CultMovies = 0
        Genre_Documentary = 0      
        Genre_Drama = 0
        Genre_Gay_Lesbian = 0
        Genre_Horror = 0
        Genre_Kids_Family = 0
        Genre_Musical_PerformingArts = 0
        Genre_Mystery_Suspense = 0
        Genre_Romance = 0
        Genre_ScienceFiction_Fantasy = 0
        Genre_SpecialInterest = 0
        Genre_Sports_Fitness = 0
        Genre_Television = 0
        Genre_Western = 0
        Rating_G = 0 
        Rating_NC17 = 0
        Rating_NR = 0 
        Rating_PG = 0 
        Rating_PG_13 = 0 
        Rating_R = 0
            
        for line in movie_file:
            line_col_name = line[:line.find(':')].strip()
            line_col_value = line[line.find(':')+1:].strip()
            for col in col_list:
                col_name = col.strip()                
                if col_name == line_col_name: 
                    movie_att[col_name] = line_col_value        
     
        for key, value in movie_att.items():

        
            if key == '1_Movie': Movie = value
            if key == 'AudienceAverageRating': AudienceAverageRating = value[:value.find('/')] 
            if key == 'AudienceScore': AudienceScore = value
            if key == 'BoxOffice': BoxOffice = value
            if key == 'Cast': Cast = value
            if key == 'CriticAverageRating': CriticAverageRating = value[:value.find('/')] 
            if key == 'CriticNegativeReviewsCount': CriticNegativeReviewsCount = value
            if key == 'CriticPositiveReviewsCount': CriticPositiveReviewsCount = value 
            if key == 'CriticScore': CriticScore = value
            if key == 'DirectedBy': DirectedBy = value
            if key == 'Genre': Genre = value 
            if key == 'InTheaters': InTheaters = value
            if key == 'OnDisc/Streaming': OnDisc_Streaming = value
            if key == 'Rating': Rating = value
            if key == 'Runtime': Runtime = value
            if key == 'Studio': Studio = value
            if key == 'WrittenBy': WrittenBy = value
        
            if Genre.find('Action&Adventure') > -1: Genre_Action_Adventure = 1
            if Genre.find('Animation') > -1: Genre_Animation = 1
            if Genre.find('Anime&Manga') > -1: Genre_Anime_Manga = 1
            if Genre.find('ArtHouse&International') > -1: Genre_ArtHouse_International = 1
            if Genre.find('Classics') > -1: Genre_Classics = 1
            if Genre.find('Comedy') > -1: Genre_Comedy = 1 
            if Genre.find('CultMovies') > -1: Genre_CultMovies = 1
            if Genre.find('Documentary') > -1: Genre_Documentary = 1
            if Genre.find('Drama') > -1: Genre_Drama = 1 
            if Genre.find('Gay&Lesbian') > -1: Genre_Gay_Lesbian = 1
            if Genre.find('Horror') > -1: Genre_Horror = 1
            if Genre.find('Kids&Family') > -1: Genre_Kids_Family = 1 
            if Genre.find('Musical&PerformingArts') > -1: Genre_Musical_PerformingArts = 1
            if Genre.find('Mystery&Suspense') > -1: Genre_Mystery_Suspense = 1
            if Genre.find('Romance') > -1: Genre_Romance = 1 
            if Genre.find('ScienceFiction&Fantasy') > -1: Genre_ScienceFiction_Fantasy = 1 
            if Genre.find('SpecialInterest') > -1: Genre_SpecialInterest = 1
            if Genre.find('Sports&Fitness') > -1: Genre_Sports_Fitness = 1 
            if Genre.find('Television') > -1: Genre_Television = 1
            if Genre.find('Western') > -1: Genre_Western = 1 
            
            if Rating.find('G') == 0: Rating_G = 1 
            if Rating.find('NC17') == 0: Rating_NC17 = 1
            if Rating.find('NR') == 0: Rating_NR = 1 
            if Rating.find('PG') == 0: 
                if Rating.find('PG-13') == -1: Rating_PG = 1 
            if Rating.find('PG-13') == 0: Rating_PG_13 = 1 
            if Rating.find('R') == 0: Rating_R = 1 
            
        movie_att_val_list = [Movie,AudienceAverageRating,AudienceScore,BoxOffice,Cast,CriticAverageRating,CriticNegativeReviewsCount,CriticPositiveReviewsCount,CriticScore,DirectedBy,Genre,Genre_Action_Adventure,Genre_Animation,Genre_Anime_Manga,Genre_ArtHouse_International,Genre_Classics,Genre_Comedy,Genre_CultMovies,Genre_Documentary,Genre_Drama,Genre_Gay_Lesbian,Genre_Horror,Genre_Kids_Family,Genre_Musical_PerformingArts,Genre_Mystery_Suspense,Genre_Romance,Genre_ScienceFiction_Fantasy,Genre_SpecialInterest,Genre_Sports_Fitness,Genre_Television,Genre_Western,InTheaters,OnDisc_Streaming,Rating,Rating_G,Rating_NC17,Rating_NR,Rating_PG,Rating_PG_13,Rating_R,Runtime,Studio,WrittenBy]
        
        print(movie_att_val_list)
        movies_df.loc[k] = movie_att_val_list
        k += 1
        
               
    col_vals.close()
    movie_file.close()

    return movies_df

if __name__ == "__main__":
    movies_df = create_pandas_df()
    movies_df.to_csv('movies.csv')
#    insert_data()
    