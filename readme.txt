~~~~~~~~Rotten Tomatoes Score Predictor Modelling~~~~~~~~~~

==============How to Run==============

1) cd to “processing_codes” folder.
2) open “main.py”
3) Set the path to the input raw data text file in line num - 15.
4) Set the mode to “test” in line num - 21.
5) Run the “main.py” in Python 3.


==============Full Description of the project==============

The high-level goal of this project is to predict the difference between the audience
score and the critic score of a movie given the movie’s information.

The low-level tasks include, 
1) Data scraping extraction from rotten tomatoes website.
2) Data cleaning and augmenting (Clean and extract some useful information (objects) for later use in data wrangling).
3) Data wrangling - (Convert the input raw data into cleaned and ready-to-train features using the objects obtained in step 2).
4) Model Training 
5) Model Testing


Description of Low-Level Tasks:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Data scraping:
	The data was scraped from rotten tomatoes website using selenium web driver and Beautiful soup. A total of 14642 movies were scrapped from the website. The information scraped for a movie included; movie name, cast, director, writer, studio, genre, rating, release date and type, runtime, synopsis, critic score, audience score, box-office.
	
	*Highlight*: Rotten tomatoes website’s “show-all-movies” will display only upto around 9000 movies which proved to be a bottleneck for this project. Instead, the movies were scraped in an automated way by going to movies in each genre and finally removing the duplicates. This gave 14642 movies.

2) Data cleaning and augmenting: (processing_codes/clean_data.py)
	Out of all the features scrapped, synopsis, box-office were ignored. Data was parsed, cleaned and converted to various dict objects for use in next step. This is an one-time run step. Once all the dictionary objects have been created and stored, this step won’t be run.

	*Actornames, Writtenby, Directedby*: Each actor in the entire data set was 	given a normalized score.This score was obtained by dividing the total movies the actor has appeared in the data set by the total number of movies. Then this set of values was normalized across all the actors and this score was 	given to each actor. This dict was stored in an object and loaded in data wrangling.

	*Studio*: To be filled

	*Genre, Rating*: Various genre’s and rating that was found in the entire dataset were extracted and stored in dict object and later loaded in data wrangling. These features were later one-hot encoded in data wrangling.

	*Runtime, Release date and type*: These features were binned. Runtime was grouped by 15 mins i.e. a runtime between 60-75 mins will be grouped in one 	bin and the index of the bin is used in features. This index was later one-hot encoded in data wrangling. For release date, the year’s were binned i.e. movies released in 1920-1929 were binned together and got the index 192. Release type (wide or limited) was also one-hot encoded and added to the features.
	

3) Data wrangling: (processing_codes/wrangle_data.py)

	The model’s pipeline starts from here. It loads the dictionary objects that were created and saved in the previous step and uses it here to map the incoming raw data text file to feature matrix. The one-hot encodable features (genre, runtime, studio, rating, release date, release type) were one-hot encoded and the column names is stored as a model so that it could be loaded during testing. The score based features (actor names, writtenby, directedby) are computed using the normalized scores computed in data cleaning. (Only top two actors in a movie are chosen for score computation and their mean is stored).  The critic score is subtracted from audience score and stored as “y” (label) feature column vector. All these features are assembled (horizontally stacked) and converted to numpy matrix and stored as npz files. These npz files are then loaded in “Model Building” step.

4) Model Training: (processing_codes/build_model.py)

	The input raw text file was randomly split into 80/20 ratio for training and testing respectively. The stored npz files were loaded and trained here. All the trained models are stored and loaded in testing.

	*Highlights*:
	1) The total number of features were 223.
	2) More than 20 models were tried and the best three were picked based on Mean Absolute Error and Mean Squared Error.
	3) All the models were trained using 5-fold cross validation and the best model from the 5-folds is picked.	
	4) The three models that gave the lowest error values are “Random Forest”, “Bagging”, “Extra Trees” regressor.
	5) GridSearch CV was used to pick the number of estimators in each of the model.
	6) Once the y values are predicted using individual models, the three y-vectors from three models were again trained using “Support Vector Regressor” with the actual y-values as the y-vector. This is a known method in machine learning called “model-stacking”. The idea of model-stacking is that if one model fails in certain test-cases, there is a chance that the other models could perform in those. The second level model tries to model that.

5) Model Testing: (processing_codes/predict_difference.py)

	The model was tested on the test raw data text file using the loaded models that were saved in the previous step.

6) Final Touch:

	Once, all the models have been selected and fine-tune, the models were trained on the whole 14642 movie data set to increase the training sample set.


======================================================================

	

	

	
	
	