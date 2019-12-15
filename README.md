### Disaster Response Pipeline Project
This python project performs multiclass multioutput text classification of 36 different disaster response categories.  The data for the project is stored in two different .csv files: 1)didaster_messages.csv 2)disaster categories.csv.  A python script data/process_data.py was used to clean and merge the two datasets and store them in a sqlite database, data/DisasterResponse.db , in the table train_test_data.  Asecond python script, models/train_classifier.py, is used to pull data from DisatserResponse.db, split the data into train and test datasets, and to train a multiclass classifier to classify disaster response messages into 36 different categories.  A flask app is used to run the model which allows a user to input a text message and the resulting classification categories will be highlighted on the webpage.  All required python modules can be found in requirements.txt.
### Instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


