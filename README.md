# Disaster Response Pipeline Project

### Contents of repository
- *app*: Flask web app with a few graphs about the data and message categorization functionality:
  - *templates*
    - *master.html*: index page
    - *go.html*: html needed when user inputs message to be categorized
  -  *run.py*: script to run web app
- *data*: contains sample data and etl pipeline script
  - *disaster_messages.csv*: sample csv file of messages
  - *disaster_categories.csv*: sample csv file of categories
  - *process_data.py*: etl pipeline script to read in two csv files and generate sqlite database
- *models*: machine learning pipeline
  - *train_classifier.py*: script to read in sqlite database and generate classification model

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
