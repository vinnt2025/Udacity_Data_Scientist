# Disaster Response Pipeline Project

## Table of Contents

 * [Project Overview](#project-overview)
 * [Requirements](#requirements)
 * [Repository Descriptions](#repository-descriptions)
 * [Deployment Guide](#deployment-guide)
 * [Acknowledgements](#acknowledgements)

## Project Overview
In this project, I try to build a classifies disaster messages model and visualization to analyze data. 

The dataset contain messages that were sent during disaster events can be found in here [Link](https://github.com/vinnt2025/Udacity_Data_Scientist/tree/master/Project2/data). 

And then, I create a machine learning pipeline to categorize these events. The detail code to make model can be found in here [Link](https://github.com/vinnt2025/Udacity_Data_Scientist/tree/master/Project2/models).

Besides that, I developed a web app base on Udacity's template which can display classification's result from a new message. You can see in here [Link](https://github.com/vinnt2025/Udacity_Data_Scientist/tree/master/Project2/app)

## Requirements
This project should run with develop environment as below:
- numPy
- pandas
- nltk
- sklearn
- sqlalchemy
- pickle
- flask
- plotly


## Repository Descriptions
This project has 3 folders: 
1. `Data`: That include: 
    - A python script "process_data.py"  to process data: Megre, clearn and store data. 
    - Two original data files: 
        - disaster_categories.csv
        - disaster_messages.csv 

2. `Models`: That include: 
    - A python script "train_classifier.py", to run a machine learning pipeline and create model to predict.

3. `App`: That include:
    - A web app which can input a new message and get classification results in several categories. 
    - The web app also can display some plots to analyze data. 

## Deployment Guide

1. Run the following commands in the project's root directory to create database and model.

    - To run ETL pipeline that cleans data and stores in database:
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves:
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

## Acknowledgements
- This project is part of the Udacity Data Scientist Nanodegree Program. The dataset used in this project is provided by Figure Eight (now part of Appen), which contains real messages sent during disaster events. Special thanks to Udacity for providing the platform and resources to complete this project.
