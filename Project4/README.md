# Project: Used Car Price Prediction

## Table of Contents

 * [Requirements](#requirements)
 * [Project Overview](#project-overview)
 * [File Descriptions](#file-descriptions)
 * [Source code](#source-code)
 * [Acknowledgements](#acknowledgements)

## Requirements
 - NumPy
 - Pandas
 - Seaborn
 - Matplotlib
 - Scikit-learn
 - Flask 
 - SQLite3
 - Pickle
 - Json
 
No additional installations beyond the Anaconda distribution of Python and Jupyter notebooks.

## Project Overview
For this project I was interested in predicting price for used car. 

The project involved:
- This dataset could be founded  in the Kaggle Dataset [used-car-price](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction)
 - Loading and cleaning a small data with ~6K records and ~1.8K different car's name.
 - Conducting Exploratory Data Analysis to understand the data and what features are useful for predicting car price.
 - Feature Engineering to create features that will be used in the modeling process
 - Modeling using machine learning algorithms such as Linear Regression, Random Forest, Gradient Boosting Regressor, XGB Regressor.

## File Descriptions
There is one exploratory notebook and html file of the notebook available here to showcase my work in predicting price. Markdown cells were used throughout to explain the process taken.


## Source code
- The main findings of the code can be found on Github available [here](https://github.com/vinnt2025/Udacity_Data_Scientist/tree/master/Project4) which explain the technical details of my project.

- View the project notebook at `usedcar_price_prediction.ipynb`.

- Open folder `app` and type `python app.py` to run the web app.

- A Random Forest Classifier was chosen to be the best model by evaluating R2 score metrics. 
The final model achieved an R2 score of 0.905 on Test dataset. 

## Acknowledgements.
I'd like to acknowledge Udacity for the project idea and workspace.

