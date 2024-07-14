import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import time
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """Load data from the SQLite database and prepare for training.

    This function performs the following steps:
    1. Connects to the SQLite database specified by the file path.
    2. Loads the 'DisasterMessages' table into a pandas DataFrame.
    3. Separates the dataframe into input features (messages) and target variables (categories).
    4. Extracts the category names for target variables.

    Parameters:
        database_filepath (str): The filepath of the SQLite database.

    Returns:
        X (pd.Series): The input features (messages).
        Y (pd.DataFrame): The target variables (categories).
        Y_names (np.ndarray): The names of the category columns."""
    
    # Cerate engine database
    engine = create_engine('sqlite:///'+ str(database_filepath))
    # Load data
    df = pd.read_sql_table('DisasterMessages', engine)
    # Create training data
    X = df.message
    Y = df.drop(columns = ["id","message", "genre", "original"])
    Y_names = Y.columns.values
    
    return X, Y, Y_names


def tokenize(text):
    """Tokenize and clean text data.

    This function performs the following steps:
    1. Removes all characters except alphanumeric from the text.
    2. Tokenizes the text into individual words.
    3. Lemmatizes each word to its base form.
    4. Converts each token to lowercase and strips any surrounding whitespace.
    5. Collects and returns the cleaned tokens as a list.

    Parameters:
        text (str): The text to be tokenized and cleaned.

    Returns:
        list: A list of cleaned and lemmatized tokens."""
    
    clean_text = re.sub(r"[^a-zA-Z0-9]", "", text)
    tokens = word_tokenize(clean_text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens=[]
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    """Build a machine learning pipeline for multi-output classification.

    This function creates a machine learning pipeline that performs the following steps:
    1. Vectorizes the text data using CountVectorizer with a custom tokenizer.
    2. Transforms the vectorized data using TfidfTransformer.
    3. Classifies the data using a RandomForestClassifier wrapped in a MultiOutputClassifier to handle multiple target variables.
    Parameters:
        None
        
    Returns:
        sklearn.pipeline.Pipeline: A machine learning pipeline."""
    
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return model

def tuning_model(pipeline, x_train, y_train):
    """Tune the hyperparameters of the machine learning pipeline using GridSearchCV.

    Parameters:
        pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline to be tuned.
        x_train (pd.Series or np.ndarray): The training input data (messages).
        y_train (pd.DataFrame or np.ndarray): The true labels for the training data.

    Returns:
        GridSearchCV: The GridSearchCV object after fitting it to the training data."""
    parameters = {
        'clf__estimator__n_estimators': [20, 40],
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(x_train, y_train)
    
    # Display best parameters
    print("Best parameters: ", cv.best_params_)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the performance of the machine learning model.

    This function performs the following steps:
    1. Uses the trained model to predict on the test set.
    2. Calculates and prints the mean accuracy for each category.
    3. Prints the classification report for each category.

    Parameters:
        model (sklearn.pipeline.Pipeline): The trained machine learning model.
        X_test (pd.Series or np.ndarray): The test input data (messages).
        Y_test (pd.DataFrame or np.ndarray): The true labels for the test data.
        category_names (list): The list of category names.

    Returns:
        None"""
    
    Y_predict = model.predict(X_test)
    accuracy = (Y_predict == Y_test).mean()
    print("Accuracy mean: \n", accuracy)
    print("---------------------------------------------------------")
    for idx, col in enumerate(category_names):
        print(col, classification_report(Y_test[col],Y_predict[:,idx]))


def save_model(model, model_filepath):
    """Save the trained machine learning model as a pickle file.

    This function serializes the provided machine learning model and saves it to the specified file path using the pickle module.

    Parameters:
        model (sklearn.pipeline.Pipeline): The trained machine learning model to be saved.
        model_filepath (str): The file path where the model should be saved.

    Returns:
        None"""
    
    # Exports model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """Main function to execute the machine learning pipeline.

    This function performs the following steps:
    1. Loads data from the SQLite database specified by the first command-line argument.
    2. Splits the data into training and test sets.
    3. Builds a machine learning model.
    4. Trains and fine-tuning model on the training set.
    5. Evaluates the model on the test set.
    6. Saves the trained model to a pickle file specified by the second command-line argument.

    The function expects two command-line arguments:
    1. Filepath for the SQLite database containing disaster messages.
    2. Filepath for the pickle file to save the trained model.
    If the correct number of arguments is not provided, it prints usage instructions.
    Parameters:
        None
    Returns:
        None"""
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Fine-tuning model ...')
        best_model = tuning_model(model, X_train, Y_train)
        
#         print('Training model...')
#         model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(best_model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()