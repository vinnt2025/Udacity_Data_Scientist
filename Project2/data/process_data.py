# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories datasets.

    Parameters:
        messages_filepath (str): File path for the messages dataset in CSV format.
        categories_filepath (str): File path for the categories dataset in CSV format.

    Returns:
        pd.DataFrame: A dataframe containing merged data from messages and categories datasets."""
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge the messages and categories datasets using the common id
    output_df = pd.merge(messages, categories, how ='outer', on ='id')
    
    return output_df


def clean_data(df):
    """Clean and transform the merged dataframe by splitting categories into individual columns and converting values.

    This function processes the dataframe by performing the following steps:
    1. Splits the 'categories' column into separate columns for each category.
    2. Renames the new category columns appropriately.
    3. Converts category values to numerical values (0 or 1).
    4. Replaces any occurrences of the value 2 in the 'related' column with 1.
    5. Drops the original 'categories' column from the dataframe.
    6. Concatenates the original dataframe with the new category columns.
    7. Removes any duplicate rows from the dataframe.

    Parameters:
        df (pd.DataFrame): The merged dataframe containing messages and categories.

    Returns:
        pd.DataFrame: The cleaned dataframe with separate category columns and no duplicates."""
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
#     print("first row: \n", row)
#     print("----------------------")
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda x: x[:-2]))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Check all unique values
#     for col in categories.columns:
#         print("Column's name: ",col, '-> unique values:', categories[col].unique())
    
    # Replace 2 -> 1
    categories['related'] = categories['related'].replace(to_replace=2, value=1)
    
    # replace `categories` column in `df` with new category columns
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    output_df = pd.concat([df, categories], axis=1)
    
#     # check number of duplicates
#     output_df[output_df.duplicated()]
    
    # drop duplicates
    output_df.drop_duplicates(inplace=True, keep='first')
    
    return output_df

def save_data(df, database_filename):
    """Save the cleaned dataframe to a SQLite database.

    Parameters:
        df (pd.DataFrame): The cleaned dataframe to be saved.
        database_filename (str): The filename for the SQLite database.

    Returns:
        None"""
    
    engine = create_engine('sqlite:///'+ str(database_filename))
    df.to_sql("DisasterMessages", engine, index=False,if_exists='replace')

def main():
    """Main function to execute the data processing pipeline.

    This function performs the following steps:
    1. Loads messages and categories datasets from the provided file paths.
    2. Cleans the merged dataset.
    3. Saves the cleaned data to a SQLite database.

    The function expects three command-line arguments:
    1. Filepath for the messages dataset (CSV file).
    2. Filepath for the categories dataset (CSV file).
    3. Filepath for the SQLite database to save the cleaned data.
    If the correct number of arguments is not provided, it prints usage instructions.
    Parameters:
        None
    Returns:
        None"""
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()