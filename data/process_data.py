# import libraries
import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Function that loads data from 'messages_filepath' and 'categories_filepath'
        and returns a df with merged data of the two datasets.
    Args:
        message_filepath (str): relative file path to 'disaster_messages.csv'
        message_filepath (str): relative file path to 'disaster_categories.csv'
    Returns:
        df (pandas df): dataframe with merged data
    """
    try:
        # load messages data set and categories dataset
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
    except Exception as e:
        print e
    
    # merge datasets
    df = messages.merge(categories, on = 'id', how='left')

    return df



def clean_data(df):
    """Function that cleans data in df.
    Args:
        df (pandas df): df returned from load_data()
    Returns:
        df (pandas df): cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # clean the categories by removing the last 2 characters
    row = row.apply(lambda x: x[:-2])
    # create a list of col names
    category_colnames = list(row)
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the int of last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # remove duplicate rows of data
    
    return df



def save_data(df, database_filename):
    """Function that creates a sqlite table from a pandas df
    Args:
        df (pandas df): df to make table
    Returns:
        None
    """
    # create connection and convert df to table, close conn after
    try:
        conn = create_engine('sqlite:///InsertDatabaseName.db')
    except Exception as e:
        print(f'''The following error occured while trying to make 
              a sqlit db connection: {e}'''
    
    df.to_sql('InsertTableName', conn, index=False)
    conn.close()
    


def main():
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

