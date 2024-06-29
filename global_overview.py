#%% 
from cloud_functions import*
from download_big_WIP import download_data
from data_helpers import*
import os
from data_wrangling import*
import datetime as dt
import shutil
from datetime import date, timedelta, datetime

def save_current_date_to_file():
    # Get current date
    current_date = date.today()
    
    # Format the date as a string in DD/MM/YYYY format
    formatted_date = current_date.strftime("%d/%m/%Y")
    
    # Write the date to a text file (overwriting existing content)
    filename = "last_updated.txt"
    with open(filename, 'w') as file:
        file.write(formatted_date + "\n")
    
    print(f"Current date {formatted_date} has been saved to {filename}")
    
def calculate_date_difference():
    # Get current date
    current_date = date.today()
    
    # Read the saved date from file
    filename = "current_date.txt"
    with open(filename, 'r') as file:
        saved_date_str = file.read().strip()
    
    # Convert the saved date string to a date object (handle different date formats)
    try:
        saved_date = datetime.strptime(saved_date_str, "%Y-%m-%d").date()
    except ValueError:
        saved_date = datetime.strptime(saved_date_str, "%d/%m/%Y").date()
    
    # Calculate the difference in days
    difference = current_date - saved_date
    
    # Return the difference in days as an integer
    return difference.days


#%%
# theme = 'HUMAN_RIGHTS'
last_updated='28/06/2024'
title = 'global_news'
n = 20
download_data(1,n)
# filename = str(input('what is the file name?'))
# file_input_path = filename+".csv"

def create_index_name(query,prefix):
    query = query.replace("#",'.').lower()
    return f"{query}_{prefix}"

# input_date = '2/10/2019'
index_name = create_index_name(title,'testing')

pwd = None
directory_path = 'big_dump/'

files = os.listdir(directory_path)
for file in files:
    create_and_load_es_index(9200, directory_path+file, index_name,pwd)

shutil.rmtree('/big_dump')

# %%
download_es_index_to_csv(9200, pwd, index_name, f'{index_name}.csv')
transform_csv(f'{index_name}.csv').to_csv(f'{index_name}_raw.csv')
filter_columns(f'{index_name}_raw.csv',f'{index_name}_filtered.csv')
# %%
