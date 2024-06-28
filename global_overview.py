#%% 
from cloud_functions import*
from download_big_WIP import download_data
from data_helpers import*
import os
from data_wrangling import*
import datetime as dt
#%%
# theme = 'HUMAN_RIGHTS'
last_updated='28/06/2024'
title = 'global_news'
n = 20
download_data(n)
# filename = str(input('what is the file name?'))
# file_input_path = filename+".csv"

def create_index_name(date,query,prefix):
    date = date.replace("/", ".")
    query = query.replace("#",'.').lower()
    return f"{query}{date}_{prefix}"

# input_date = '2/10/2019'
index_name = create_index_name(None,title,'testing')

pwd = None
directory_path = 'big_dump/'

files = os.listdir(directory_path)
for file in files:
    create_and_load_es_index(9200, directory_path+file, index_name,pwd)

os.remove('/big_dump')
