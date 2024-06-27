#%% 
from cloud_functions import*
from download_big_WIP import download_data
from data_helpers import*
import os
#%%
# theme = 'HUMAN_RIGHTS'
input_date = '2/10/2019'
country = '#AO#'
n = int(int(input('How many days? '))/2)
download_data(None,n,input_date,locations_regex=country)
# filename = str(input('what is the file name?'))
# file_input_path = filename+".csv"

def create_index_name(date,query,prefix):
    date = date.replace("/", ".")
    query = query.replace("#",'.').lower()
    return f"{query}{date}_{prefix}"

# input_date = '2/10/2019'
index_name = create_index_name(input_date,country,'test2')

pwd = None
directory_path = 'big_dump/'

files = os.listdir(directory_path)
for file in files:
    create_and_load_es_index(9200, directory_path+file, index_name,pwd)

os.remove('/big_dump')

#%%~
download_es_index_to_csv(9200, pwd, index_name, f'{index_name}.csv')

bucket_name = 'hr_news_1'
upload_to_gcs(bucket_name, f'{index_name}.csv', f'news_test/{index_name}.csv')
