#%% 
from cloud_functions import*
from download_big_WIP import download_data
from data_helpers import*
import os

# theme = 'HUMAN_RIGHTS'
input_date = '2/10/2019'
country = '#AO#'
n = int(int(input('How many days? '))/2)
download_data(None,n,input_date,locations_regex=country)

# filename = str(input('what is the file name?'))
# file_input_path = filename+".csv"

pwd = None
prefix = 'test'
index_name = f"{country}{input_date}_{prefix}"

create_and_load_es_index(9200, 'test.csv', 'index_name',pwd)
 
# bucket_name = 'hr_news_1'
# directory_path = 'big_dump/'
# files = os.listdir(directory_path)

# for file in files:
#     destination_blob_name = 'news-test/'+file
#     upload_to_gcs(bucket_name, directory_path+file, destination_blob_name)
