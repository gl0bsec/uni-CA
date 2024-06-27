#%% 
from cloud_functions import*
from download_big_WIP import download_data
import os

# theme = 'HUMAN_RIGHTS'
input_date = '2/10/2019'
country = '#AO#'
download_data(None,int(int(input('How many days? '))/2),input_date,locations_regex=country)
# filename = str(input('what is the file name?'))
# file_input_path = filename+".csv"

bucket_name = 'hr_news_1'
directory_path = 'big_dump/'
files = os.listdir(directory_path)

for file in files:
    destination_blob_name = 'news-test/'+file
    upload_to_gcs(bucket_name, directory_path+file, destination_blob_name)
