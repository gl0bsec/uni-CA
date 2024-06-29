# Collection of media for the human rights treaties project

from cloud_functions import *
from download_big_WIP import download_data
from data_helpers import *
import os
from data_wrangling import *
from tqdm import tqdm
import shutil
treaties = ['CAT']
valid_events = GDELT_valid('updated_merged_tables.csv', 'fips-10-4-to-iso-country-codes.csv',treaty_acronyms=treaties)

# Download data for the treaties 
n = int(int(input('How many days? ')) / 2)
directory_path = 'big_dump/'
bucket_name = 'hr_news_1'
blob = 'bulk-news-test'

def create_index_name(date, query, prefix):
    date = date.replace("/", "_")
    query = query.replace("#", '_')
    return f"{query}{date}_{prefix}"

for event in tqdm(valid_events, desc=f"Downloading news coverage for {len(valid_events)} treaties"):
    # theme = 'HUMAN_RIGHTS'
    input_date = event[2]
    country = f'#{event[0]}#'
    download_data(None, n, input_date, locations_regex=country)
    index_name = create_index_name(input_date, country, f'{event[1]}_{n*2}days')
    files = os.listdir(directory_path)
    for file in files:
        filter_columns(file,f'filtered/GDELTfiltered_{file}')
        upload_to_gcs(bucket_name, f'{directory_path}/{file}', f'{blob}/{index_name}/raw/GDELTraw_{file}')
        upload_to_gcs(bucket_name, f'filtered/GDELTfiltered_{file}', f'{blob}/{index_name}/filtered/GDELTfiltered_{file}')
    shutil.rmtree(directory_path)