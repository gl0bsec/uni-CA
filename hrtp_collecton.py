# Collection of media for the human rights treaties project
#%%
from cloud_functions import *
from download_big_WIP import download_data
from data_helpers import *
import os
from data_wrangling import *
from tqdm import tqdm
import shutil
treaties = ['CAT']
valid_events = GDELT_valid('updated_merged_tables.csv', 'fips-10-4-to-iso-country-codes.csv',)


#%%
# Download data for the treaties 
n = int(int(input('How many days? ')) / 2)
pwd = None
directory_path = 'big_dump/'

def create_index_name(date, query, prefix):
    date = date.replace("/", ".")
    query = query.replace("#", '.').lower()
    return f"{query}{date}_{prefix}"

for event in tqdm(valid_events, desc=f"Downloading news coverage for {len(valid_events)} treaties"):
    # theme = 'HUMAN_RIGHTS'
    input_date = event[2]
    country = f'#{event[0]}#'
    download_data(None, n, input_date, locations_regex=country)
    index_name = create_index_name(input_date, country, f'{event[1].lower()}_{n*2}days')
    files = os.listdir(directory_path)
    for file in files:
        create_and_load_es_index(9200, directory_path + file, index_name, pwd)
    print(f"successfully created {index_name}")
    
    shutil.rmtree(directory_path)
    download_es_index_to_csv(9200, pwd, index_name, f'{index_name}.csv')
    transform_csv(f'{index_name}.csv').to_csv(f'{index_name}_raw.csv')
    filter_columns(f'{index_name}_raw.csv',f'{index_name}_filtered.csv')

    bucket_name = 'hr_news_1'
    blob = 'news-test'
    upload_to_gcs(bucket_name, f'{index_name}.csv', f'{blob}/{index_name}_rawGDELT.csv')
    upload_to_gcs(bucket_name, f'{index_name}.csv', f'{blob}/{index_name}_filteredGDELT.csv')

# %%
