#%% 
# Set up your API key and import needed things
import mediacloud.api as mapi
from importlib.metadata import version
from dotenv import load_dotenv
import datetime as dt
from IPython.display import JSON
import bokeh.io
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import datetime as dt
import csv
from tqdm import tqdm
bokeh.io.reset_output()
bokeh.io.output_notebook()
MC_API_KEY = 'dc322e0b964683f8ca93839e71025a040137899b'
search_api = mapi.SearchApi(MC_API_KEY)

def generate_date_range(center_date_str, n_days):
    center_date = dt.datetime.strptime(center_date_str, '%d/%m/%Y').date()
    half_range = n_days // 2
    start_date = center_date - dt.timedelta(days=half_range)
    end_date = center_date + dt.timedelta(days=half_range)
    
    if n_days % 2 == 0:
        end_date -= dt.timedelta(days=1)
    
    return start_date, end_date

# Fetch recent stories
def fetch_recent_stories(query, start_date, end_date):
    stories, _ = search_api.story_list(query, start_date, end_date)
    return stories

# Fetch all stories for a specific day
def fetch_all_stories(query, start_date,end_date, collection_ids=[]):
    all_stories = []
    more_stories = True
    pagination_token = None
    while more_stories:
        page, pagination_token = search_api.story_list(query, start_date, end_date + dt.timedelta(days=1), collection_ids=collection_ids, pagination_token=pagination_token)
        all_stories += page
        more_stories = pagination_token is not None
    return all_stories

# Write stories to CSV
def write_stories_to_csv(stories, filename):
    fieldnames = ['id', 'publish_date', 'title', 'url', 'language', 'media_name', 'media_url', 'indexed_date']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for story in stories:
            writer.writerow(story)

# Fetch all stories and write each page to CSV with progress bar
def fetch_all_stories_with_progress(query, start_date, end_date, collection_ids=[], filename='test.csv'):
    all_stories = []
    pagination_token = None
    date = start_date
    with open(filename, 'w', newline='',encoding='utf-8') as csvfile:
        fieldnames = ['id', 'publish_date', 'title', 'url', 'language', 'media_name', 'media_url', 'indexed_date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        with tqdm(total=(end_date - start_date).days + 1, desc="Fetching Stories") as pbar:
            while date <= end_date:
                more_stories = True
                while more_stories:
                    page, pagination_token = search_api.story_list(query, date, date + dt.timedelta(days=1), collection_ids, pagination_token=pagination_token)
                    all_stories += page
                    for story in page:
                        writer.writerow(story)
                    more_stories = pagination_token is not None
                pagination_token = None
                date += dt.timedelta(days=1)
                pbar.update(1)
    
    return all_stories

def plot_series(df):
    df['date']= pd.to_datetime(df['date'])
    source = ColumnDataSource(df)   
    p = figure(x_axis_type="datetime", width=900, height=250)
    p.line(x='date', y='count', line_width=2, source=source)  # your could use `ratio` instead of `count` to see normalized attention
    return show(p)

def get_timesries(query, start_date,end_date):
    results = search_api.story_count_over_time(query, start_date, end_date)
    df = pd.DataFrame.from_dict(results)
    df['date']= pd.to_datetime(df['date'])
    source = ColumnDataSource(df)
    return df, source 

# %%
