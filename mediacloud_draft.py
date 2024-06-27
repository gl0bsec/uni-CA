#%% 
# Set up your API key and import needed things
import os
import mediacloud.api
from importlib.metadata import version
from dotenv import load_dotenv
import datetime as dt
from IPython.display import JSON
import bokeh.io
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
bokeh.io.reset_output()
bokeh.io.output_notebook()
MC_API_KEY = 'dc322e0b964683f8ca93839e71025a040137899b'
search_api = mediacloud.api.SearchApi(MC_API_KEY)

#%%
# check how many stories include the phrase "climate change" in the Washington Post (media id #2)
my_query = '"climate change"' # note the double quotes used to indicate use of the whole phrase
start_date = dt.date(2023, 11, 1)
end_date = dt.date(2023, 12,1)
sources = [2]
search_api.story_count(my_query, start_date, end_date, source_ids=sources)
# you can see this count by day as well
results = search_api.story_count_over_time(my_query, start_date, end_date, source_ids=sources)
JSON(results)

#%%
# and you can chart attention over time with some simple notebook work (using Bokeh here)
df = pd.DataFrame.from_dict(results)
df['date']= pd.to_datetime(df['date'])
source = ColumnDataSource(df)
p = figure(x_axis_type="datetime", width=900, height=250)
p.line(x='date', y='count', line_width=2, source=source)  # your could use `ratio` instead of `count` to see normalized attention
show(p)

#%% 

# or compare to another country (India in this case)
INDIA_NATIONAL = 34412118
results = search_api.story_count('"climate change"', start_date, end_date, collection_ids=[INDIA_NATIONAL])
india_country_ratio = results['relevant'] / results['total']
'{:.2%} of stories from national-level Indian media sources in 2019 mentioned "climate change"'.format(india_country_ratio)

coverage_ratio =  1 / (india_country_ratio / us_country_ratio)
'at the national level "climate change" is covered {:.2} times less in India than the US'.format(coverage_ratio)

#%%

# Fetch recent stories
def fetch_recent_stories(query, start_date, end_date):
    stories, _ = search_api.story_list(query, start_date, end_date)
    return stories

# Fetch all stories for a specific day
def fetch_all_stories(query, date, collection_id):
    all_stories = []
    more_stories = True
    pagination_token = None
    while more_stories:
        page, pagination_token = search_api.story_list(query, date, date + dt.timedelta(days=1), collection_ids=[collection_id], pagination_token=pagination_token)
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

#%%
# grab the most recent stories about this issue
stories, _ = search_api.story_list(my_query, start_date, end_date)
stories[:3]

#%% 
# let's fetch all the stories matching our query on one day
all_stories = []
more_stories = True
pagination_token = None
while more_stories:
    page, pagination_token = search_api.story_list(my_query, dt.date(2023,11,29), dt.date(2023,11,30),
                                                   collection_ids=[US_NATIONAL_COLLECTION],
                                                   pagination_token=pagination_token)
    all_stories += page
    more_stories = pagination_token is not None
len(all_stories)

#%% 
# Writing a CSV of Story Data
import csv
fieldnames = ['id', 'publish_date', 'title', 'url', 'language', 'media_name', 'media_url', 'indexed_date']

with open('story-list.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for s in all_stories:
        writer.writerow(s)

# and let's make sure it worked by checking out by loading it up as a pandas DataFrame
import pandas
df = pandas.read_csv('story-list.csv')
df.head()

