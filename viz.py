# Import stuff
from collections import Counter
import pandas as pd
import networkx as nx
from itertools import combinations
import warnings
import re
import numpy as np
import matplotlib.pyplot as plt
import ast

def interactive_plot_top_tags(df, column, date_column, top_n=10, include_regexes='', exclude_regexes='', start_date='2019-01-01', end_date='2019-12-31'):
    def plot(top_n=10, include_regexes='', exclude_regexes='', start_date='2019-01-01', end_date='2019-12-31'):
        # Convert the date column to datetime format
        df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y')

        # Filter by date range if specified
        if start_date:
            df_filtered = df[df[date_column] >= pd.to_datetime(start_date)]
        else:
            df_filtered = df
        if end_date:
            df_filtered = df_filtered[df_filtered[date_column] <= pd.to_datetime(end_date)]

        # Process regex lists
        include_patterns = include_regexes.split(',') if include_regexes else []
        exclude_patterns = exclude_regexes.split(',') if exclude_regexes else []

        # Initialize a Counter to count tags
        tag_counter = Counter()

        # Iterate through each row in the specified column
        for items in df_filtered[column].dropna():
            item_list = items.split(';')

            # Apply include regex filters if specified
            if include_patterns:
                item_list = [item for item in item_list if any(re.search(pattern, item) for pattern in include_patterns)]

            # Apply exclude regex filters if specified
            if exclude_patterns:
                item_list = [item for item in item_list if not any(re.search(pattern, item) for pattern in exclude_patterns)]

            # Update the counter with the tags
            tag_counter.update(item_list)

        # Get the top N tags
        top_tags = tag_counter.most_common(top_n)

        # Separate the tags and their counts
        tags, counts = zip(*top_tags) if top_tags else ([], [])

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.bar(tags, counts, color='skyblue')
        plt.title(f'Top {top_n} Tags')
        plt.xlabel('Tags')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    interact(plot,
             top_n=widgets.IntSlider(min=1, max=50, step=1, value=10, description='Top N'),
             include_regexes=widgets.Text(value='', description='Include Regexes (comma-separated)'),
             exclude_regexes=widgets.Text(value='', description='Exclude Regexes (comma-separated)'),
             start_date=widgets.DatePicker(value=pd.to_datetime('2019-01-01'), description='Start Date'),
             end_date=widgets.DatePicker(value=pd.to_datetime('2019-12-31'), description='End Date')
    )


def interactive_time_series_bar_plot(df, date_column, target_column):
    def plot(start_date='2019-01-01', end_date='2019-12-31', include_regex='', exclude_regex=''):
        # Convert the date column to datetime format
        df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y')

        # Filter by date range if specified
        if start_date:
            df_filtered = df[df[date_column] >= pd.to_datetime(start_date)]
        else:
            df_filtered = df
        if end_date:
            df_filtered = df_filtered[df_filtered[date_column] <= pd.to_datetime(end_date)]

        # Apply include regex filter if specified
        if include_regex:
            include_patterns = include_regex.split(',')
            df_filtered = df_filtered[df_filtered[target_column].str.contains('|'.join(include_patterns), na=False)]

        # Apply exclude regex filter if specified
        if exclude_regex:
            exclude_patterns = exclude_regex.split(',')
            df_filtered = df_filtered[~df_filtered[target_column].str.contains('|'.join(exclude_patterns), na=False)]

        # Count occurrences of each date
        date_counts = df_filtered[date_column].value_counts().sort_index()

        # Plotting
        plt.figure(figsize=(12, 6))
        date_counts.plot(kind='bar', color='skyblue')
        plt.title('Time Series Bar Plot')
        plt.xlabel('Date')
        plt.ylabel('Frequency')

        # Customize x-ticks to show day, month, and year
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(plt.FixedFormatter(date_counts.index.strftime('%d-%m-%Y')))

        plt.tight_layout()
        plt.show()

    interact(plot,
             start_date=widgets.DatePicker(value=pd.to_datetime('2019-01-01'), description='Start Date'),
             end_date=widgets.DatePicker(value=pd.to_datetime('2019-12-31'), description='End Date'),
             include_regex=widgets.Text(value='', description='Include Regexes (comma-separated)'),
             exclude_regex=widgets.Text(value='', description='Exclude Regexes (comma-separated)')
    )
