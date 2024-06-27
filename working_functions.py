import pandas as pd
import networkx as nx
from itertools import combinations
import warnings 
import re 
import numpy as np
# from ipysigma import Sigma
import matplotlib.pyplot as plt
# from pyvis.network import Network

# Working functions library
def filter_csv_by_substring(data, column, substring):
    """
    Filters the given DataFrame by looking for a substring in the specified column.

    Parameters:
    - data: The DataFrame to filter.
    - column: The column to search the substring in. Can be 'country_code', 'latitude', 'longitude', or 'geo_location'.
    - substring: The substring to search for in the column.

    Returns:
    - A filtered DataFrame containing only the rows where the substring is found in the specified column.
    """
    # For 'geo_location' which contains dictionary-like strings, special handling is required
    if column == 'geo_location':
        # Convert the string representation of dictionaries to actual dictionaries
        data[column] = data[column].apply(eval)
        # Filter rows where any of the dictionaries contain the substring in either 'lat' or 'lon' keys as strings
        filtered_data = data[data[column].apply(lambda x: any(substring in str(loc['lat']) or substring in str(loc['lon']) for loc in x))]
    else:
        # Standard filtering for strings
        filtered_data = data[data[column].astype(str).str.contains(substring, na=False)]
    
    return filtered_data

def clean_item(item):
    """
    Clean the given string by removing square brackets and extra whitespace.
    
    Args:
        item (str): The string to clean.
    
    Returns:
        (str): The cleaned string.
    """
    item = re.sub(r"[\[\]]", "", item)  # Remove square brackets
    return item.strip()

def create_cooccurrence_matrix(df, column_name, filter_list, min_occurrences):
    """
    Creates a co-occurrence matrix from a column in a DataFrame, including string cleaning to remove square brackets.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing organizations or themes.
        filter_list (list): A list of organizations or themes to exclude.
        min_occurrences (int): The minimum number of occurrences to include an edge.
    
    Returns:
        G (nx.Graph): A networkx graph of the co-occurrence network.
    """
    G = nx.Graph()

    for items in df[column_name].dropna():
        # Clean items and split; also remove any items in the filter list
        items = [clean_item(item) for item in items.split(',') if clean_item(item) not in filter_list]
        
        for org_pair in combinations(set(items), 2):
            if G.has_edge(*org_pair):
                G[org_pair[0]][org_pair[1]]['weight'] += 1
            else:
                G.add_edge(*org_pair, weight=1)

    # Filter edges by weight (min_occurrences)
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] <= min_occurrences]
    G.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return G

def compute_network_metrics(G):
    """
    Computes network metrics for a given graph.
    
    Args:
        G (nx.Graph): The networkx graph of the co-occurrence network.
    
    Returns:
        metrics_df (pd.DataFrame): DataFrame with nodes and their metrics.
    """
    degree_centrality = nx.degree_centrality(G)
    pagerank_centrality = nx.pagerank(G, weight='weight')
    clustering_coefficient = nx.clustering(G, weight='weight')
    
    metrics_df = pd.DataFrame({
        'Node': list(G.nodes),
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'PageRank Centrality': [pagerank_centrality[node] for node in G.nodes()],
        'Clustering Coefficient': [clustering_coefficient[node] for node in G.nodes()]
    })
    
    return metrics_df

def plot_filtered_time_series(df, search_substrings, field_to_check):
    """
    Plots a time series bar chart of entries in the dataframe filtered based on whether the specified field
    contains any of the given substrings.

    Parameters:
    - df: DataFrame containing the data.
    - search_substrings: List of substrings to search for within the specified field.
    - field_to_check: The field within the DataFrame to check for the presence of substrings ('THEMES' or 'ORGANIZATIONS').

    Returns:
    - A ggplot2-styled time series bar chart.
    """
    
    # Ensure the DataFrame copies aren't being modified
    df_copy = df.copy()
    
    # Function to check if any substring is present in the selected field
    def contains_substring(text, substrings):
        if pd.isna(text):
            return False
        text = text.lower()
        return any(substring.lower() in text for substring in substrings)

    # Filter dataframe based on the presence of any substrings in the selected field
    df_filtered = df_copy[df_copy[field_to_check].apply(contains_substring, substrings=search_substrings)]

    # Convert DATE to datetime format and extract date part
    df_filtered['DATE'] = pd.to_datetime(df_filtered['DATE']).dt.date

    # Count the number of entries for each date
    date_counts = df_filtered['DATE'].value_counts().sort_index()

    # Set the ggplot style
    plt.style.use('ggplot')

    # Plotting as a time series bar plot with ggplot style
    plt.figure(figsize=(14, 7))  # Adjust the figure size as necessary
    plt.bar(date_counts.index, date_counts.values, color='skyblue', width=0.8)

    plt.xlabel('Date')
    plt.ylabel('Number of Entries')
    plt.title(f'Frequency of Entries by Date - Time Series (Field Checked: {field_to_check})')

    # Improve readability of the x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()

def visualize_network_with_colormap(G):
    """
    Visualizes the network using ipysigma. Nodes are sized by degree centrality
    and colored by clustering coefficient using a cool-to-warm colormap.
    
    Args:
        G (nx.Graph): A NetworkX graph object.
    """
    # Calculating metrics
    degree_centrality = nx.pagerank(G, weight='weight')
    clustering_coefficient = nx.clustering(G)
    
    # Getting the coolwarm colormap
    cmap = plt.get_cmap('coolwarm')
    
    # Normalize clustering coefficient values to [0, 1] for colormap
    norm = plt.Normalize(0, 1)
    
    # Assigning size and color based on metrics
    for node in G.nodes:
        G.nodes[node]['size'] = degree_centrality[node] * 20  # Adjust size scaling as needed
        # Mapping clustering coefficient to color using colormap
        G.nodes[node]['color'] = cmap(norm(clustering_coefficient[node]))
    
    # Visualization with ipysigma (assumes ipysigma handles color correctly)
    sigma = Sigma(G, start_layout=True)
    return sigma

def visualize_network_pyvis(G):
    """
    Visualizes a network with nodes sized by degree centrality and colored
    by clustering coefficient from cool to warm colors using PyVis.
    
    Args:
        G (nx.Graph): The NetworkX graph object to visualize.
    """
    # Calculating metrics
    degree_centrality = nx.degree_centrality(G)
    clustering_coefficient = nx.clustering(G)
    
    # Create a PyVis network
    nt = Network(notebook=True, height="750px", width="100%")
    
    # Normalize clustering coefficient for color mapping
    norm = plt.Normalize(0, 1)
    cmap = plt.get_cmap('coolwarm')
    
    # Adding nodes to the PyVis network
    for node in G.nodes:
        size = degree_centrality[node] * 50  # Scaling size for visibility
        color_value = clustering_coefficient[node]
        color = plt.colors.rgb2hex(cmap(norm(color_value)))
        
        nt.add_node(node, title=node, size=size, color=color)
    
    # Adding edges to the PyVis network
    for edge in G.edges:
        nt.add_edge(edge[0], edge[1])
    
    # Generate network visualization
    nt.show("network.html")


def plot_filtered_time_series_tone1_sum_v2(df, search_substrings, field_to_check):
    """
    Adjusted function to plot a time series line chart of the sum of 'tone1' for entries in the dataframe 
    filtered based on whether the specified field contains any of the given substrings, without using eval().

    Parameters:
    - df: DataFrame containing the data.
    - search_substrings: List of substrings to search for within the specified field.
    - field_to_check: The field within the DataFrame to check for the presence of substrings ('THEMES' or 'ORGANIZATIONS').

    Returns:
    - A ggplot2-styled time series line chart showing the sum of 'tone1' by date.
    """
    
    # Ensure the DataFrame copies aren't being modified
    df_copy = df.copy()
    
    # Convert 'TONE' column from string to dictionary if it's not already a dictionary
    df_copy['TONE'] = df_copy['TONE'].apply(lambda x: x if isinstance(x, dict) else eval(x))
    
    # Extract 'tone1' from the 'TONE' dictionary
    df_copy['tone1'] = df_copy['TONE'].apply(lambda tone: tone.get('tone1', 0))
    
    # Function to check if any substring is present in the selected field
    def contains_substring(text, substrings):
        if pd.isna(text):
            return False
        text = text.lower()
        return any(substring.lower() in text for substring in substrings)

    # Filter dataframe based on the presence of any substrings in the selected field
    df_filtered = df_copy[df_copy[field_to_check].apply(contains_substring, substrings=search_substrings)]

    # Convert DATE to datetime format and extract date part
    df_filtered['DATE'] = pd.to_datetime(df_filtered['DATE']).dt.date

    # Aggregate the sum of tone1 for each date
    tone1_sum_by_date = df_filtered.groupby('DATE')['tone1'].sum()

    # Set the ggplot style
    plt.style.use('ggplot')

    # Plotting as a time series line plot with ggplot style
    plt.figure(figsize=(14, 7))  # Adjust the figure size as necessary
    tone1_sum_by_date.plot(kind='line', color='skyblue', marker='o', linewidth=2)

    plt.xlabel('Date')
    plt.ylabel('Sum of Tone1')
    plt.title(f'Sum of Tone1 by Date - Time Series (Field Checked: {field_to_check})')

    # Improve readability of the x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()

import pandas as pd
import ast

def map_themes_to_categories(es_data, data_to_share, es_themes_column='THEMES', node_column='Node', category_column='Catagory/Colour'):
    # Function to safely parse a string representation of a list into a Python list
    def parse_themes(theme_str):
        try:
            return ast.literal_eval(theme_str)
        except Exception as e:
            print(f"Error parsing themes: {e}")
            return []

    # Function to map themes to categories based on a lookup dictionary
    def map_themes_to_categories(themes, node_to_category):
        categories = set()
        for theme in themes:
            if theme in node_to_category:
                categories.add(node_to_category[theme])
        return list(categories)

    # Load the datasets
    # Extract themes for each row in es_data
    es_data['THEMES_LIST'] = es_data[es_themes_column].apply(parse_themes)

    # Create a dictionary for fast lookup from Node to Category/Colour
    node_to_category = dict(zip(data_to_share[node_column].str.strip("'"), data_to_share[category_column]))

    # Apply the mapping function to each row's themes
    es_data['categories'] = es_data['THEMES_LIST'].apply(lambda x: map_themes_to_categories(x, node_to_category))

    return es_data

def transform_categories(dataframe, category_column):
    """
    Transforms a column containing categorical values separated by commas into multiple one-hot encoded columns.
    
    Parameters:
    - dataframe: A pandas DataFrame containing the data.
    - category_column: The name of the column containing the categories as a string.
    
    Returns:
    - A pandas DataFrame with the original data and additional columns for each unique category.
    """
    # Explode the category column into separate rows for each category
    exploded_data = dataframe.assign(categories=dataframe[category_column].str.split(', ')).explode(category_column)
    
    # Remove any leading/trailing spaces and quotes
    exploded_data[category_column] = exploded_data[category_column].str.strip("[]'\" ")
    
    # Get unique categories
    unique_categories = exploded_data[category_column].dropna().unique()
    
    # Create a new dataframe to hold one-hot encoding
    for category in unique_categories:
        dataframe[category] = exploded_data[category_column].apply(lambda x: 1 if x == category else 0)
    
    return dataframe

