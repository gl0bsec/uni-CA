import os
import urllib.request
from zipfile import ZipFile
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import glob

def gen_dates(input_date, n):
    try:
        input_date = datetime.strptime(input_date, "%d/%m/%Y")
        date_range = []
        for i in range(n, 0, -1):
            past_date = input_date - timedelta(days=i)
            date_range.append(past_date.strftime("%Y%m%d"))  # Format as yyyymmdd
        date_range.append(input_date.strftime("%Y%m%d"))
        for i in range(1, n+1):
            succeeding_date = input_date + timedelta(days=i)
            date_range.append(succeeding_date.strftime("%Y%m%d"))  # Format as yyyymmdd

        return date_range
    except ValueError:
        return []

gen_dates('28/09/2010', 10)


def dates_from(delta, number):
    end_date = datetime.today() - timedelta(delta)
    start_date = end_date - timedelta(number)

    def date_range(start, end):
        delta = end - start  # as timedelta
        days = [start + timedelta(days=i) for i in range(delta.days + 1)]
        return days

    days = [str(day.strftime("%Y%m%d")) for day in date_range(start_date, end_date)]
    return days

def download_data(k=1, n=None, input_date=None, locations_regex=None, themes_regex=None):
    if input_date is None:
        date_range = dates_from(k, n)
    else:
        date_range = gen_dates(input_date, n)

    # Create a directory to store temporary JSON files
    temp_dir = 'big_dump/'
    os.makedirs(temp_dir, exist_ok=True)

    # Download, filter, and save GDELT data
    for day in tqdm(date_range, desc="Downloading GDELT data"):
        url = "http://data.gdeltproject.org/gkg/" + day + ".gkg.csv.zip"
        identifier = "ID"

        try:
            urllib.request.urlretrieve(url, temp_dir + "/" + "GEvents1" + day + ".zip")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            continue

        # Extract the file
        with ZipFile(temp_dir + "/" + "GEvents1" + day + ".zip", "r") as zipObj:
            zipObj.extractall(temp_dir)
        os.remove(temp_dir + "/" + "GEvents1" + day + ".zip")

        # Load the CSV data into a pandas DataFrame
        try:
            df = pd.read_csv(temp_dir + "/" + day + ".gkg.csv", delimiter="\t")
        except Exception as e:
            print(f"Failed to read CSV file for {day}: {e}")
            continue

        # Apply optional filters for themes using contains and regex
        if themes_regex is not None:
            identifier = identifier+themes_regex
            df = df[df['THEMES'].str.contains(themes_regex, case=False, na=False, regex=True)]

        # Apply optional filters for locations using contains and regex
        if locations_regex is not None:
            identifier = identifier+locations_regex
            df = df[df['LOCATIONS'].str.contains(locations_regex, case=False, na=False, regex=True)]

        # Save the filtered data as a temporary JSON file
        temp_json_file = os.path.join(temp_dir, f'GDELT_#{day}#{identifier}#.json')
        df.to_json(temp_json_file, orient='records')
        os.remove(temp_dir + "/" + day + ".gkg.csv")

    print('Done')


def combine_json_files(folder_path, output_file):
    # Create an empty DataFrame to hold all the data
    combined_df = pd.DataFrame()

    # Loop through all the files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)
            # Load the current JSON file into a DataFrame
            current_df = pd.read_json(file_path)
            # Append the data from the current file to the combined DataFrame
            combined_df = pd.concat([combined_df, current_df], ignore_index=True)

    # Convert the combined DataFrame back to JSON format
    combined_json = combined_df.to_csv()

    # Write the combined JSON data to the output file
    with open(output_file, 'w') as f:
        f.write(combined_json)

def convert_date_format(df, column_name):
    """
    Params: df (pandas.DataFrame) column_name (str)
    Returns: The DataFrame with the converted column (YYYYMMDD format to DD/MM/YYYY).
    """
    # Convert the column to string format in case it isn't
    df[column_name] = df[column_name].astype(str)

    # Convert from YYYYMMDD to YYYY-MM-DD to facilitate parsing
    df[column_name] = pd.to_datetime(df[column_name], format='%Y%m%d')

    # Convert to the desired DD/MM/YYYY format
    df[column_name] = df[column_name].dt.strftime('%d/%m/%Y')

    return df


# Usage example