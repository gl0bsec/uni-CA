#%%
import os
import pandas as pd
import ast

def process_csv(input_file_path, output_file_path):
    # Define the list of stopwords
    stopwords = set([
        'i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
        'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
        'dms', 'nytimes', 'bbc', 'apnews', 'outlookindia', 'two', '10', '000', '##'
    ])

    # Load the input CSV file
    data = pd.read_csv(input_file_path)

    # Create a list to store the new rows without stopwords
    filtered_source_target_data = []

    for index, row in data.iterrows():
        if pd.notna(row['entities']) and pd.notna(row['entity_types']):
            entities = row['entities'].split(', ')
            entity_types = row['entity_types'].split(', ')
            
            # Filter out stopwords and ensure the lengths match
            filtered_entities = []
            filtered_entity_types = []
            for entity, entity_type in zip(entities, entity_types):
                if entity.lower() not in stopwords:
                    filtered_entities.append(entity)
                    filtered_entity_types.append(entity_type)
            
            # Ensure entities and entity_types have the same length
            min_length = min(len(filtered_entities), len(filtered_entity_types))
            
            if min_length > 1:
                for i in range(min_length - 1):
                    new_row = row.to_dict()
                    new_row.update({
                        'source_entity': filtered_entities[i],
                        'source_type': filtered_entity_types[i],
                        'target_entity': filtered_entities[i+1],
                        'target_type': filtered_entity_types[i+1]
                    })
                    filtered_source_target_data.append(new_row)

    # Convert the list of dictionaries to a new DataFrame
    filtered_source_target_df = pd.DataFrame(filtered_source_target_data)

    # Reformat the timestamp column
    filtered_source_target_df['timestamp'] = pd.to_datetime(filtered_source_target_df['timestamp'], errors='coerce', utc=True)
    filtered_source_target_df['timestamp'] = filtered_source_target_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')

    # Save the enhanced DataFrame to a new CSV file
    filtered_source_target_df.to_csv(output_file_path, index=False)
    return output_file_path

def GDELT_valid(updated_tables_path, fips_codes_path, treaty_acronyms=None):
    # Load the CSV files
    updated_merged_tables = pd.read_csv(updated_tables_path)
    fips_country_codes = pd.read_csv(fips_codes_path)

    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str.strip(), format='%d %b %Y', errors='coerce')
        except:
            return pd.NaT

    # Add parsed date columns to the dataset
    updated_merged_tables['Parsed Ratification Date'] = updated_merged_tables[
        'Ratification Date, Accession(a), Succession(d) Date'
    ].apply(parse_date)
    updated_merged_tables['Parsed Signature Date'] = updated_merged_tables['Signature Date'].apply(parse_date)

    # Filter dates after 2013
    filtered_df = updated_merged_tables[updated_merged_tables['Parsed Ratification Date'] > '2013-12-31']
    
    # Merge with FIPS country codes
    merged_df = pd.merge(filtered_df, fips_country_codes, left_on='COUNTRY', right_on='Name', how='left')
    
    # Exclude results where the country code is not found
    merged_df = merged_df.dropna(subset=['FIPS 10-4'])
    
    # Filter by treaty acronyms if provided
    if treaty_acronyms is not None:
        merged_df = merged_df[merged_df['Treaty Name'].isin(treaty_acronyms)]
    
    # Create list of quadruples (FIPS, Treaty Acronym, Ratification Date, Signature Date)
    quadruples = list(merged_df.apply(lambda row: (
        row['FIPS 10-4'], 
        row['Treaty Name'], 
        row['Parsed Ratification Date'].strftime('%d/%m/%Y'),
        row['Parsed Signature Date'].strftime('%d/%m/%Y') if pd.notna(row['Parsed Signature Date']) else 'N/A'
    ), axis=1))
    
    return quadruples

def count_rows_average_column_category(directory: str, index_column: str, average_column: str, category_column: str) -> pd.DataFrame:
    # List to hold the data
    data = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Full path of the file
            file_path = os.path.join(directory, filename)
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Count the number of rows
            row_count = len(df)
            # Check if the necessary columns exist in the DataFrame
            if index_column in df.columns and average_column in df.columns and category_column in df.columns:
                # Get the unique combinations of index_column and category_column
                unique_combinations = df[[index_column, category_column]].drop_duplicates()
                for _, row in unique_combinations.iterrows():
                    index_value = row[index_column]
                    category_value = row[category_column]
                    # Filter the DataFrame by the current unique combination
                    filtered_df = df[(df[index_column] == index_value) & (df[category_column] == category_value)]
                    # Extract the first number from the average_column strings and convert to numeric
                    numbers = filtered_df[average_column].str.split(',').str[0].astype(float)
                    average_value = numbers.mean() if not numbers.empty else None
                    data.append({
                        'IndexValue': index_value, 
                        'CategoryValue': category_value,
                        'RowCount': len(filtered_df), 
                        'AverageValue': average_value
                    })
            else:
                print(f"Columns {index_column}, {average_column}, or {category_column} not found in {filename}")

    # Create a DataFrame from the data
    result_df = pd.DataFrame(data)
    
    return result_df

# Example usage
# directory = 'path_to_your_directory'
# index_column = 'your_index_column'
# average_column = 'your_average_column'
# category_column = 'your_category_column'
# result_df = count_rows_and_average_column_by_category(directory, index_column, average_column, category_column)
# print(result_df)

def count_average_category_file(file_path: str, index_column: str, average_column: str, category_column: str) -> pd.DataFrame:
    # List to hold the data
    data = []

    # Read the CSV file
    df = pd.read_csv(file_path)
    # Count the number of rows
    row_count = len(df)
    # Check if the necessary columns exist in the DataFrame
    if index_column in df.columns and average_column in df.columns and category_column in df.columns:
        # Get the unique combinations of index_column and category_column
        unique_combinations = df[[index_column, category_column]].drop_duplicates()
        for _, row in unique_combinations.iterrows():
            index_value = row[index_column]
            category_value = row[category_column]
            # Filter the DataFrame by the current unique combination
            filtered_df = df[(df[index_column] == index_value) & (df[category_column] == category_value)]
            # Extract the first number from the average_column strings and convert to numeric
            numbers = filtered_df[average_column].str.split(',').str[0].astype(float)
            average_value = numbers.mean() if not numbers.empty else None
            data.append({
                'IndexValue': index_value, 
                'CategoryValue': category_value,
                'RowCount': len(filtered_df), 
                'AverageValue': average_value
            })
    else:
        print(f"Columns {index_column}, {average_column}, or {category_column} not found in the file")

    # Create a DataFrame from the data
    result_df = pd.DataFrame(data)
    
    return result_df

# Example usage
# file_path = 'path_to_your_file.csv'
# index_column = 'your_index_column'
# average_column = 'your_average_column'
# category_column = 'your_category_column'
# result_df = count_and_average_by_category_single_file(file_path, index_column, average_column, category_column)
# print(result_df)

def es_count_average_category_single(file_path: str, index_column: str, average_column: str, category_column: str) -> pd.DataFrame:
    # List to hold the data
    data = []

    # Read the CSV file
    df = pd.read_csv(file_path)
    # Check if the necessary columns exist in the DataFrame
    if index_column in df.columns and average_column in df.columns and category_column in df.columns:
        # Get the unique combinations of index_column and category_column
        unique_combinations = df[[index_column, category_column]].drop_duplicates()
        for _, row in unique_combinations.iterrows():
            index_value = row[index_column]
            category_value = row[category_column]
            # Filter the DataFrame by the current unique combination
            filtered_df = df[(df[index_column] == index_value) & (df[category_column] == category_value)]
            # Extract the first number from the average_column strings and convert to numeric
            # Handle the specific format of the average column values
            numbers = filtered_df[average_column].apply(lambda x: float(x.split(',')[0]) if pd.notna(x) and ',' in x else None)
            average_value = numbers.mean() if not numbers.empty else None
            data.append({
                'IndexValue': index_value, 
                'CategoryValue': category_value,
                'RowCount': len(filtered_df), 
                'AverageValue': average_value
            })
    else:
        print(f"Columns {index_column}, {average_column}, or {category_column} not found in the file")

    # Create a DataFrame from the data
    result_df = pd.DataFrame(data)
    
    return result_df

# Example usage
# file_path = '/mnt/data/.ao.2.10.2019_test2.csv'
# index_column = 'NUMARTS'
# average_column = 'CAMEOEVENTIDS'
# category_column = 'DATE'
# result_df = count_and_average_by_category_single_file(file_path, index_column, average_column, category_column)
# result_df

def es_count_average_category_multiple(directory: str, index_column: str, average_column: str, category_column: str) -> pd.DataFrame:
    # List to hold the data
    data = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Full path of the file
            file_path = os.path.join(directory, filename)
            # Read the CSV file
            df = pd.read_csv(file_path)
            # Check if the necessary columns exist in the DataFrame
            if index_column in df.columns and average_column in df.columns and category_column in df.columns:
                # Get the unique combinations of index_column and category_column
                unique_combinations = df[[index_column, category_column]].drop_duplicates()
                for _, row in unique_combinations.iterrows():
                    index_value = row[index_column]
                    category_value = row[category_column]
                    # Filter the DataFrame by the current unique combination
                    filtered_df = df[(df[index_column] == index_value) & (df[category_column] == category_value)]
                    # Extract the first number from the average_column strings and convert to numeric
                    # Handle the specific format of the average column values
                    numbers = filtered_df[average_column].apply(lambda x: float(x.split(',')[0]) if pd.notna(x) and ',' in x else None)
                    average_value = numbers.mean() if not numbers.empty else None
                    data.append({
                        'IndexValue': index_value, 
                        'CategoryValue': category_value,
                        'RowCount': len(filtered_df), 
                        'AverageValue': average_value
                    })
            else:
                print(f"Columns {index_column}, {average_column}, or {category_column} not found in {filename}")

    # Create a DataFrame from the data
    result_df = pd.DataFrame(data)
    
    return result_df

# Example usage
# directory = '/path_to_your_directory'
# index_column = 'NUMARTS'
# average_column = 'CAMEOEVENTIDS'
# category_column = 'DATE'
# result_df = count_and_average_by_category_multiple_files(directory, index_column, average_column, category_column)
# print(result_df)

def clean_column(column):
    return column.str.replace(r"[\[\]']", '', regex=True)

def transform_csv(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # List of columns to clean
    columns_to_clean = [
        'THEMES', 'PERSONS', 'ORGANIZATIONS', 'CAMEOEVENTIDS', 'SOURCES',
        'country_code', 'adm1_code', 'latitude', 'longitude', 'feature_id',
        'location_type', 'location_name'
    ]
    
    # Clean the specified columns
    for col in columns_to_clean:
        if col in data.columns:
            data[col] = clean_column(data[col])
    
    # Format the DATE column
    if 'DATE' in data.columns:
        data['DATE'] = pd.to_datetime(data['DATE']).dt.strftime('%d/%m/%Y')
    
    # Separate the TONE column into individual columns based on the dictionary keys
    if 'TONE' in data.columns:
        tone_data = data['TONE'].apply(ast.literal_eval).apply(pd.Series)
        data = pd.concat([data.drop(columns=['TONE']), tone_data], axis=1)
    
    return data

def transform_json(file_path):
    # Load the CSV file
    data = pd.read_json(file_path)
    
    # List of columns to clean
    columns_to_clean = [
        'THEMES', 'PERSONS', 'ORGANIZATIONS', 'CAMEOEVENTIDS', 'SOURCES',
        'country_code', 'adm1_code', 'latitude', 'longitude', 'feature_id',
        'location_type', 'location_name'
    ]
    
    # Clean the specified columns
    for col in columns_to_clean:
        if col in data.columns:
            data[col] = clean_column(data[col])
    
    # Format the DATE column
    if 'DATE' in data.columns:
        data['DATE'] = pd.to_datetime(data['DATE']).dt.strftime('%d/%m/%Y')
    
    # Separate the TONE column into individual columns based on the dictionary keys
    if 'TONE' in data.columns:
        tone_data = data['TONE'].apply(ast.literal_eval).apply(pd.Series)
        data = pd.concat([data.drop(columns=['TONE']), tone_data], axis=1)
    
    return data

# # Usage example
# file_path = '/path/to/your/csvfile.csv'
# cleaned_data = transform_csv(file_path)

def filter_columns(input_csv_path, output_csv_path):
    data = pd.read_csv(input_csv_path)
    columns_to_keep = ['DATE', 'tone1', 'SOURCES', 'SOURCEURLS', 'country_code', 'THEMES']
    filtered_data = data[columns_to_keep]
    filtered_data.to_csv(output_csv_path, index=False)
    return 

def filter_columns_json(input_csv_path, output_csv_path):
    data = pd.read_json(input_csv_path)
    columns_to_keep = ['DATE', 'tone1', 'SOURCES', 'SOURCEURLS', 'country_code', 'THEMES']
    filtered_data = data[columns_to_keep]
    filtered_data.to_csv(output_csv_path, index=False)
    return 


def process_gdelt_json_with_locations(input_file_path, output_file_path):
    # Load the JSON file
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # Create a DataFrame from the JSON data
    df = pd.json_normalize(data)

    # Split the TONE field into individual columns
    tone_columns = ['TONE_AvgTone', 'TONE_PositiveScore', 'TONE_NegativeScore', 'TONE_Polarity', 'TONE_ActivityRefDensity', 'TONE_Gram']
    df[tone_columns] = df['TONE'].str.split(',', expand=True)

    # Process the LOCATIONS field to extract country codes
    df['LOCATIONS'] = df['LOCATIONS'].fillna('')
    df['CountryCodes'] = df['LOCATIONS'].apply(lambda x: ';'.join(set([loc.split('#')[2] for loc in x.split(';') if loc])))

    # Select the desired columns
    selected_columns = ['DATE', 'SOURCES', 'SOURCEURLS', 'CountryCodes', 'ORGANIZATIONS','THEMES'] + tone_columns
    df_selected = df[selected_columns]

    # Save the DataFrame to a CSV file
    df_selected.to_csv(output_file_path, index=False)

# %%
