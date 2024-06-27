#%%
import os
import pandas as pd
import ast

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

# # Usage example
# file_path = '/path/to/your/csvfile.csv'
# cleaned_data = transform_csv(file_path)

def filter_columns(input_csv_path, output_csv_path):
    data = pd.read_csv(input_csv_path)
    columns_to_keep = ['DATE', 'tone1', 'SOURCES', 'SOURCEURLS', 'country_code', 'THEMES']
    filtered_data = data[columns_to_keep]
    filtered_data.to_csv(output_csv_path, index=False)
    return 
    
# %%
