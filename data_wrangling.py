import os
import pandas as pd
import ast

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
    """
    Cleans a column by removing square brackets and single quotes.
    """
    return column.str.replace(r"[\[\]']", '', regex=True)

def transform_csv(file_path):
    """
    Transforms a CSV file into a DataFrame with specific formatting.
    - Removes square brackets and single quotes from certain columns
    - Separates the 'TONE' column into individual columns based on dictionary keys
    """
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
    
    # Separate the TONE column into individual columns based on the dictionary keys
    if 'TONE' in data.columns:
        tone_data = data['TONE'].apply(ast.literal_eval).apply(pd.Series)
        data = pd.concat([data.drop(columns=['TONE']), tone_data], axis=1)
    
    return data

# # Usage example
# file_path = '/path/to/your/csvfile.csv'
# cleaned_data = transform_csv(file_path)
