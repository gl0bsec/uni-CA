import os
import pandas as pd

def count_rows_and_average_column_by_category(directory: str, index_column: str, average_column: str, category_column: str) -> pd.DataFrame:
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

def count_and_average_by_category_single_file(file_path: str, index_column: str, average_column: str, category_column: str) -> pd.DataFrame:
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
