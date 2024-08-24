import os
import pandas as pd

# Define the root directory
root_dir = r"TrainingDataset\mg_comet"

# Define the folder names
folders = ['heavy', 'medium', 'mild']

# Define the file names

# Comet EV
file_names = [
    'remaining_capacity_0_to_3.csv',
    'remaining_capacity_3_to_6.csv',
    'remaining_capacity_6_to_10.csv',
    'remaining_capacity_10_to_13.csv',
    'remaining_capacity_13_to_17.csv'
]

# Tiago EV
# file_names = [
#     'remaining_capacity_0_to_4.csv',
#     'remaining_capacity_4_to_9.csv',
#     'remaining_capacity_9_to_14.csv',
#     'remaining_capacity_14_to_19.csv',
#     'remaining_capacity_19_to_24.csv'
# ]

# Loop through each file name
for file_name in file_names:
    # List to hold data from each folder
    dfs = []
    
    # Loop through each folder
    for folder in folders:
        file_path = os.path.join(root_dir, folder, file_name)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Concatenate the dataframes along the rows
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Save the merged dataframe to a new CSV file
    merged_file_name = f'{file_name}'
    merged_file_path = os.path.join('TrainingDataset\mg_comet\merged', merged_file_name)
    merged_df.to_csv(merged_file_path, index=False)

    print(f'Merged file saved as: {merged_file_path}')
