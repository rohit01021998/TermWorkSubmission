import pandas as pd

def split_and_save_dataset(file_path, output_dir, num_parts):
    """
    Splits the dataset based on remaining capacity percentage ranges and saves each part as a CSV file.

    Args:
    - file_path (str): Path to the input CSV file.
    - output_dir (str): Directory where CSV parts will be saved.
    - num_parts (int): Number of parts to split the dataset into.

    Returns:
    - None
    """
    # Read the dataset from CSV
    df = pd.read_csv(file_path)

    # Calculate the maximum remaining capacity
    max_remaining_capacity = df['RemainingCapacity'].max()

    # Calculate the percentage thresholds for splitting
    thresholds = [(i / num_parts * max_remaining_capacity, (i + 1) / num_parts * max_remaining_capacity) for i in range(num_parts)]

    # Create a list to hold dataframes for each part
    dataframes = []

    # Iterate over each threshold range
    for i, (lower, upper) in enumerate(thresholds):
        # Filter the dataframe for rows within the current range
        filtered_df = df[(df['RemainingCapacity'] <= upper) & (df['RemainingCapacity'] > lower)]
        dataframes.append(filtered_df)

        # Save each part as a CSV file
        part_file_path = f"{output_dir}remaining_capacity_{int(lower)}_to_{int(upper)}.csv"
        filtered_df.to_csv(part_file_path, index=False)
        print(f"Part {i+1} saved to: {part_file_path}")

    # Print confirmation message
    print("All parts saved successfully.")

# Example usage:

filepaths = [
    r'TrainingDataset\tiago_ev\heavy\tiago_EV_heavy_traffic_training_data_updated.csv',
    r'TrainingDataset\tiago_ev\medium\tiago_EV_medium_traffic_training_data_updated.csv',
    r'TrainingDataset\tiago_ev\mild\tiago_EV_mild_traffic_training_data_updated.csv',
    r'TrainingDataset\mg_comet\heavy\comet_EV_heavy_traffic_training_data_updated.csv',
    r'TrainingDataset\mg_comet\medium\comet_EV_medium_traffic_training_data_updated.csv',
    r'TrainingDataset\mg_comet\mild\comet_EV_mild_traffic_training_data_updated.csv'
    # Add more file paths as needed
]

output_path = [
    'TrainingDataset\\tiago_ev\\heavy\\',
    'TrainingDataset\\tiago_ev\\medium\\',
    'TrainingDataset\\tiago_ev\\mild\\',
    'TrainingDataset\\mg_comet\\heavy\\',
    'TrainingDataset\\mg_comet\\medium\\',
    'TrainingDataset\\mg_comet\\mild\\'
]

j = 0

for path in filepaths:
    if __name__ == "__main__":
        # Example parameters
        file_path = path
        output_dir = output_path[j]
        num_parts = 5  # Number of parts to split the dataset into

        # Call the function
        split_and_save_dataset(file_path, output_dir, num_parts)
        j+=1
