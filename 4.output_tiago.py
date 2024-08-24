import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm

# Read the CSV file into a DataFrame
df = pd.read_csv(r'TrainingDataset\tiago_ev\medium\tiago_EV_medium_traffic_training_data_updated.csv')

# Mathematical model for range estimation
time = df['Time']  # time in seconds
fuel_efficiency = df['UsedCapacity'] * 1000 / df['DistanceKm']
estimated_range = df['PT.BattHV.Energy'] * 1000 / fuel_efficiency  # the estimated remaining range of vehicle from current situation
distance_covered = df['DistanceKm']  # distance covered by the vehicle at present
for_representation = distance_covered + estimated_range

plt.plot(time, estimated_range, label='Estimated remaining range by mathematical model', color='green')
plt.plot(time, distance_covered, label='Actual Distance covered by vehicle', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Mathematical model output.')
plt.legend()
plt.show()

plt.plot(time, for_representation, label='Estimated remaining range by mathematical model', color='green')
plt.plot(time, distance_covered, label='Actual Distance covered by vehicle', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Distance covered By Vehicle and Estimated Remaining Range of the Vehicle at that particular point')
plt.legend()
plt.show()

# Load all models upfront
models = {
    (0, 4): load_model(r"rnn_models\tiago_ev\model_remaining_capacity_0_to_4.h5"),
    (4, 9): load_model(r"rnn_models\tiago_ev\model_remaining_capacity_4_to_9.h5"),
    (9, 14): load_model(r"rnn_models\tiago_ev\model_remaining_capacity_9_to_14.h5"),
    (14, 19): load_model(r"rnn_models\tiago_ev\model_remaining_capacity_14_to_19.h5"),
    (19, 24): load_model(r"rnn_models\tiago_ev\model_remaining_capacity_19_to_23.h5")
}

output = []

# Extract input features (X) and normalize once
X = df.drop(columns=["target", "Time",]).values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Use tqdm to create a progress bar
progress_bar = tqdm(total=len(df), desc='Processing Rows')

# Iterate through rows
for index, row in df.iterrows():
    remaining_capacity = row['RemainingCapacity']

    # Find appropriate model
    for (low, high), model in models.items():
        if low <= remaining_capacity < high:
            break

    # Make prediction
    prediction = model.predict(X_reshaped[index:index+1]).flatten()
    output.append(prediction)

    # Update progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()

# After loop, convert output to numpy array
output = np.array(output).flatten()

# Plot predictions or perform further analysis
plt.plot(time, output, label='Prediction by RNN', color='green')
plt.xlabel('Time')
plt.ylabel('Prediction')
plt.show()

# Data fusion of the estimated range and predicted output
average_estimated_range_prediction = (for_representation * 0.15 + output * 0.90)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time, output, label='Prediction by RNN', color='green')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('RNN model output.')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, output, label='Prediction by RNN', color='green')
plt.plot(time, distance_covered, label='Distance covered', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('RNN model output.')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, output, label='Prediction by RNN', color='green')
plt.plot(time, for_representation, label='Estimated remaining range by mathematical model', color='purple')
plt.plot(time, distance_covered, label='Distance covered', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, output, label='Prediction by RNN')
plt.plot(time, for_representation, label='Estimated remaining range by mathematical model', color='red')
plt.plot(time, average_estimated_range_prediction, label='Fused output of Estimated Range and Predictions', color='green')
plt.plot(time, distance_covered, label='Distance covered', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

difference = average_estimated_range_prediction - distance_covered
plt.figure(figsize=(10, 6))
plt.plot(time, difference, label='Remaining predicted range after data fusion', color='green')
plt.plot(time, distance_covered, label='Actual Distance covered by vehicle', color='blue')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
