import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf


# List of CSV file paths
file_paths = [
    'TrainingDataset\\tiago_ev\\merged\\remaining_capacity_0_to_4.csv',
    'TrainingDataset\\tiago_ev\\merged\\remaining_capacity_4_to_9.csv',
    'TrainingDataset\\tiago_ev\\merged\\remaining_capacity_9_to_14.csv',
    'TrainingDataset\\tiago_ev\\merged\\remaining_capacity_14_to_19.csv',
    'TrainingDataset\\tiago_ev\\merged\\remaining_capacity_19_to_24.csv'    # Add more file paths as needed
]

# Function to train RNN model for each dataset
def train_rnn_model(file_path, model_save_dir):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Split the dataset into input features (X) and target variable (y)
    X = df.drop(columns=["target", "Time"]).values
    y = df["target"].values
    
    # Normalize the input features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape the input features for the RNN model
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
    
    # Define the RNN model
    model = Sequential([
        LSTM(units=12, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=30, return_sequences=True),
        Dropout(0.2),
        LSTM(units=30),
        Dropout(0.2),
        Dense(units=1)
    ])

#     model = Sequential([
#     tf.compat.v1.keras.layers.GRU(units=12, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#     Dropout(0.2),
#     tf.compat.v1.keras.layers.GRU(units=30, return_sequences=True),
#     Dropout(0.2),
#     tf.compat.v1.keras.layers.GRU(units=30),
#     Dropout(0.2),
#     Dense(units=1)
# ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    checkpoint_path = os.path.join(model_save_dir, f"model_{os.path.basename(file_path).replace('.csv', '.h5')}")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    history = model.fit(X_train, y_train, epochs=50, batch_size=20, validation_data=(X_test, y_test), callbacks=[checkpoint])
    
    # Print confirmation message
    print(f"Trained RNN model for {os.path.basename(file_path)} saved to {checkpoint_path}")
    
    # Export the trained model
    model.save(checkpoint_path)
    print(f"Trained model exported successfully to {checkpoint_path}")
    
    # Plot accuracy vs iterations
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss vs Iterations for {os.path.basename(file_path)}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Directory to save trained models
model_save_dir = "rnn_models\\tiago_ev"

# Ensure the directory exists or create it
os.makedirs(model_save_dir, exist_ok=True)

# Train RNN model for each dataset in the list
for file_path in file_paths:
    train_rnn_model(file_path, model_save_dir)
