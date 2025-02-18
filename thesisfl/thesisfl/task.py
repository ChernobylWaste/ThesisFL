"""thesisfl: A Flower / TensorFlow app."""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reduce TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load CICIoT2023 dataset
DATASET_PATH = "/home/mbc/thesissatria/CICIOT2023_processed.csv"

def load_data(partition_id, num_partitions):
    dataset = pd.read_csv(DATASET_PATH)

    # Mapping class labels to numbers
    label_map = {
        'BENIGN': 0, 'DDoS': 1, 'DoS': 2, 'Mirai': 3,
        'Network Attack': 4, 'Recon': 5, 'Brute Force': 6,
        'Injection': 7, 'Malware': 8,
    }
    dataset['Attack Type'] = dataset['Attack Type'].map(label_map)

    # Splitting features and labels
    X = dataset.drop(columns=['Attack Type']).astype('float64')
    y = dataset['Attack Type']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert labels to categorical format
    y_train_cat = to_categorical(y_train, num_classes=9)
    y_test_cat = to_categorical(y_test, num_classes=9)

    return X_train, y_train_cat, X_test, y_test_cat

def load_model():
    # Define the DNN model for CICIoT2023
    model = Sequential([
        Dense(units=128, activation='relu', input_dim=23),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=9, activation='softmax')  # Output layer for 9 classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model