"""thesisfl: A Flower / TensorFlow app."""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATASET_PATH = "/home/mbc/thesissatria/Dataset/CICIoT2023_IPCASTD10%.csv"

def load_data(partition_id, num_partitions):
    """Memuat dataset, membagi menjadi partisi untuk FL, dan melakukan preprocessing."""
    df = pd.read_csv(DATASET_PATH)

    df.drop_duplicates(inplace=True)

    label_map = {
        'BENIGN': 0, 'DDoS': 1, 'DoS': 2, 'Mirai': 3,
        'Network Attack': 4, 'Recon': 5, 'Brute Force': 6,
        'Injection': 7, 'Malware': 8,
    }
    df["Attack Type"] = df["Attack Type"].map(label_map)

    # Stratified IID Split: Pembagian Label merata (50:50) ke setiap client 
    client_data = [[] for _ in range(num_partitions)]
    for label in df["Attack Type"].unique():
        df_label = df[df["Attack Type"] == label]
        df_label = df_label.sample(frac=1).reset_index(drop=True) # Pengacakan Index Setiap Round
        splits = np.array_split(df_label, num_partitions)
        for i in range(num_partitions):
            client_data[i].append(splits[i])

    # Menggabungkan data yg sudah di split ke client
    df_client = pd.concat(client_data[partition_id]).reset_index(drop=True)

    # Pisahkan fitur dan label kembali
    X = df_client.drop(columns=["Attack Type"])
    y = df_client["Attack Type"]

    # Normalisasi menggunakan StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_train_cat = to_categorical(y_train, num_classes=9)
    y_test_cat = to_categorical(y_test, num_classes=9)

    return X_train, y_train_cat, X_test, y_test_cat

def load_model():
    """Membangun dan mengembalikan model DNN."""
    model = Sequential([
        Dense(units=128, activation='relu', input_dim=23),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=9, activation='softmax')  # Output layer untuk 9 kelas
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

