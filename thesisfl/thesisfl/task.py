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

DATASET_PATH = "/home/mbc/thesissatria/Dataset/CICIOT2023 10%.csv"

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

    features = df.drop(columns=["Attack Type"]).astype("float64")
    labels = df["Attack Type"]

    # Normalisasi menggunakan StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Mengacak urutan dataset agar setiap round berbeda beda
    df = df.sample(frac=1,random_state=np.random.randint(0,10000)).reset_index(drop=True)

    # Partisi dataset sesuai jumlah client
    total_samples = len(df)
    partition_size = total_samples // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size

    x_partition = features[start_idx:end_idx]
    y_partition = labels[start_idx:end_idx]

    # Bagi menjadi training dan testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_partition, y_partition, test_size=0.2, random_state=42
    )

    # Ubah label ke format one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes=9)
    y_test_cat = to_categorical(y_test, num_classes=9)

    return x_train, y_train_cat, x_test, y_test_cat

def load_model():
    """Membangun dan mengembalikan model DNN."""
    model = Sequential([
        Dense(units=128, activation='relu', input_dim=46),
        Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=9, activation='softmax')  # Output layer untuk 9 kelas
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


