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

    # df.drop_duplicates(inplace=True)

    label_map = {
        'BENIGN': 0, 'DDoS': 1, 'DoS': 2, 'Mirai': 3,
        'Network Attack': 4, 'Recon': 5, 'Brute Force': 6,
        'Injection': 7, 'Malware': 8,
    }
    df["Attack Type"] = df["Attack Type"].map(label_map)

    features = df.drop(columns=["Attack Type"]).astype("float64")
    labels = df["Attack Type"]

    # # Normalisasi menggunakan StandardScaler
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)

    # # Mengacak urutan dataset agar setiap round berbeda beda
    # df = df.sample(frac=1,random_state=np.random.randint(0,10000)).reset_index(drop=True)

    # Gabungkan fitur dan label ke satu DataFrame
    df_all = pd.DataFrame(features)
    df_all["Attack Type"] = labels.values

    # Stratified IID Split: Bagi per label ke semua client
    client_data = [[] for _ in range(num_partitions)]
    for label in df_all["Attack Type"].unique():
        df_label = df_all[df_all["Attack Type"] == label]
        df_label = df_label.sample(frac=1, random_state=42).reset_index(drop=True)
        splits = np.array_split(df_label, num_partitions)
        for i in range(num_partitions):
            client_data[i].append(splits[i])

    # Gabungkan data dari semua kelas untuk client ini
    df_client = pd.concat(client_data[partition_id]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Pisahkan fitur dan label kembali
    X = df_client.drop(columns=["Attack Type"]).values
    y = df_client["Attack Type"].values

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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

