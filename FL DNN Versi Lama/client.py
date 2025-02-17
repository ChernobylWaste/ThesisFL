import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import socket
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Load dataset
file_path = '/home/mbc/thesissatria/FL/CICIOT2023_processed.csv'
dataset = pd.read_csv(file_path)

# Label encoding
label_map = {
    'BENIGN': 0, 'DDoS': 1, 'DoS': 2, 'Mirai': 3, 'Network Attack': 4,
    'Recon': 5, 'Brute Force': 6, 'Injection': 7, 'Malware': 8
}
dataset['Attack Type'] = dataset['Attack Type'].map(label_map)

# Split features and labels
X = dataset.drop('Attack Type', axis=1).astype('float64')
y = dataset['Attack Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encoding labels
y_train_cat = to_categorical(y_train, num_classes=9)
y_test_cat = to_categorical(y_test, num_classes=9)

# Define the DNN model
def create_dnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Initialize the model
model = create_dnn_model(input_shape=X_train.shape[1], num_classes=9)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Get unique identifier for the client
client_id = socket.gethostname()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train_cat, epochs=5, batch_size=64, verbose=1)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Compute confusion matrix
        y_pred = np.argmax(model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
        plt.title(f'Confusion Matrix - {client_id}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f"confusion_matrix_{client_id}.png")
        plt.close()
        
        return loss, len(X_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
