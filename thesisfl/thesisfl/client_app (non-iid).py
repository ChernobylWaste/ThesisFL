"""thesisfl: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from thesisfl.task import load_data, load_model
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import time 

class FlowerClient(NumPyClient):
    def __init__(self, model, data, epochs, batch_size, verbose):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, parameters, config):
        """Melatih model dan mencatat waktu pelatihan."""
        self.model.set_weights(parameters)

        start_time = time.time()

        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        end_time = time.time()

        train_loss = history.history["loss"][-1]
        train_accuracy = history.history["accuracy"][-1]
        training_time = end_time - start_time

        return self.model.get_weights(), len(self.x_train), {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "training_time" : training_time
        }

    def evaluate(self, parameters, config):
        """Mengevaluasi model yang sudah di training dan dibuatkan metrics."""
        self.model.set_weights(parameters)
        
        start_time = time.time()  

        y_pred_prob = self.model.predict(self.x_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        # time.sleep(1.0)

        end_time = time.time()

        test_time = end_time - start_time

        return loss, len(self.x_test), {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "test_time": test_time
        }

def client_fn(context: Context):
    # Load model and dataset
    model = load_model()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    
    # Training parameters
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    return FlowerClient(model, data, epochs, batch_size, verbose).to_client()

app = ClientApp(client_fn=client_fn)
