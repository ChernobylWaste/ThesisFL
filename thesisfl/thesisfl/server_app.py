"""thesisfl: A Flower / TensorFlow app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from thesisfl.task import load_model

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Menghitung rata-rata akurasi berdasarkan jumlah sample per client."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]

    return {"accuracy": sum(accuracies) / total_examples}

def aggregate_evaluate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Menghitung rata-rata loss & akurasi dari hasil evaluasi (testing)."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]

    global_loss = sum(losses) / total_examples
    global_accuracy = sum(accuracies) / total_examples

    log_msg = (f"\n[Global Evaluation] - Test Loss: {global_loss:.4f}, "
               f"Test Accuracy: {global_accuracy:.4f}\n")

    print(log_msg)
    with open("results.txt", "a") as log_file:
        log_file.write(log_msg)

    return {"loss": global_loss, "accuracy": global_accuracy}

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Mencatat hasil training tiap client."""
    with open("results.txt", "a") as log_file:
        for client_id, (num_examples, m) in enumerate(metrics, start=1):
            log_msg = (f"Client {client_id} - Train Loss: {m['train_loss']:.4f}, "
                       f"Train Accuracy: {m['train_accuracy']:.4f}, Samples: {num_examples}\n")
            print(log_msg, end="")
            log_file.write(log_msg)
    return {}

def handle_evaluate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Mencatat hasil testing tiap client."""
    with open("results.txt", "a") as log_file:
        for client_id, (num_examples, m) in enumerate(metrics, start=1):
            log_msg = (f"Client {client_id} - Test Loss: {m['loss']:.4f}, "
                       f"Test Accuracy: {m['accuracy']:.4f}, Samples: {num_examples}\n")
            print(log_msg, end="")
            log_file.write(log_msg)

    return aggregate_evaluate_metrics(metrics)

def server_fn(context: Context):
    # Read number of rounds from configuration
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model with weights
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=handle_evaluate_metrics,
        fit_metrics_aggregation_fn=handle_fit_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

