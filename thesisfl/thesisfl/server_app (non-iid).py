"""thesisfl: A Flower / TensorFlow app."""

import time
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy import FedProx
from thesisfl.task import load_model

def aggregate_evaluate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Menghitung rata-rata loss, akurasi, precision, recall, dan F1-score global."""

    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    f1_scores = [num_examples * m["f1_score"] for num_examples, m in metrics]

    # Memastikan `test_time` ada di setiap metrik, jika tidak, set default 0
    test_times = [m.get("test_time", 0) for _, m in metrics]  

    global_loss = sum(losses) / total_examples
    global_accuracy = sum(accuracies) / total_examples
    global_precision = sum(precisions) / total_examples
    global_recall = sum(recalls) / total_examples
    global_f1 = sum(f1_scores) / total_examples

    avg_test_time = sum(test_times) / len(test_times)

    log_msg = (f"[Global Evaluation] - Loss: {global_loss:.4f}, "
               f"Accuracy: {global_accuracy:.4f}, "
               f"Precision: {global_precision:.4f}, "
               f"Recall: {global_recall:.4f}, "
               f"F1-Score: {global_f1:.4f}, "
               f"Avg Test Time: {avg_test_time:.4f} seconds\n\n\n")

    print(log_msg)
    with open("results.txt", "a") as log_file:
        log_file.write(log_msg)

    return {
        "loss": global_loss,
        "accuracy": global_accuracy,
        "precision": global_precision,
        "recall": global_recall,
        "f1_score": global_f1,
        "avg_test_time": avg_test_time
    }

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Mencatat hasil training tiap client."""
    with open("results.txt", "a") as log_file:
        for client_id, (num_examples, m) in enumerate(metrics, start=1):
            log_msg = (f"Client {client_id} - Train Loss: {m['train_loss']:.4f}, "
                       f"Train Accuracy: {m['train_accuracy']:.4f}, "
                       f"Training Time: {m['training_time']:.4f} seconds, "
                       f"Samples: {num_examples}\n" )
            print(log_msg, end="")
            log_file.write(log_msg)
    return {}

def handle_evaluate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Mencatat hasil testing tiap client."""
    with open("results.txt", "a") as log_file:
        for client_id, (num_examples, m) in enumerate(metrics, start=1):
            log_msg = (f"Client {client_id} - Test Loss: {m['loss']:.4f}, "
                       f"Test Accuracy: {m['accuracy']:.4f}, "
                       f"Test Time: {m['test_time']:.4f} seconds, "
                       f"Samples: {num_examples}\n")
            print(log_msg, end="")
            log_file.write(log_msg)

    return aggregate_evaluate_metrics(metrics)

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model with weights
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define the federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=handle_evaluate_metrics,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        # proximal_mu=1.0
    )
    config = ServerConfig(num_rounds=num_rounds)


    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

