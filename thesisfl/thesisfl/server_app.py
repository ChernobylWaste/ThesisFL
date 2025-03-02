"""thesisfl: A Flower / TensorFlow app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from thesisfl.task import load_model

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics :
    # Fungsi aggregate akurasi dari global model yang udah di terima ke server
    
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)

    return {"accuracy": sum(accuracies) / total_examples}

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    for client_id, (_, m) in enumerate(metrics):
        print(f"Client {client_id + 1} - Train Loss: {m['train_loss']:.4f}, Train Accuracy: {m['train_accuracy']:.4f}")
    return {}


# def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics :
#     for _,m in metrics:
#         print(m)
#     return{}

def server_fn(context: Context):
    # Read number of rounds from configuration
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model with weights
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

