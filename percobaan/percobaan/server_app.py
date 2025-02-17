"""percobaan: A Flower / TensorFlow app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from percobaan.task import load_model

def server_fn(context: Context):
    # Read number of rounds from configuration
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model with weights
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define the federated learning strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)

