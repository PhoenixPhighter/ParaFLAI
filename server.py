from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
import torch

import util
import flmodel
import warnings

warnings.filterwarnings("ignore")

def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = util.load_data()

    n_train = len(trainset)
    # Use the last 1k training examples as a validation set
    valset = torch.utils.data.Subset(trainset, range(n_train - 1000, n_train))

    valLoader = DataLoader(valset, batch_size=16)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        print('start eval')
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        print('start test')

        loss, accuracy = flmodel.test(model, valLoader)
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--ncli", type=int, default=1, required=True, help="Number of clients"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, required=True, help="Number of rounds to run"
    )


    args = parser.parse_args()
    model = flmodel.load_efficientnet(classes=10)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_evaluate=0.2,
        min_fit_clients=args.ncli,
        min_evaluate_clients=2,
        min_available_clients=args.ncli,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.epochs),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
