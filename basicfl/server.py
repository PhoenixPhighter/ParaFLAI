from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import argparse
import torch
import flwr as fl
import cifar
import numpy as np

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    _, testloader, _ = cifar.load_partition(0, 1)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        model.train()
        keys = [k for k in model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)
        loss, accuracy = cifar.test(model, testloader, device=DEVICE)
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate


if __name__ == "__main__":
    model = cifar.Net().to(DEVICE)

    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--ncli", type=int, default=1, required=True, help="Number of clients"
    )

    args = parser.parse_args()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_evaluate=0.2,
            min_fit_clients=args.ncli,
            min_evaluate_clients=2,
            min_available_clients=args.ncli,
            evaluate_fn=get_evaluate_fn(model),
            # on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        ),
    )
