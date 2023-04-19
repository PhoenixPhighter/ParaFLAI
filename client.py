import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision
from pathlib import Path
import argparse
from torch.utils.data import DataLoader

import flmodel
import util

USE_FEDBN: bool = True

# pylint: disable=no-member
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


# Flower Client
class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split

    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = flmodel.load_efficientnet(classes=10)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )

        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)

        results = flmodel.train(model, trainLoader, valLoader, epochs, self.device)

        parameters_prime = flmodel.get_model_params(model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=16)

        loss, accuracy = flmodel.test(model, testloader, steps, self.device)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--id", type=int, default=0, required=True, help="Participant ID"
    )

    parser.add_argument(
        "--ncli", type=int, default=1, required=True, help="Number of clients"
    )

    parser.add_argument(
        "--ip", type=str, default="127.0.0.1", required=False, help="Server IP"
    )

    args = parser.parse_args()

    # Load data
    trainset, testset = util.load_partition(args.id, args.ncli)

    client = CifarClient(trainset, testset, DEVICE)
    fl.client.start_numpy_client(server_address=f"{args.ip}:8080", client=client)


if __name__ == "__main__":
    main()
