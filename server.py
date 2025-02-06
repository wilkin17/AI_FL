import flwr as fl
import os

def start_server():
    # Prompt user for method    
    method_map = {"0": "FedAvg", "1": "FedDyn", "2": "IDA"}
    method = input("Enter the federated learning method (0: FedAvg, 1: FedDyn, 2: IDA): ").strip()
    while method not in method_map:
        print("Invalid selection. Choose 0 for FedAvg, 1 for FedDyn, or 2 for IDA.")
        method = input("Enter the federated learning method (0: FedAvg, 1: FedDyn, 2: IDA): ").strip()
    method = method_map[method]  # Convert choice to method name
    print(f"Using method: {method}")  # Confirm selection

    # Prompt user for number of rounds
    num_rounds = input("Enter the number of training rounds: ").strip()
    while not num_rounds.isdigit():
        print("Please enter a valid integer for the number of rounds.")
        num_rounds = input("Enter the number of training rounds: ").strip()
    num_rounds = int(num_rounds)

    # Prompt user for dataset selection
    default_dataset = "Noiseless"
    dataset_map = {"y": "Noisy", "n": "Noiseless"}
    dataset_input = input(f"Would you like to use a noisy dataset? (y/N) [default: {default_dataset}]: ").strip().lower()
    if dataset_input in dataset_map:
        dataset = dataset_map[dataset_input]
    else:
        dataset = default_dataset

    # Save parameters to a file to pass to the client
    with open("config.txt", "w") as f:
        f.write(f"{method}\n{num_rounds}\n{dataset}")

    # Start the server
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    start_server()
