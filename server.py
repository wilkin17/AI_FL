import flwr as fl

def start_server():
    fl.server.start_server(
        server_address="0.0.0.0:8081",  # Corrected server address
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=fl.server.strategy.FedAvg()
    )

if __name__ == "__main__":
    start_server()
