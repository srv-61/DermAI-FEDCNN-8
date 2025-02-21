import flwr as fl

# Define the Federated Averaging strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,  # 50% of clients participate in each round
    min_fit_clients=2,  # At least 2 clients per round
    min_evaluate_clients=2,
    min_available_clients=2,  # Total clients required
)

if __name__ == "__main__":
    try:
        fl.server.start_server(
            server_address="0.0.0.0:9091",  # Allows external clients to connect
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy,
        )
    except Exception as e:
        print(f"Server error: {e}")
