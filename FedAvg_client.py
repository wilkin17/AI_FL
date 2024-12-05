import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

train_df = pd.read_csv(r".\Train_Aggregated.csv")
test_df = pd.read_csv(r".\Test_Aggregated.csv")

# Parameters
data_per_client = 247  # Training data points per client.
test_data_points = 102  # Number of test data points.
sequence_length = 50
prediction_length = 50

# Calculate the number of clients
train_data_size = len(train_df)
num_clients = train_data_size // data_per_client

# Verify that we have enough data
total_required_data = num_clients * data_per_client
if train_data_size < total_required_data:
    raise ValueError(f"Not enough training data. Required: {total_required_data}, Available: {train_data_size}")

device_evaluation_df = pd.DataFrame(columns=["client_id", "loss"])
client_loss_data = {}

def dynamic_weight_decay(loss, base_decay=0.01, min_decay=0.0001, max_decay=0.1):
    """
    Adjust weight decay based on the loss.
    Higher loss leads to higher regularization (to prevent overfitting).
    Lower loss leads to lower regularization (to allow better fitting).
    
    Args:
        loss (float): Current loss value.
        base_decay (float): Base weight decay value.
        min_decay (float): Minimum weight decay allowed.
        max_decay (float): Maximum weight decay allowed.
        
    Returns:
        float: Adjusted weight decay value.
    """
    # Adjust weight decay inversely with the loss (clipped to min/max bounds)
    decay = base_decay * (1 + loss)
    decay = np.clip(decay, min_decay, max_decay)
    return decay

# Custom Dataset for Time Series Data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=sequence_length, prediction_length=prediction_length):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.data = data.reset_index(drop=True)
        self.X, self.y = self._prepare_data()

    def _prepare_data(self):
        data_array = self.data.values
        X, y = [], []
        for i in range(len(data_array) - self.sequence_length - self.prediction_length + 1):
            X.append(data_array[i:i + self.sequence_length])
            y.append(
                data_array[
                    i + self.sequence_length : i + self.sequence_length + self.prediction_length, -1
                ]
            )  # Assuming target is last column
        X = np.array(X)
        y = np.array(y)
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, prediction_length):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, prediction_length)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size
        ).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last output
        out = self.fc(out)
        return out

# Federated Learning Client
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_data):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.input_size = client_data.shape[1]  # Number of features
        self.hidden_size = 64
        self.num_layers = 2

        # Initialize the model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            prediction_length=self.prediction_length,
        ).to(self.device)

        # Prepare datasets and dataloaders
        self.train_dataset = TimeSeriesDataset(client_data, self.sequence_length, self.prediction_length)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)

        # Define loss function (MSE)
        self.criterion = nn.MSELoss()
        self.base_weight_decay = 0.01  # Base value for weight decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(np.copy(v), dtype=torch.float32) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        epoch_loss = 0.0

        # Adjust weight decay based on the current performance
        current_weight_decay = dynamic_weight_decay(epoch_loss)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=current_weight_decay)

        for epoch in range(1):  # Single epoch per round
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

        # Return the updated model parameters
        return self.get_parameters(config), len(self.train_loader.dataset), {"weight_decay": current_weight_decay}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.model(X_batch)
                loss += self.criterion(output, y_batch).item() * X_batch.size(0)
                total += X_batch.size(0)
        loss /= total
        return float(loss), total, {"MSE": float(loss)}

# Function to create client instances
def client_fn(cid: str):
    # Extract client ID from cid
    client_id = int(cid)

    # Validate client_id
    if client_id >= num_clients:
        raise ValueError(f"Invalid client_id {client_id}. It should be less than {num_clients}.")

    # Calculate the start and end indices for this client
    start_idx = client_id * data_per_client
    end_idx = start_idx + data_per_client

    # Ensure we don't exceed the dataset length
    if end_idx > len(train_df):
        end_idx = len(train_df)

    # Slice the data for this client
    client_data = train_df.iloc[start_idx:end_idx].reset_index(drop=True)

    # Handle the case where there might be less data than expected
    if len(client_data) < data_per_client:
        print(f"Client {client_id} has {len(client_data)} data points (less than expected).")

    # Debugging output
    print(f"Client {client_id} has {len(client_data)} training data points.")

    # Create and return a client instance with this data
    return FLClient(client_data)

# Prepare test data
test_data_size = len(test_df)
if test_data_size >= test_data_points:
    test_data = test_df.iloc[:test_data_points].reset_index(drop=True)
else:
    raise ValueError(f"Not enough test data. Required: {test_data_points}, Available: {test_data_size}")

# Prepare global test dataset
test_dataset = TimeSeriesDataset(
    test_data, sequence_length=sequence_length, prediction_length=prediction_length
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

# Define a function for server-side evaluation
def get_evaluate_fn(model):
    def evaluate(server_round, parameters, config):
        global client_loss_data

        # Update the model with the current global parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(np.copy(v), dtype=torch.float32) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Simulate loss evaluation for each client
        model.eval()
        for client_id in range(num_clients):  # Iterate over all possible clients
            try:
                client = client_fn(str(client_id))  # Get the client
                client_loss = 0.0
                total_samples = 0

                # Evaluate the client's model on its own dataset
                with torch.no_grad():
                    for X_batch, y_batch in client.train_loader:
                        X_batch = X_batch.to(model.device)
                        y_batch = y_batch.to(model.device)
                        output = model(X_batch)
                        client_loss += nn.MSELoss()(output, y_batch).item() * X_batch.size(0)
                        total_samples += X_batch.size(0)

                # Compute the average loss for this round
                if total_samples > 0:
                    round_loss = client_loss / total_samples
                else:
                    round_loss = float("nan")

                # Update the cumulative loss data
                if client_id not in client_loss_data:
                    client_loss_data[client_id] = {"total_loss": 0.0, "rounds": 0}
                client_loss_data[client_id]["total_loss"] += round_loss
                client_loss_data[client_id]["rounds"] += 1

            except ValueError as e:
                # If a client does not exist (not selected), skip it
                print(f"Client {client_id} not included in this round: {e}")

        return None, {}  # No need to return global metrics since it's per-client loss
    return evaluate

# After the simulation, save the averaged client losses to a CSV file
def save_averaged_device_losses(filename="averaged_device_losses.csv"):
    global client_loss_data

    # Compute the average loss for each client
    averaged_data = []
    for client_id, data in client_loss_data.items():
        avg_loss = data["total_loss"] / data["rounds"] if data["rounds"] > 0 else float("nan")
        averaged_data.append({"client_id": client_id, "average_loss": avg_loss})

    # Create a DataFrame and save it to CSV
    averaged_df = pd.DataFrame(averaged_data)
    averaged_df.to_csv(filename, index=False)

# Initialize the global model
global_model = LSTMModel(
    input_size=train_df.shape[1],
    hidden_size=64,
    num_layers=2,
    prediction_length=prediction_length
)
global_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model.to(global_model.device)

# Start the simulation
if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=fl.server.strategy.FedAvg(
            evaluate_fn=get_evaluate_fn(global_model)
        ),
    )

    # Save the averaged losses to a CSV file
    save_averaged_device_losses("fedavg_averaged_device_losses.csv")
