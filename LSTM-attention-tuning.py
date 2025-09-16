'''
    @Author: Konstantinos Vasili
    @Date: 07/15/2025



    @Description: This script performs random grid search hyperparameter tuning for an LSTM-based autoencoder model using PyTorch.

    USAGE:
    python LSTM-attention-tuning.py


'''

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from dataclasses import dataclass, field


# <------------------------ START ------------------------------>

@dataclass
class Config:
    batch_size: int = 64
    seq_len: int = 20
    learning_rate: float = 0.001
    scaler: str = "minmax_scaler_power_cycle"
    data_path: str = "./path/to/data"
    columns: list = field(default_factory=lambda: [
        "nfd-1-cps", "nfd-4-flux", "cont air counts", "ram-con-lvl", "ram-pool-lvl", "ram-wtr-lvl"])


def load_data(filename, cols_to_be_read, percentage):
    '''
        This function loads the data from a .csv file to a pandas dataframe.
        Percentage parameter defines the number of rows to be loaded
    '''

    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df = df.loc[:, cols_to_be_read]
    print(df.head())
    print(f"Shape of the data: {df.shape}")

    length = len(df)

    try:
        if 0 < percentage <= 1.0:

            number_of_rows = int(percentage*length)
            df = df[:number_of_rows]

            return df
    except:
        raise ValueError("values in percentage should be in the range (0, 1]")


def split_data(df, test_size=0.20):
    split_index = int(len(df) * (1-test_size))

    # return train, val
    return df[:split_index], df[split_index:]


def to_sequences(data, seq_size=10):
    """Convert data to sequences of given size."""
    sequences = []
    for i in range(len(data) - seq_size + 1):
        sequences.append(data[i:i + seq_size])
    return np.array(sequences)


def to_sequences_(x, seq_size):
    """
    Creates sequences from the data by checking if the index is consecutive.
    Only sequences with consecutive indices are included.
    """
    x_values = []
    index_values = x.iloc[:, 0].astype(int).values  # Ensure integer type

    for i in range(len(x) - seq_size + 1):
        seq_index = index_values[i:i + seq_size]
        # Check if all indices are consecutive
        if np.all(np.diff(seq_index) == 1):
            seq = x.iloc[i:(i + seq_size)]
            x_values.append(seq.drop(columns=["Index"]).values)

    return np.array(x_values)


def objective(trial):
    #  hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    seq_len = trial.suggest_categorical("seq_len", [10, 20, 30, 50])
    hidden1 = trial.suggest_categorical("hidden1", [64, 128, 256])
    hidden2 = trial.suggest_categorical("hidden2", [32, 64, 128])
    bottleneck = trial.suggest_categorical("bottleneck", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    #  Prepare data
    config = Config(seq_len=seq_len, batch_size=batch_size,
                    learning_rate=learning_rate)
    data = load_data(config.data_path, config.columns, 1.0)
    train, val = split_data(data)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)

    # Use to_sequences since data is consecutive and no Index column
    trainX = to_sequences(train_scaled, seq_size=config.seq_len)
    valX = to_sequences(val_scaled, seq_size=config.seq_len)

    train_dataset = TensorDataset(torch.tensor(trainX, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(valX, dtype=torch.float32))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #  Define model with suggested hidden dims
    class TuningLSTMAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden1, hidden2, bottleneck, dropout):
            super().__init__()
            self.encoder_lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.encoder_lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
            self.bottleneck_lstm = nn.LSTM(
                hidden2, bottleneck, batch_first=True)
            self.decoder_lstm1 = nn.LSTM(bottleneck, hidden2, batch_first=True)
            self.decoder_lstm2 = nn.LSTM(hidden2, hidden1, batch_first=True)
            self.output_layer = nn.Linear(hidden1, input_dim)

        def forward(self, x):
            # Encoder
            enc1_out, _ = self.encoder_lstm1(x)
            enc1_out = self.dropout(enc1_out)
            enc2_out, _ = self.encoder_lstm2(enc1_out)
            bottleneck_out, _ = self.bottleneck_lstm(enc2_out)
            # Decoder
            dec1_out, _ = self.decoder_lstm1(bottleneck_out)
            dec2_out, _ = self.decoder_lstm2(dec1_out)
            # Output layer (apply to each timestep)
            out = self.output_layer(dec2_out)
            return out

    model = TuningLSTMAutoencoder(
        input_dim=trainX.shape[2], hidden1=hidden1, hidden2=hidden2, bottleneck=bottleneck,  dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Train for a few epochs
    for epoch in range(10):  # keep short to save time
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = loss_fn(outputs, batch)
            total += loss.item() * batch.size(0)
            count += batch.size(0)
    return total / count


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    # try 30 hyperparameter combinations
    study.optimize(objective, n_trials=30, n_jobs=4)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value}")
    print("  Best Hyperparameters:", trial.params)
