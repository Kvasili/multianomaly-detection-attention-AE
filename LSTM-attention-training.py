
'''
    @Date: 07/21/2025

    @Description: New training with attention mechanism for LSTM. 
    
    Check the following repository for more information:
    https://github.com/JulesBelveze/time-series-autoencoder/tree/master

    USAGE
    python LSTM-attention-training.py

'''


import torch.nn.functional as F
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import joblib
from dataclasses import dataclass, field
from torch.autograd import Variable


# <------------------------ START ------------------------------>

@dataclass
class Config:
    epochs: int = 30
    batch_size: int = 64
    seq_len: int = 20
    learning_rate: float = 0.001
    models_folder: str = "./models"
    scaler: str = "minmax_scaler_power_cycle"
    trained_model: str = "lstm_autoencoder_power_cycle_20_TimeAttention_FeatureAttention.pth"
    data_path: str = "../data/Power_Cycle_1779463.csv"
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


# ---- Feature Attention Module ---- #
class FeatureAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + 1, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch, seq_len, input_dim]
        # decoder_hidden: [batch, hidden_dim]

        batch, seq_len, input_dim = encoder_outputs.size()
        # Average encoder_outputs over time (seq_len) to get feature vector: shape [batch, input_dim]
        encoder_feature_avg = encoder_outputs.mean(dim=1)  # [batch, input_dim]

        # Expand decoder_hidden to [batch, input_dim, hidden_dim]
        decoder_hidden_exp = decoder_hidden.unsqueeze(1).repeat(
            1, input_dim, 1)  # [batch, input_dim, hidden_dim]

        # Concatenate along last dim: feature vector (as 1D) + decoder hidden (hidden_dim)
        # So first unsqueeze encoder_feature_avg to [batch, input_dim, 1]
        # [batch, input_dim, hidden_dim+1]
        concat = torch.cat(
            [encoder_feature_avg.unsqueeze(2), decoder_hidden_exp], dim=2)

        # [batch, input_dim, hidden_dim]
        energy = torch.tanh(self.attn(concat))
        attn_weights = torch.softmax(
            self.v(energy).squeeze(-1), dim=1)  # [batch, input_dim]

        return attn_weights


# # ---- Feature Attention Over Time Module ---- #
class FeatureAttentionOverTime(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim + 1, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: [batch, seq_len, input_dim]
        # decoder_hidden: [batch, hidden_dim]

        batch, seq_len, input_dim = encoder_outputs.size()

        # For each time step separately
        attn_weights_time = []

        for t in range(seq_len):
            # Get feature vector at time t: [batch, input_dim]
            encoder_feature_t = encoder_outputs[:, t, :]  # [batch, input_dim]

            # Expand decoder hidden: [batch, input_dim, hidden_dim]
            decoder_hidden_exp = decoder_hidden.unsqueeze(
                1).repeat(1, input_dim, 1)

            # Concatenate: [batch, input_dim, hidden_dim+1]
            concat = torch.cat(
                [encoder_feature_t.unsqueeze(2), decoder_hidden_exp], dim=2)

            # [batch, input_dim, hidden_dim]
            energy = torch.tanh(self.attn(concat))
            attn_weights = torch.softmax(
                self.v(energy).squeeze(-1), dim=1)  # [batch, input_dim]

            attn_weights_time.append(attn_weights)

        # Stack over time: [batch, seq_len, input_dim]
        attn_matrix = torch.stack(attn_weights_time, dim=1)

        return attn_matrix  # attention over time and features


# # ---- Modified Autoencoder with Feature Attention ---- #
class AttentionLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.encoder_lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.bottleneck = nn.LSTM(64, 32, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(32, 64, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.output_layer = nn.Linear(128, input_dim)

        # self.feature_attention = FeatureAttention(
        #     input_dim=input_dim, hidden_dim=32)

        # New Feature Attention Over Time
        self.feature_attention_over_time = FeatureAttentionOverTime(
            input_dim=input_dim, hidden_dim=32
        )

        self.feature_attention = FeatureAttention(
            input_dim=input_dim, hidden_dim=32)

    def forward(self, x):
        # Encoder
        out, _ = self.encoder_lstm1(x)
        out, _ = self.encoder_lstm2(out)
        encoder_outputs, (h, _) = self.bottleneck(out)

        # Bottleneck representation (final hidden state)
        h_bottleneck = h[-1]  # shape: [batch, 32]

        # Feature Attention
        attn_weights = self.feature_attention(
            x, h_bottleneck)  # [batch, input_dim]
        attn_applied = x * attn_weights.unsqueeze(1)  # weight input features

        # Feature Attention over time
        attn_matrix = self.feature_attention_over_time(
            x, h_bottleneck)  # [batch, seq_len, input_dim]

        # Apply attention to input features, element-wise
        attn_applied = x * attn_matrix  # [batch, seq_len, input_dim]

        # Decoder input
        repeated = h_bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm1(repeated)
        out, _ = self.decoder_lstm2(out)
        out = self.output_layer(out)

        return out,  attn_weights, attn_matrix


def main():

    # create an instance of the config Class
    config = Config()

    # load the data
    data = load_data(config.data_path, config.columns, 1.0)

    # split in train and validation
    train, val = split_data(data, test_size=0.2)

    print(f'[INFO] Training data shape: {train.shape}')
    print(f'[INFO] Validation data shape: {val.shape}')

    # Normalize data with standard scaler
    # Columns to normalize (exclude 'Index')

    # Perform normalization
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    # Reconstruct dataframes with unscaled 'Index'
    train_scaled = train.copy()
    val_scaled = val.copy()

    # # This code for exclusing a column from normalizing
    # train_scaled.loc[:, config.columns[1:]] = scaler.fit_transform(
    #     train[config.columns[1:]])
    # val_scaled.loc[:, config.columns[1:]] = scaler.transform(
    #     val[config.columns[1:]])

    train_scaled = scaler.fit_transform(train_scaled)
    val_scaled = scaler.transform(val_scaled)

    # Save the Normalization model
    if not os.path.exists(config.models_folder):
        os.makedirs(config.models_folder)
    joblib.dump(scaler, os.path.join(
        config.models_folder, config.scaler))
    print(f"[INFO] Scaler saved to {config.models_folder}")

    # convert back to DataFrame
    train_scaled = pd.DataFrame(
        train_scaled, columns=train.columns, index=train.index)
    val_scaled = pd.DataFrame(val_scaled, columns=val.columns, index=val.index)

    print(f'[INFO] Training data shape after norm: {train_scaled.shape}')
    print(f'[INFO] Validation data shape after norm: {val_scaled.shape}')

    # create the tensors for training and validation
    trainX = to_sequences(
        train_scaled, seq_size=config.seq_len)
    print(f'[INFO] Training X data shape: {trainX.shape}')
    valX = to_sequences(
        val_scaled, seq_size=config.seq_len)
    print(f'[INFO] Training Val data shape: {valX.shape}')

    # # # Convert to torch
    train_dataset = torch.tensor(trainX, dtype=torch.float32)
    print(f'[INFO] Training tensors dimensions: {train_dataset.shape}')
    val_dataset = torch.tensor(valX, dtype=torch.float32)
    print(f'[INFO] Training tensors dimensions: {val_dataset.shape}')

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False)

    # # ---- Train the Model ---- #
    model = AttentionLSTMAutoencoder(
        input_dim=train_dataset.shape[2], seq_len=config.seq_len)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    training_losses = []
    validation_losses = []

    for epoch in range(config.epochs):
        # Training Loop
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs,  _, _ = model(batch)
            # outputs = model(batch)
            loss = loss_fn(outputs, batch)
            loss.backward(retain_graph=True)

            optimizer.step()
            epoch_loss += loss.item()

        training_losses.append(epoch_loss / len(train_loader))

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs, _, _ = model(batch)
                # outputs = model(batch)
                loss = loss_fn(outputs, batch)
                val_loss += loss.item()

        validation_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch + 1}/{config.epochs}, "
              f"Train Loss: {training_losses[-1]:.4f}, Val Loss: {validation_losses[-1]:.4f}")

    # Before saving the model create the models folder if it does not exist
    if not os.path.exists(config.models_folder):
        os.makedirs(config.models_folder)

    torch.save(model.state_dict(), os.path.join(
        config.models_folder, config.trained_model))
    print("[INFO] Model saved.")

    # Plot Training and Validation Loss
    plt.figure(figsize=(6, 6))
    plt.plot(range(1, config.epochs + 1),
             training_losses, label="Training Loss", color="blue")
    plt.plot(range(1, config.epochs + 1), validation_losses,
             label="Validation Loss", color="orange", linestyle="--")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss", fontsize=16)
    plt.legend()
    # plt.grid()
    plt.show()

    # Reconstruction error histograms
    model.eval()
    for loader, label, color in [(train_loader, 'Train', 'blue'), (val_loader, 'Validation', 'green')]:
        all_mae = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                outputs, _, _ = model(batch)
                # shape: [batch, seq_len, features]
                reconstruction_errors = torch.abs(outputs - batch)
                mae = torch.mean(reconstruction_errors, dim=(
                    1, 2)).cpu().numpy()  # mean over time and features
                all_mae.extend(mae)

        # Convert all_mae to a 1D array
        all_mae = np.array(all_mae).flatten()
        plt.hist(all_mae, bins=30, color=color, alpha=0.7, label=label)
        plt.title(f'Reconstruction Error {label} Dataset')
        plt.xlabel('Mean Absolute Error')
        plt.ylabel('Frequency')
        plt.show()


if __name__ == "__main__":
    main()
