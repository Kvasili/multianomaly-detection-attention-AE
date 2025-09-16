
'''
    @Date: 07/10/2025

    @Description: This code evaluates a trained LSTM Autoencoder with Attention mechanism on a specific instance from an abnormal dataset.
    it loads the model and the data, processes the data into sequences, and then computes the reconstruction and attention weights for the selected instance.
    It visualizes the attention weights assigned to each feature over time, helping to understand which features the model focuses on during reconstruction.


    USAGE
    python LSTM-attention-evaluate.py

'''


import torch.nn.functional as F
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib
from dataclasses import dataclass, field


# <------------------------ START ------------------------------>

@dataclass
class Config:

    seq_len: int = 20
    models_folder: str = "./models"
    min_max_scaler: str = "minmax_scaler_power_cycle"
    trained_model: str = "lstm_autoencoder_power_cycle_20_TimeAttention_FeatureAttention.pth"
    instance: int = 160  # Index of the instance to evaluate

    # "./data/abnormal_smoothed_drifted.csv"
    # "./data/abnormal_smoothed_spikes.csv"
    # "./data/abnormal_individual_spike.csv"
    # "./data/abnormal_smoothed.csv"
    data_path_abnormal: str = "./data/abnormal_smoothed_spikes.csv"
    columns: list = field(default_factory=lambda: [
        # "Index", "nfd-3-pwr", "nfd-4-flux", "cam-cnt", "nfd-1-cps", "nfd-2-cr",
        "nfd-1-cps", "nfd-4-flux", "cont air counts", "ram-con-lvl", "ram-pool-lvl", "ram-wtr-lvl"])

    labels: list = field(default_factory=lambda: [
        # "Index", "nfd-3-pwr", "nfd-4-flux", "cam-cnt", "nfd-1-cps", "nfd-2-cr",
        "neutron counts", "flux", "cam", "ram con", "ram pool", "ram wtr"])


def load_data(filename, cols_to_be_read, percentage):
    '''
        This function loads the data from a .csv file to a pandas dataframe.
        Percentage parameter defines the number of rows to be loaded
    '''

    df = pd.read_csv(filename)
    print(df.head())
    # df.dropna(inplace=True)
    df = df.loc[:, cols_to_be_read]

    length = len(df)

    try:
        if 0 < percentage <= 1.0:

            number_of_rows = int(percentage*length)
            df = df[:number_of_rows]

            return df
    except:
        raise ValueError("values in percentage should be in the range (0, 1]")


def split_data(df, test_size=0.10):
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


# Feature Attention Module


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

        # attn_weights = torch.sigmoid(self.v(energy).squeeze(-1))

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

            # attn_weights = torch.sigmoid(self.v(energy).squeeze(-1))

            attn_weights_time.append(attn_weights)

        # Stack over time: [batch, seq_len, input_dim]
        attn_matrix = torch.stack(attn_weights_time, dim=1)

        return attn_matrix  # attention over time and features


#  Autoencoder with Feature Attention
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

        # New Feature Attention Over Time
        self.feature_attention = FeatureAttention(
            input_dim=input_dim, hidden_dim=32)

        self.feature_attention_over_time = FeatureAttentionOverTime(
            input_dim=input_dim, hidden_dim=32
        )

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
        attn_applied_time = x * attn_matrix  # [batch, seq_len, input_dim]

        # Combine both attentions (e.g., element-wise multiply or add)
        combined_attention = attn_applied * \
            attn_applied_time  # or use another combination

        # Use attn_applied or attn_applied_time as input to encoder LSTM1
        out, _ = self.encoder_lstm1(combined_attention)
        out, _ = self.encoder_lstm2(out)
        encoder_outputs, (h, _) = self.bottleneck(out)
        h_bottleneck = h[-1]

        # Decoder input
        repeated = h_bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder_lstm1(repeated)
        out, _ = self.decoder_lstm2(out)
        out = self.output_layer(out)

        return out,  attn_weights, attn_matrix


def main():

    import matplotlib.pyplot as plt

    # create an instance of the config Class
    config = Config()

    # load the data
    abnormal_data = load_data(config.data_path_abnormal, config.columns, 1.0)
    print(f'[INFO] Initial abnormal data shape: {abnormal_data.shape}')

    # load normalization model
    scaler = joblib.load(os.path.join(
        config.models_folder, config.min_max_scaler))

    # Reconstruct dataframes with unscaled 'Index'
    abnormal_data = abnormal_data.copy()

    # abnormal_scaled.loc[:, config.columns[1:]] = scaler.transform(
    #     abnormal_data[config.columns[1:]])

    abnormal_scaled = scaler.transform(abnormal_data[config.columns])

    abnormal_scaled = pd.DataFrame(
        abnormal_scaled, columns=abnormal_data.columns, index=abnormal_data.index)

    # Export abnormal data normallized for visualization
    # abnormal_scaled.to_csv('./data/abnormal_individual_scaled.csv')
    # if os.path.exists('./data/abnormal_individual_scaled.csv') and os.path.getsize('./data/abnormal_individual_scaled.csv') > 0:
    #     print('Exported CSV file exists and is not empty.')
    # else:
    #     print('Export failed or file is empty.')

    print(f'[INFO] Abnormal data shape after norm: {abnormal_scaled.shape}')

    # Convert to sequences and torch tensor
    abnormalX = to_sequences(abnormal_scaled, seq_size=config.seq_len)
    print(f'[INFO] Training X data shape: {abnormalX.shape}')

    abnormal_tensor = torch.tensor(abnormalX, dtype=torch.float32)
    print(f'[INFO] Training X data shape: {abnormal_tensor.shape}')

#     # # ---- load the Model ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # Load trained model from folder
    model = AttentionLSTMAutoencoder(
        input_dim=abnormal_tensor.shape[2], seq_len=config.seq_len)

    model.load_state_dict(torch.load(
        os.path.join(config.models_folder, config.trained_model), map_location=device))
    model.to(device)

    # ---- Test on Abnormal Sample ---- #
    model.eval()
    tensor_instance = config.instance
    with torch.no_grad():

        # Select a single instance from the abnormal tensor
        # Need to be cahnged in the future to account for the entire abnormal tensor

        instance = abnormal_tensor[tensor_instance].unsqueeze(
            0).to(device)  # add batch dimension

        # get predictions for that instance
        recon, attn_weights, attn_matrix = model(instance)  #

    # absolute error
    error = torch.abs(instance - recon).squeeze(0).cpu().numpy()

    # mean_error = error.mean(axis=0)

    attn_weights = attn_weights[0].detach().cpu().numpy()  # shape: [input_dim]

    original = instance.squeeze(0).detach(
    ).cpu().numpy()  # shape: [seq_len, input_dim]

    attn_matrix = attn_matrix[0].detach(
    ).cpu().numpy()  # [seq_len, input_dim]
    # Average over time (seq_len axis)
    # avg_attn = attn.mean(axis=0)

    for i, w in enumerate(attn_weights):
        print(f"Sensor {i}: weight = {w:.3f}")

    # Plot attention weights per feature as heatmap over time
    time_labels = list(
        range(config.instance, config.instance + config.seq_len))
    plt.figure(figsize=(10, 6))
    sns.heatmap(attn_matrix.T, cmap='viridis', xticklabels=time_labels, yticklabels=[
                config.labels[i] for i in range(len(config.labels))])
    plt.xlabel("Time (s)", fontsize=16)
    # plt.ylabel("Sensor Index")
    # plt.title("Feature Attention Over Time")
    plt.yticks(rotation=90, fontsize=12)  # Rotate y-axis labels by 90 degrees
    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt

    # Plot attention weights per feature as line over time
    # Generate custom time labels
    time_labels = list(
        range(config.instance, config.instance + config.seq_len))
    plt.figure(figsize=(10, 6))
    for i in range(attn_matrix.shape[1]):  # input_dim
        plt.plot(time_labels, attn_matrix[:, i], label=f'Sensor {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('Attention Weight')
    plt.title('Attention Weights per Feature Across Time Steps')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # print weights as a table
    # Convert to DataFrame for readability
    df = pd.DataFrame(
        attn_matrix,
        columns=config.labels
    )

    print("\nAttention Weights per Time Step and Feature:\n")
    print(df)   # round to 3 decimals
    # df.to_csv(
    #     f'./spikes/attention_weights_spikes_{tensor_instance}.csv', index=False)


if __name__ == "__main__":
    main()
