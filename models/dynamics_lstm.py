import torch
import torch.nn as nn

class DynmaicsLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(DynmaicsLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout


        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout)

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))

    def forward(self, input, hidden):
        lstm_out, lstm_hidden = self.lstm(input, hidden)
        return self.linear(lstm_out), lstm_hidden
        