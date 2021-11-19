import torch
import torch.nn as nn

from models.price_encoder import PriceEncoder

class StateEncoder(nn.Module):
    def __init__(self, num_coins, price_dim, state_dim, output_dim, device):
        super(StateEncoder, self).__init__()
        self.num_coins = num_coins
        self.device = device

        self.price_encoder =  PriceEncoder(num_coins, price_dim, output_dim, device)

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, prices, state):
        price_latent = self.price_encoder(prices)
        state_latent = self.state_encoder(state)
        return torch.vstack([price_latent, state_latent.unsqueeze(0)])
