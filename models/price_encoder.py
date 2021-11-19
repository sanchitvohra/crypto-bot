import torch
import torch.nn as nn

class PriceEncoder(nn.Module):
    def __init__(self, num_coins, price_dim, output_dim, device):
        super(PriceEncoder, self).__init__()
        self.num_coins = num_coins
        self.device = device

        self.price_encoders =  [] 
        
        for coin in range(num_coins):
            self.price_encoders.append(
                nn.Sequential(
                    nn.Linear(price_dim, output_dim),
                    nn.Tanh()
                )
            )

    def forward(self, prices):
        outputs = []
        for coin in range(self.num_coins):
            outputs.append(self.price_encoders[coin](prices[coin]))

        return torch.stack(outputs)
