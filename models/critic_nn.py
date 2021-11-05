import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

class ActorNN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, device):
        super(ActorNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        for depth in range(len(hidden_layers)):
            if depth == 0:
                layers.append(nn.Linear(self.state_dim, hidden_layers[depth]))
                layers.append(nn.Tanh())
            else:
                layers.append(nn.Linear(hidden_layers[depth - 1], hidden_layers[depth]))
                layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layers[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, state):
        value = self.model(state)
        return value