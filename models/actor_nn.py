import torch.nn as nn

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
        # layers.append(nn.Linear(hidden_layers[-1], action_dim*2))
        # layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.mean = nn.Sequential(*[nn.Linear(hidden_layers[-1], action_dim), nn.Tanh()])
        self.var = nn.Sequential(*[nn.Linear(hidden_layers[-1], action_dim), nn.Softmax(1)])

    def forward(self, state):
        action_hidden = self.model(state)
        action_mean = self.mean(action_hidden)
        action_var = self.var(action_hidden)
        return action_mean, action_var