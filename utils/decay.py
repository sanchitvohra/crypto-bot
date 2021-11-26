def linear_decay(initial_value, decay_value, min_value):
    decayed_values = [initial_value]
    while(1):
        next_value = decayed_values[-1] - decay_value
        if next_value <= min_value:
            decayed_values.append(min_value)
            return decayed_values
        else:
            decayed_values.append(next_value)

def step_linear_decay(initial_value, decay_values, min_value, thresholds):
    decayed_values = [initial_value]
    for decay_value, threshold in zip(decay_values, thresholds):
        for i in range(threshold):
            next_value = decayed_values[-1] - decay_value
            if next_value <= min_value:
                decayed_values.append(min_value)
                return decayed_values
            else:
                decayed_values.append(next_value)
    remaining_values = linear_decay(decayed_values[-1], decay_values[-1], min_value)
    return decay_values.append(remaining_values)


import torch.nn as nn

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias




def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
    