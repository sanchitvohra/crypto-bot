import torch
import torch.nn as nn
import numpy as np

import logging

import preprocessing
import env
import models

FORMAT = '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger('common')
logger.setLevel(logging.INFO)

data = preprocessing.load_data()

env = env.CryptoEnv(data, 10000.0, 1000.0, 0.01, 1)

device = torch.device('cpu')

# state_encoder = models.StateEncoder(5, 9, 6, 5, device)

# prices, state = env.get_state()
# prices, state = torch.from_numpy(prices).to(torch.float32), torch.from_numpy(state).to(torch.float32)
# latent = state_encoder(prices, state)
# print(latent)

batch_size = 32
input_dim = 16
hidden_dim = 32
output_dim = 16
num_layers = 3
dropout = 0.1

dynamics = models.DynmaicsLSTM(input_dim, hidden_dim, output_dim, num_layers, dropout)
(h_0, c_0) = dynamics.init_hidden(batch_size, device)

dummy = torch.rand(32, 32, input_dim)
out, (h_f, c_f) = dynamics(dummy, (h_0, c_0))

print(out.shape)
