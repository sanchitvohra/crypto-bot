import torch
import numpy as np

import environments
import preprocessing

def main():
    device = torch.device('cpu')

    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    data = preprocessing.load_data()
    env = environments.CryptoEnv(data, 1e6, 1e5, 0.05, 4)

    cumulative_reward = 0
    for i in range(10000):
        action = (np.random.rand(5) - 0.5) * 2
        cumulative_reward += env.step(action)

    print(1e6 + cumulative_reward, env.portfolio)


main()
