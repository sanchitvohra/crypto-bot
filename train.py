import torch
import torch.nn as nn
import numpy as np

import logging
import os
import matplotlib.pyplot as plt

import preprocessing
import environments
import models
import agents

def train():
    # Setup logging settings
    FORMAT = '[%(levelname)s] %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger('common')
    logger.setLevel(logging.INFO)

    logger.info("Training loop starting...")


    # setup training configuration
    training_steps = 32
    K_epochs = 100
    ep_len = 10000
    action_std = 0.5                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = 4           # action_std decay frequency (in num training steps)


    # environment configuration
    starting_balance = 1000000.0        # starting portfolio amount in dollars
    max_trade = 10000.0                # max number of $ amount for buy/sell
    trading_fee = 0.05                  # trading fee during buy
    history = 4                         # number of stacks in state
    reward_scaling = 10 ** -4           # scale the reward signal down

    # data loading
    data = preprocessing.load_data()

    # generate environments
    envs = []
    num_envs = 4
    for i in range(num_envs): 
        envs.append(environments.CryptoEnv(data, starting_balance, max_trade, trading_fee, history))
    state = envs[0].get_state(flatten=True)

    # generate validation environment
    venv = environments.CryptoEnv(data, starting_balance, max_trade, trading_fee, history)
    validate_freq = 10

    state_dim = state.shape[0]
    action_dim = 5

    device = torch.device('cpu')

    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    pretrained = False
    pretrained_path = None
    model_save_path = "checkpoints/"
    model_save_freq = 4

    plot_save_path = "plots/"
    plot_save_freq = 1

    logger.info(f'Training steps: {training_steps}')
    logger.info(f'Model Optimization epochs: {K_epochs}')
    logger.info(f'Episode length: {ep_len}')
    logger.info(f'Action std init: {action_std}')
    logger.info(f'Action std decay: {action_std_decay_rate}')
    logger.info(f'Min action std: {min_action_std}')
    logger.info(f'Action std decay freq: {action_std_decay_freq}')

    logger.info(f'Starting balance: {starting_balance}')
    logger.info(f'Maximum trade action: {max_trade}')
    logger.info(f'Trading fee: {trading_fee}')
    logger.info(f'State History: {history}')
    logger.info(f'Reward Scaling: {reward_scaling}')

    logger.info(f'State dimension: {state_dim}')
    logger.info(f'Action dimension: {action_dim}')

    logger.info(f'Pytoch device: {device}')
    logger.info(f'Pretrained: {pretrained}')
    if pretrained:
        logger.info(f'Pretrained model path: {pretrained_path}')
    logger.info(f'Model save path: {model_save_path}')
    logger.info(f'Model save frequecy: {model_save_freq}')


    # setup actor critic networks
    actor = models.ActorNN(state_dim, action_dim, [512, 256, 256], device)
    critic = models.CriticNN(state_dim, action_dim, [512, 256], device)
    lr_actor = 1e-4      # learning rate for actor network
    lr_critic = 1e-4     # learning rate for critic network

    logger.info('Actor: ')
    logger.info(actor)
    logger.info(f'Actor LR: {lr_actor}')
    logger.info('Critic: ')
    logger.info(critic)
    logger.info(f'Critic LR: {lr_critic}')

    # setup training agent
    agent_name = 'PPO'
    # PPO settings
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    logger.info(f'Agent Policy: {agent_name}')
    logger.info(f'Epsilon clip: {eps_clip}')
    logger.info(f'Gamma: {gamma}')

    if agent_name == 'PPO':
        agent = agents.PPO(state_dim, action_dim, actor, critic, lr_actor, lr_critic,
        num_envs, gamma, K_epochs, eps_clip, action_std, device)
    else:
        agent = None

    if pretrained:
        agent.load(pretrained_path)
        logger.info(f'Loaded saved model: {pretrained_path}')

    traj_step = 0
    time_step = 0
    max_validation_reward = 0

    while traj_step <= training_steps:

        # collect starting states for environments
        trajectory_data = np.zeros((len(envs), ep_len+1, 5 + 1 + 5 + 1), dtype=np.float32)
        states = []
        for i, env in enumerate(envs):
            env.reset()
            state = env.get_state(flatten=True)
            states.append(state)

            price_data = env.get_price_state(False, False)
            account_data = env.get_account_state(False)
            trajectory_data[i, 0, :5] = price_data[:, 1] # only high price
            trajectory_data[i, 0, 5:-1] = account_data
            trajectory_data[i, 0, -1] = env.portfolio

        states = np.array(states)

        # collect ep_len trajectories for each env
        average_return = 0

        for t in range(ep_len):
            actions = agent.select_action(states)
            states = []
            for i, env in enumerate(envs):
                action = actions[i]
                reward = env.step(action)
                average_return += reward
                reward = reward * reward_scaling
                agent.buffer.rewards[i].append(reward)
                states.append(env.get_state(flatten=True))
                time_step += 1

                price_data = env.get_price_state(False, False)
                account_data = env.get_account_state(False)
                trajectory_data[i, 1+t, :5] = price_data[:, 1] # only high price
                trajectory_data[i, 1+t, 5:-1] = account_data
                trajectory_data[i, 1+t, -1] = env.portfolio

            states = np.array(states)

        if traj_step % plot_save_freq == 0:
            print("saving plots")
            for i in range(len(envs)):
                prices = trajectory_data[i, :, :5]
                plt.figure()
                plt.subplot(511)
                plt.plot(prices[:, 0])                
                plt.subplot(512)
                plt.plot(prices[:, 1])
                plt.subplot(513)
                plt.plot(prices[:, 2])
                plt.subplot(514)
                plt.plot(prices[:, 3])
                plt.subplot(515)
                plt.plot(prices[:, 4])
                plt.savefig(os.path.join(plot_save_path, f'step_{traj_step}_env_{i}_price.png'))

                accounts = trajectory_data[i, :, 5:]
                plt.figure()
                plt.subplot(711)
                plt.plot(accounts[:, 6])                
                plt.subplot(712)
                plt.plot(accounts[:, 0])
                plt.subplot(713)
                plt.plot(accounts[:, 1])
                plt.subplot(714)
                plt.plot(accounts[:, 2])
                plt.subplot(715)
                plt.plot(accounts[:, 3])
                plt.subplot(716)
                plt.plot(accounts[:, 4])
                plt.subplot(717)
                plt.plot(accounts[:, 5])             
                plt.savefig(os.path.join(plot_save_path, f'step_{traj_step}_env_{i}_account.png'))

                plt.close('all')

        
        # increment step counter
        traj_step += 1
        average_return = average_return / num_envs

        # update agent using data
        median_loss = agent.update()

        agent_std = agent.action_std

        logger.info(f'Time Steps: {time_step}')
        logger.info(f'Average Reward: {average_return:15.3f}')
        logger.info(f'Median Loss: {median_loss:10.4f}')
        logger.info(f'Action Std: {agent_std:10.9f}')

        # update agent std
        if traj_step % action_std_decay_freq == 0:
            agent.action_std = action_std - traj_step * action_std_decay_rate
            if agent.action_std < min_action_std:
                agent.action_std = min_action_std
                agent.set_action_std(agent.action_std)

        if traj_step % model_save_freq == 0:
            if model_save_path != None:
                agent.save(checkpoint_path=os.path.join(model_save_path, "model" + str(time_step).zfill(10) + ".pth"))

        if traj_step % validate_freq == 0:
            venv.validate()
            state = venv.get_state(flatten=True)
            validation_return = 0

            mean_val_action = np.zeros(action_dim, dtype=np.float32)
            for t in range(ep_len):
                state = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action = agent.policy.validate(state)
                    mean_val_action += action
                    reward = venv.step(action)
                    validation_return += reward
                    state = venv.get_state(flatten=True)
            
            mean_val_action /= ep_len
            mean_val_action = str(list(mean_val_action))
            logger.info(f'Model Validation: {validation_return}')
            logger.info(f'Mean Val action:  {mean_val_action}')
            if validation_return > max_validation_reward:
                max_validation_reward = validation_return
                if model_save_path != None:
                    agent.save(checkpoint_path=os.path.join(model_save_path, "model.pth"))

if __name__ == "__main__":
    train()