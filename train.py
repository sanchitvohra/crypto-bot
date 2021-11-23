import torch
import torch.nn as nn
import numpy as np

import logging

import preprocessing
import environments
import models
import agents

def train():
    FORMAT = '[%(levelname)s] %(message)s'
    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger('common')
    logger.setLevel(logging.INFO)

    logger.info("Training loop starting...")

    training_steps = int(3e6)
    K_epochs = 20
    ep_len = 10000
    update_timestep = ep_len

    action_std = 0.1                       # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.005         # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.01            # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = 5 * update_timestep  # action_std decay frequency (in num timesteps)

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    logger.info(f'Training steps: {training_steps}')
    logger.info(f'Model Optimization epochs: {K_epochs}')
    logger.info(f'Episode length: {ep_len}')
    logger.info(f'Policy update frequency: {update_timestep}')
    logger.info(f'Action std init: {action_std}')
    logger.info(f'Action std decay: {action_std_decay_rate}')
    logger.info(f'Min action std: {min_action_std}')
    logger.info(f'Action std decay freq: {action_std_decay_freq}')
    logger.info(f'Epsilon clip: {eps_clip}')
    logger.info(f'Gamma: {gamma}')

    starting_balance = 1000000.0 # starting portfolio amount in dollars
    max_trade = 100000.0 # max number of stocks
    trading_fee = 0.01
    history = 4 # number of stacks in state
    reward_scaling = 10 ** -4
    
    logger.info(f'Starting balance: {starting_balance}')
    logger.info(f'Maximum trade action: {max_trade}')
    logger.info(f'Trading fee: {trading_fee}')
    logger.info(f'State History: {history}')
    logger.info(f'Reward Scaling: {reward_scaling}')

    data = preprocessing.load_data()
    envs = []
    num_envs = 4
    for i in range(num_envs): 
        envs.append(environments.CryptoEnv(data, starting_balance, max_trade, trading_fee, history))
    state = envs[0].get_state(flatten=True)

    venv = environments.CryptoEnv(data, starting_balance, max_trade, trading_fee, history)

    state_dim = state.shape[0]
    action_dim = 5

    logger.info(f'State dimension: {state_dim}')
    logger.info(f'Action dimension: {action_dim}')

    device = torch.device('cpu')
    pretrained = False
    pretrained_path = None
    model_save_path = "checkpoints/model.pth"
    model_save_freq = 5000

    logger.info(f'Pytoch device: {device}')
    logger.info(f'Pretrained: {pretrained}')
    if pretrained:
        logger.info(f'Pretrained model path: {pretrained_path}')
    logger.info(f'Model save path: {model_save_path}')
    logger.info(f'Model save frequecy: {model_save_freq}')

    actor = models.ActorNN(state_dim, action_dim, [512, 256, 256], device)
    critic = models.CriticNN(state_dim, action_dim, [512, 256], device)
    lr_actor = 3e-4      # learning rate for actor network
    lr_critic = 1e-3     # learning rate for critic network

    logger.info('Actor: ')
    logger.info(actor)
    logger.info(f'Actor LR: {lr_actor}')
    logger.info('Critic: ')
    logger.info(critic)
    logger.info(f'Critic LR: {lr_critic}')

    agent_name = 'PPO'
    logger.info(f'Agent Policy: {agent_name}')
    if agent_name == 'PPO':
        agent = agents.PPO(state_dim, action_dim, actor, critic, lr_actor, lr_critic,
        num_envs, gamma, K_epochs, eps_clip, action_std, device)
    else:
        agent = None

    traj_step = 0
    time_step = 0
    max_validation_reward = 0

    while time_step <= training_steps:
        states = []
        for env in envs:
            env.reset()
            state = env.get_state(flatten=True)
            states.append(state)

        states = np.array(states)

        # collect trajectories for each env
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

            states = np.array(states)
        
        traj_step += 1
        average_return = average_return / num_envs
        mean_loss = agent.update()
        print(mean_loss.shape)
        agent_std = agent.action_std
        logger.info(f'Average Reward: {average_return:15.3f}')
        logger.info(f'Mean Loss: {mean_loss[0]:10.4f}, A/C/E: {mean_loss[1]:10.4f},{mean_loss[2]:10.4f},{mean_loss[3]:10.4f}')
        logger.info(f'Action Std: {agent_std:10.9f}')

        agent.action_std = action_std - traj_step * action_std_decay_rate
        agent.set_action_std(agent.action_std)
        # agent.decay_action_std(action_std, traj_step, min_action_std)

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
        logger.info(f'Best Model Validation: {validation_return}')
        logger.info(f'Mean Val action:  {mean_val_action}')
        if validation_return > max_validation_reward:
            max_validation_reward = validation_return
            if model_save_path != None:
                agent.save(checkpoint_path=model_save_path)
            

if __name__ == "__main__":
    train()