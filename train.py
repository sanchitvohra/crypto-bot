######################################IMPORTS START#################################################
import torch
import numpy as np

import logging
import os
import sys
import matplotlib.pyplot as plt

import preprocessing
import environments
import models
import agents
import utils
from torch.utils.tensorboard import SummaryWriter

######################################IMPORTS DONE#################################################

def train(logDir = None):

    ######################################SETUP START#################################################
    FORMAT = logging.Formatter('[%(levelname)s] %(message)s')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logDir:
        logDir = "logs/"
        logDir += "config" + str(len(os.listdir(logDir))).zfill(3)
        os.mkdir(logDir)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(logDir, "training"))
    fileHandler.setFormatter(FORMAT)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(FORMAT)
    logger.addHandler(consoleHandler)

    logger.info("Training loop starting...")

    # setup training configuration
    training_steps = 32
    K_epochs = 100
    ep_len = 10000

    # create abstraction    
    action_std = 0.5                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = 4           # action_std decay frequency (in num training steps)
    action_std_compute = utils.linear_decay(action_std, action_std_decay_rate, min_action_std)

    # environment configuration
    starting_balance = 1000000.0        # starting portfolio amount in dollars
    max_trade = 10000.0                 # max number of $ amount for buy/sell
    trading_fee = 0.01                  # trading fee during buy
    history = 4                         # number of stacks in state
    reward_scaling = 10 ** -3           # scale the reward signal down

    # data loading
    data = preprocessing.load_data()

    # generate environments
    num_envs = 4
    envs = []
    for i in range(num_envs): 
        envs.append(environments.CryptoEnv(data, starting_balance, max_trade, trading_fee, history))
    state = envs[0].get_state(flatten=True)

    # generate validation environment
    venv = environments.CryptoEnv(data, starting_balance, max_trade, trading_fee, history)
    validate_freq = 5

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
    model_save_path = os.path.join(logDir, "checkpoints/")
    os.mkdir(model_save_path)
    model_save_freq = 4

    plot_save_path = os.path.join(logDir, "plots/")
    os.mkdir(plot_save_path)
    plot_save_freq = 1

    for i in range(num_envs):
        os.mkdir(os.path.join(plot_save_path, 'env' + str(i).zfill(2)))

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
        num_envs, gamma, K_epochs, eps_clip, action_std, device, False, 0.05, 0.01)
    else:
        agent = None

    if pretrained:
        agent.load(pretrained_path)
        logger.info(f'Loaded saved model: {pretrained_path}')

    writer = SummaryWriter(logDir)

    #########################################SETUP DONE##############################################
    #########################################TRAINING START##############################################

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

        # increment step counter
        traj_step += 1

        average_return = average_return / num_envs
        # update agent using data
        median_loss, median_breakdown = agent.update()

        logger.info(f'Time Steps: {time_step}')
        logger.info(f'Average Reward: {average_return:15.3f}')
        logger.info(f'Median Loss: {median_loss:10.4f}')
        logger.info(f'Action Std: {agent.action_std:10.9f}')

        writer.add_scalar("Average Return/Train", average_return, traj_step)
        writer.add_scalar("Total Loss/Train", median_loss, traj_step)
        writer.add_scalar("Action Std/Train", agent.action_std, traj_step)
        writer.add_scalar("Actor Loss/Train", median_breakdown[0], traj_step)
        writer.add_scalar("Value Loss/Train", median_breakdown[1], traj_step)
        writer.add_scalar("Entropy Loss/Train", median_breakdown[2], traj_step)

        if traj_step % plot_save_freq == 0:
            for i in range(num_envs):
                utils.plot_trajectory(trajectory_data[i], os.path.join(plot_save_path, 'env' + str(i).zfill(2)), traj_step)

        # update agent std
        if traj_step % action_std_decay_freq == 0:
            index = traj_step // action_std_decay_freq
            if index > len(action_std_compute):
                index = -1
            agent.action_std = action_std_compute[index]
            agent.set_action_std(agent.action_std)
            agent.scheduler_step()

        if traj_step % model_save_freq == 0:
            if model_save_path != None:
                agent.save(checkpoint_path=os.path.join(model_save_path, "model" + str(time_step).zfill(10) + ".pth"))

        if traj_step % validate_freq == 0:
            validation_return = 0
            for i in range(3):
                venv.validate(i)
                state = venv.get_state(flatten=True)
                for t in range(ep_len):
                    state = torch.FloatTensor(state).to(device)
                    with torch.no_grad():
                        action = agent.policy.validate(state)
                        reward = venv.step(action)
                        validation_return += reward
                        state = venv.get_state(flatten=True)
                
            logger.info(f'Validation Reward: {validation_return / 3:15.3f}')
            writer.add_scalar("Validation Reward/Test", validation_return / 3, traj_step)
            if validation_return > max_validation_reward:
                max_validation_reward = validation_return
                if model_save_path != None:
                    agent.save(checkpoint_path=os.path.join(model_save_path, "model.pth"))

    #########################################TRAINING STOP##############################################

if __name__ == "__main__":
    if len(sys.argv) == 1:
        train()
    else:
        train(sys.argv[1])