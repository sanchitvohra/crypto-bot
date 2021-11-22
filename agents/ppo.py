import torch
import torch.nn as nn
import numpy as np
import math

from models.actor_critic import ActorCritic
from utils.rollout_buffer import RolloutBuffer

class PPO:
    def __init__(self, state_dim, action_dim, actor, critic, lr_actor, lr_critic, num_envs, gamma, K_epochs, eps_clip, action_std_init, device):
        
        self.action_std = action_std_init
        self.num_envs = num_envs
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.actor = actor
        self.critic = critic
        
        self.buffer = RolloutBuffer(num_envs)
        
        self.policy = ActorCritic(state_dim, action_dim, actor, critic, action_std_init, device).to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, actor, critic, action_std_init, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)


    def decay_action_std(self, alpha, step, min_action_std):
        self.action_std = alpha / ((step+1) **2)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
        self.set_action_std(self.action_std)


    def select_action(self, states):
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions, actions_logprob = self.policy_old.act(states)

        for i in range(self.num_envs):
            self.buffer.states[i].append(states[i])
            self.buffer.actions[i].append(actions[i])
            self.buffer.logprobs[i].append(actions_logprob[i])

        return actions.detach().cpu().numpy()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        for i in range(self.num_envs):
            env_rewards = []
            discounted_reward = 0
            for reward in reversed(self.buffer.rewards[i]):
                discounted_reward = reward + (self.gamma * discounted_reward)
                env_rewards.insert(0, discounted_reward)
            env_rewards = torch.tensor(env_rewards, dtype=torch.float32).to(self.device)
            env_rewards = (env_rewards - env_rewards.mean()) / (env_rewards.std() + 1e-7)
            rewards.append(env_rewards)
            
        rewards = torch.stack(rewards)

        old_states = []
        old_actions = []
        old_logprobs = []

        for i in range(self.num_envs):
            old_states.append(torch.squeeze(torch.stack(self.buffer.states[i], dim=0)))
            old_actions.append(torch.squeeze(torch.stack(self.buffer.actions[i], dim=0)))
            old_logprobs.append(torch.squeeze(torch.stack(self.buffer.logprobs[i], dim=0)))

        old_states = torch.stack(old_states)
        old_actions = torch.stack(old_actions)
        old_logprobs = torch.stack(old_logprobs)
        
        # Optimize policy for K epochs
        e_losses = []
        for _ in range(self.K_epochs):
            loss = 0
            for i in range(self.num_envs):
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states[i], old_actions[i])

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs[i].detach())

                # Finding Surrogate Loss
                advantages = rewards[i] - state_values.detach()   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss += -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards[i]) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            e_losses.append(loss.detach().mean().cpu().numpy())
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return sum(e_losses) / self.K_epochs
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
