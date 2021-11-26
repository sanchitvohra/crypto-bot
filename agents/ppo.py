import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import TensorDataset, DataLoader

from models.actor_critic import ActorCritic
from utils.rollout_buffer import RolloutBuffer

class PPO:
    def __init__(self, state_dim, action_dim, actor, critic, lr_actor, lr_critic, num_envs, gamma, K_epochs, eps_clip, device, normalize_reward = False, value_loss_factor = 0.5, entropy_loss_factor = 0.01):
        self.num_envs = num_envs
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.actor = actor
        self.critic = critic
        
        self.normalize_reward = normalize_reward
        self.value_loss_factor = value_loss_factor
        self.entropy_loss_factor = entropy_loss_factor
        
        self.buffer = RolloutBuffer(num_envs)
        
        self.policy = ActorCritic(state_dim, action_dim, actor, critic, device).to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9)

        self.policy_old = ActorCritic(state_dim, action_dim, actor, critic, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

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
        old_states = []
        old_actions = []
        old_logprobs = []
        old_rewards = []

        for i in range(self.num_envs):
            old_states.append(torch.squeeze(torch.stack(self.buffer.states[i], dim=0)))
            old_actions.append(torch.squeeze(torch.stack(self.buffer.actions[i], dim=0)))
            old_logprobs.append(torch.squeeze(torch.stack(self.buffer.logprobs[i], dim=0)))
            old_rewards.append(torch.tensor(self.buffer.rewards[i]).to(torch.float32).to(self.device))

        old_states = torch.stack(old_states)
        old_actions = torch.stack(old_actions)
        old_logprobs = torch.stack(old_logprobs)
        old_rewards = torch.stack(old_rewards)
        last_states = self.buffer.last_states.to(self.device)
        last_critic = self.policy.critic(last_states).detach()

        # states: [#envs, #traj, #dim]
        # action: [#envs, #traj, #dim]
        # logpro: [#envs, #traj]
        # reward: [#envs, #traj]

        rewards = torch.zeros(old_rewards.size(), dtype=torch.float32).to(self.device)

        discounted_reward = 0
        for i in range(old_states.shape[1]):
            discounted_reward = old_rewards[:, -1 * (i+1)] + (self.gamma * discounted_reward)
            last_state_reward = (self.gamma ** (i+1)) * last_critic
            rewards[:, -1 * (i+1)] = discounted_reward + torch.squeeze(last_state_reward)

        old_states = old_states.view(old_states.shape[0] * old_states.shape[1], old_states.shape[2])
        old_actions = old_actions.view(old_actions.shape[0] * old_actions.shape[1], old_actions.shape[2])
        old_logprobs = old_logprobs.view(old_logprobs.shape[0] * old_logprobs.shape[1])
        rewards = rewards.view(rewards.shape[0] * rewards.shape[1])

        # print(rewards)

        # training_data = TensorDataset(old_states, old_actions, old_logprobs, rewards)
        # training_dataloader = DataLoader(training_data, batch_size=2048, shuffle=True)
        
        # Optimize policy for K epochs
        f_losses = []
        b_losses = []
        for e in range(self.K_epochs):
            # for old_states, old_actions, old_logprobs, rewards in training_dataloader:
                # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            actor_loss = -1 * torch.min(surr1, surr2)
            critic_loss = self.value_loss_factor * self.MseLoss(state_values, rewards)
            entropy_loss = -1 * self.entropy_loss_factor * dist_entropy

            loss = actor_loss.mean() + critic_loss + entropy_loss.mean()

            if self.normalize_reward:
                print(loss)
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            total_loss = loss
            actor_loss_ret = actor_loss.detach().mean()
            critic_loss_ret = critic_loss.detach()
            entropy_loss_ret = entropy_loss.detach().mean()

            f_losses.append(total_loss.detach().cpu().numpy().item())
            b_losses.append(np.array([actor_loss_ret.cpu().numpy().item(),
                critic_loss_ret.cpu().numpy().item(),
                entropy_loss_ret.cpu().numpy().item()]))
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        f_losses = np.array(f_losses)
        b_losses = np.array(b_losses)
        return np.median(f_losses), np.median(b_losses, axis=0)
    
    def scheduler_step(self):
        self.scheduler.step()
        return self.scheduler.get_last_lr()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
