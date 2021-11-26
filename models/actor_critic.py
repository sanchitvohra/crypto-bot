import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, actor, critic, device):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = actor
        self.critic = critic
        self.device = device

    def act(self, state, deterministic = False):
        action_mean, action_var = self.actor(state)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def validate(self, state):
        action, _ = self.act(state, True)
        return action

    def evaluate(self, state, action):
        action_mean, action_var = self.actor(state)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy