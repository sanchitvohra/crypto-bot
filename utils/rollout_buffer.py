class RolloutBuffer:
    def __init__(self, dim=1):
        self.dim = dim
        self.reset()
    
    def reset(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.last_states = []
        for _ in range(self.dim):
            self.actions.append([])
            self.states.append([])
            self.logprobs.append([])
            self.rewards.append([])

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        self.reset()