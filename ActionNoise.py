import numpy as np
import random
import copy

# Modification of the OrnsteinUhlenbeckActionNoise class but does not use ActionNoise base class nor x0
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

class ActionNoise(object):
    def reset(self):
        pass

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size) #np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
#         x = x_prev + self.theta * (self.mu - x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x