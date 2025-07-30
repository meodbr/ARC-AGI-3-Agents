import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F

from .conv import DQNModel, Memory

env = gym.make("CartPole-v1")

class FFN(nn.Module):
    def __init__(self, input_size, output_size):
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class CartPoleAgent(DQNModel):
    def __init__(self):
        state, info = env.reset()
        model_class = FFN
        memory = Memory(10000)
        model_instanciation_args = {
            "input_size": len(state),
            "output_size": env.action_space.n,
        }
        super().__init__(model_class, memory, model_instanciation_args)
    
    def training_loop():
        state, info = env.reset()

        action = self.select_action(state, env.action_space)
        pass


