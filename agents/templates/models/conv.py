from collections import namedtuple, deque
from typing import Type
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel
import matplotlib.pyplot as plt

from ..rl_utils import utils

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Memory:
    def __init__(self, size):
        self.transitions = deque([], maxlen=size)

    def sample(self, n: int) -> list[Transition]:
        return random.sample(self.transitions, k=n)
    
    def append(self, transition: Transition):
        self.transitions.append(transition)
    
    def __len__(self):
        return len(self.transitions)

class EpisodeStatistics(BaseModel):
    score: list[int]
    duration: list[int]


class DQNModel:
    """
    Class to wrap DQN training process
    """
    GAMMA: float     = 0.99
    LR: float        = 1e-3
    EPS_MAX: float   = 0.9
    EPS_MIN: float   = 0.1
    EPS_DECAY: float = 2500
    TAU: float       = 0.005
    BATCH_SIZE: int  = 128

    def __init__(self, model_class: Type[nn.Module], memory: Memory, model_instantation_args={}):
        self.model = model_class(**model_instantation_args)
        self.target_model = model_class(**model_instantation_args)
        self.target_model.eval()
        self.memory = memory
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LR)

        self.action_count = 0
        self.statistics = EpisodeStatistics(
            score=[],
            duration=[],
        )
    
    def predict(self, input: torch.Tensor) -> torch.Tensor:
        res = torch.Tensor()
        with torch.no_grad():
            res = self.model(input)
        return res
    
    def compute_sample_batch(self, batch_size):
        transitions = self.memory.sample(batch_size)
        transitions = Transition(*zip(*transitions))
    
        print(transitions)
        state_batch      = torch.stack(transitions.state)
        next_state_batch = torch.stack([tr for tr in transitions.next_state if tr != None])
        actions_batch    = torch.Tensor(transitions.action)
        reward_batch     = torch.Tensor(transitions.reward)

        # predicted = Q(s, a)
        reward_predictions_all_actions: torch.Tensor = self.model(state_batch)
        predicted_reward_batch = reward_predictions_all_actions.gather(1, actions_batch)

        # expected = r + gamma * max_a(Q'(s',a))
        not_final_mask = [s != None for s in next_state_batch]
        next_state_reward_prediction: torch.Tensor = torch.zeros([batch_size])
        next_state_reward_prediction[not_final_mask] = self.target_model(next_state_batch).max(dim=1).values.detach()

        expected_reward_batch = reward_batch + self.GAMMA * next_state_reward_prediction

        return (predicted_reward_batch, expected_reward_batch)
    
    def train_iterations(self, n_iterations, batch_size=None) -> None:
        if not batch_size: batch_size = self.BATCH_SIZE

        if len(self.memory) < batch_size:
            return

        self.model.train()
        for _ in range(n_iterations):
            self.train_step(batch_size)
    
    def train_step(self, batch_size=None):
        if not batch_size: batch_size = self.BATCH_SIZE

        if len(self.memory) < batch_size:
            return

        self.model.train()
        self.optimizer.zero_grad()

        x_hat, x = self.compute_sample_batch(batch_size)

        loss: torch.Tensor = self.model.loss(x, x_hat)

        loss.backward()
        self.optimizer.step()

        self.update_target_model() 

    
    def get_epsilon(self):
        return self.EPS_MIN + (self.EPS_MAX - self.EPS_MIN) * math.exp(-1 * (self.action_count/self.EPS_DECAY))

    
    def select_action(self, observations: torch.Tensor, action_space: torch.Tensor) -> int:
        p = random.random()
        epsilon = self.get_epsilon()

        if p < epsilon:
            return torch.Tensor(random.sample(action_space, k=1)[0])
        else:
            return self.model(observations).max(0).indices


    def store_transition(self, transition: Transition):
        self.memory.append(transition)
        self.action_count += 1
    
    def store_episode_statistics(self, duration: int, score: int):
        self.statistics.duration.append(duration)
        self.statistics.score.append(score)

    
    def update_target_model(self):
        policy_state_dict = self.model.state_dict()
        target_state_dict = self.target_model.state_dict()
        for key in policy_state_dict.keys():
            target_state_dict[key] = utils.linear_interp(self.TAU, target_state_dict, policy_state_dict)
        self.target_model.load_state_dict(target_state_dict)
    
    def plot_statistics(self):
        durations = np.array(self.statistics.duration)
        scores = np.array(self.statistics.score)
        x = range(len(durations))
        
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title("Durations")
        ax[0].plot(x, durations, label="Durations")

        ax[0].set_title("Scores")
        ax[1].plot(x, scores, label="Scores")

        plt.pause(0.001)



class ConvBasicModule(nn.Module):
    """
    Basic Conv2D module
    """
    def __init__(self, size=32):
        super().__init__()
        self.input_size = size*size
        self.layer1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2)
        self.layer2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.layer1(x)))
        x = self.pool(F.relu(self.layer2(x)))
        x = self.pool(F.relu(self.layer3(x)))

        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(x, x_hat)
    

class ConvBasic(DQNModel):
    pass