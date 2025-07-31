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
import time

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

    def __init__(self, model_class: Type[nn.Module], memory: Memory, model_instantation_args={}, device=None):
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "cpu"
            )
        
        print(f"Using device: {device}")
        self.device = device

        self.model = model_class(**model_instantation_args)
        self.target_model = model_class(**model_instantation_args)
        self.model.to(device)
        self.target_model.to(device)
        self.target_model.eval()
        self.memory = memory
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LR)

        self.action_count = 0
        self.fig = None
        self.statistics = {
            "score": [],
            "duration": [],
            "epsilon": [],
            "loss": [],
            "memory_size": []
        }
        self.tprof = {
            "tensor_conversion_time": [],
            "transition_zip_time": [],
            "prediction_time": [],
            "expected_computation_time": [],
            "statistics_computation_time": []
        }

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        res = torch.Tensor(device=self.device)
        self.model.eval()
        with torch.no_grad():
            res = self.model(input)
        return res
    
    def compute_sample_batch(self, batch_size):
        t0 = time.perf_counter()
        transitions = self.memory.sample(batch_size)
        transitions = Transition(*zip(*transitions))
        t_transition_zip = time.perf_counter() - t0

        # Convert transitions to tensors
        state_batch      = torch.stack(transitions.state).to(self.device)
        t0 = time.perf_counter()
        next_state_batch = torch.stack([next_s for next_s in transitions.next_state if next_s != None]).to(self.device)
        actions_batch    = torch.tensor(transitions.action, dtype=torch.int64, device=self.device).unsqueeze(1)  # Unsqueeze to make it a column vector
        reward_batch     = torch.tensor(transitions.reward, device=self.device)

        t_tensor_conversion = time.perf_counter() - t0

        t0 = time.perf_counter()

        # predicted = Q(s, a)
        reward_predictions_all_actions: torch.Tensor = self.model(state_batch)
        predicted_reward_batch = reward_predictions_all_actions.gather(1, actions_batch).squeeze(1)  # Gather the predicted rewards for the actions taken

        t_prediction = time.perf_counter() - t0


        # expected = r + gamma * max_a(Q'(s',a))
        not_final_mask = [s != None for s in transitions.next_state]
        t0 = time.perf_counter()
        next_state_reward_prediction: torch.Tensor = torch.zeros([batch_size], device=self.device)
        with torch.no_grad():
            next_state_reward_prediction[not_final_mask] = self.target_model(next_state_batch).max(dim=1).values.detach()

        expected_reward_batch = reward_batch + self.GAMMA * next_state_reward_prediction

        t_expected_computation = time.perf_counter() - t0

        t0 = time.perf_counter()

        self.tprof["tensor_conversion_time"].append(float(t_tensor_conversion))
        self.tprof["transition_zip_time"].append(float(t_transition_zip))
        self.tprof["prediction_time"].append(float(t_prediction))
        self.tprof["expected_computation_time"].append(float(t_expected_computation))

        t_statistics_computation = time.perf_counter() - t0
        self.tprof["statistics_computation_time"].append(float(t_statistics_computation))

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
            return random.sample(action_space, k=1)[0]
        else:
            self.model.eval()
            with torch.no_grad():
                return self.model(observations).max(0).indices.item()


    def store_transition(self, transition: Transition):
        self.memory.append(transition)
        self.action_count += 1
    

    def store_episode_statistics(self, statistics: dict):
        self.statistics["score"].append(statistics.get("score", 0))
        self.statistics["duration"].append(statistics.get("duration", 0))
        self.statistics["epsilon"].append(self.get_epsilon())
        self.statistics["loss"].append(statistics.get("loss", 0))
        self.statistics["memory_size"].append(len(self.memory))

    # Generic version of store_episode_statistics
    # def store_episode_statistics(self, statistics: dict):
    #     """
    #     Store episode statistics in the statistics dictionary.
    #     If the key does not exist, it will be created.
    #     """
    #     for key, value in statistics.items():
    #         if key not in self.statistics:
    #             self.statistics[key] = []
    #         self.statistics[key].append(value)

    
    def update_target_model(self):
        policy_state_dict = self.model.state_dict()
        target_state_dict = self.target_model.state_dict()
        for key in policy_state_dict.keys():
            target_state_dict[key] = utils.linear_interp(self.TAU, target_state_dict[key], policy_state_dict[key])
        self.target_model.load_state_dict(target_state_dict)
    
    # def plot_statistics(self):
    #     durations = np.array(self.statistics.duration)
    #     scores = np.array(self.statistics.score)
    #     x = range(len(durations))
        
    #     if self.fig is None:
    #         self.fig = plt.figure(1)

    #     self.fig.clf()
    #     ax = self.fig.subplots(2, 1)
    #     ax[0].set_title("Durations")
    #     ax[0].plot(x, durations, label="Durations")

    #     ax[0].set_title("Scores")
    #     ax[1].plot(x, scores, label="Scores")

    #     plt.pause(0.001)

    def plot_statistics(self):
        """
        Plot statistics collected during training
        """
        if self.fig is None:
            self.fig = plt.figure(1, figsize=(15, 10))
        self.fig.clf()
        ax = self.fig.subplots(len(self.statistics)//2 + 1, 2, sharex=True)

        x_axis = []
        sum = 0
        for x in self.statistics['duration']:
            sum += x
            x_axis.append(sum)


        for i, (key, values) in enumerate(self.statistics.items()):
            ax[i//2, i%2].set_title(key)
            ax[i//2, i%2].plot(x_axis, np.array(values), label=key)

        plt.pause(0.001)

        print("Tprofiler statistics:")
        for key, times in self.tprof.items():
            values = np.array(times)
            print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f} seconds")


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