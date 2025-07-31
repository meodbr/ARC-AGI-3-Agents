import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import DQNModel, Memory

env = gym.make("CartPole-v1")

class FFN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden_size = 1024
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(x, x_hat)

class CartPoleAgent(DQNModel):
    def __init__(self):
        state, info = env.reset()
        model_class = FFN
        memory = Memory(20000)
        model_instanciation_args = {
            "input_size": len(state),
            "output_size": env.action_space.n,
        }
        super().__init__(model_class, memory, model_instanciation_args)
    
    def training_loop(self):
        state = torch.tensor(env.reset()[0], device=self.device, dtype=torch.float32)
        last_state = state

        turn_count = 0
        while 1:
            last_state = state
            action = self.select_action(state, range(env.action_space.n))
            observation, reward, terminated, truncated, info = env.step(action)
            state = torch.tensor(observation, device=self.device, dtype=torch.float32)

            if terminated:
                state = None
                self.store_transition((last_state, action, state, reward))
                self.store_episode_statistics({
                    "score": 0,
                    "duration": turn_count,
                    "loss": 0,  # Placeholder, loss is not computed here
                })
                self.plot_statistics()
                state = torch.tensor(env.reset()[0], device=self.device, dtype=torch.float32)
                turn_count = 0
            else:
                turn_count += 1
                self.store_transition((last_state, action, state, reward))

            self.train_step(batch_size=512)

if __name__ == "__main__":
    ag = CartPoleAgent()
    ag.training_loop()


