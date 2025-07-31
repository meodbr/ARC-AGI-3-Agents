import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import DQNModel, Memory

env = gym.make("CartPole-v1")

class FFN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        print(f"model returns: {x}")
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
    
    def training_loop(self):
        state = torch.Tensor(env.reset()[0])
        last_state = state

        turn_count = 0
        while 1:
            last_state = state
            action = self.select_action(state, range(env.action_space.n))
            print(action)
            observation, reward, terminated, truncated, info = env.step(int(action))
            print(observation)
            state = torch.Tensor(observation)
            print(state)

            if terminated:
                state = None
                self.store_transition((last_state, action, state, reward))
                self.store_episode_statistics(duration=turn_count, score=0)
                self.plot_statistics()
                state = torch.Tensor(env.reset()[0])
                turn_count = 0
            else:
                turn_count += 1
                self.store_transition((last_state, action, state, reward))

            self.train_step()

if __name__ == "__main__":
    ag = CartPoleAgent()
    ag.training_loop()


