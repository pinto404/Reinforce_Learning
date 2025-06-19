import gymnasium as gym
import torch
from torch import nn
import numpy as np
from collections import deque
from torch.distributions import Categorical
import time

class Policy_net(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Policy_net, self).__init__()
        self.hidden_nodes = 24
        self.fc1 = nn.Linear(observation_dim, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, action_dim)
    
    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)
    
class Agent():
    def __init__(self, observation_dim, action_dim):
        self.policy_net = Policy_net(observation_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.log_probs = []
        self.all_log_probs = []
        self.reward_history = []
        self.gamma = 0.99
        self.entropy_coeff = 0.0001

    def select_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_net(state_tensor)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        self.all_log_probs.append(torch.exp(probs))
        return action.item()
    
    def test_action(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        return torch.argmax(self.policy_net(state_tensor)).item()
    
    def learn(self):
        R = 0
        returns = []
        for r in reversed(self.reward_history):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        #policy_loss = torch.stack(policy_loss)   
        policy_loss = torch.cat(policy_loss)

        episode_entropy = torch.tensor(0.0)
        entropies = [Categorical(logits=all_prob.squeeze(0)).entropy() for all_prob in self.all_log_probs]
        episode_entropy = torch.stack(entropies).sum()

        final_loss = policy_loss.sum() - self.entropy_coeff * episode_entropy

        self.optimizer.zero_grad()
        #final_loss.backward()
        policy_loss.sum().backward()
        self.optimizer.step()

        self.log_probs , self.reward_history, self.all_log_probs = [], [], []

class wrapper_env(gym.Wrapper):    
    def step(self,action):
        state, reward, done, truncated, info = super().step(action)
        #reward = -1 + 1*(state[0] + 0.5) ** 2 + done*200
        return state, reward, done, truncated, info

env_name = "MountainCar-v0"
env = gym.make(env_name)
env = wrapper_env(env)
agent = Agent(env.observation_space.shape[0], env.action_space.n)

def train():
    timer = time.time()
    steps = deque(maxlen=100)
    episode = 0
    while True:
        episode += 1
        step = 0
        total_reward = 0
        state, _ = env.reset()
        while True:
            step += 1
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.reward_history.append(reward)
            total_reward += reward
            state = next_state
            if done:
                break
        steps.append(step)
        print(f"\repisode:{episode:4d}  reward:{total_reward:10.2f}  mean:{np.mean(steps):8.2f}  step:{step:6d}", end="")
        agent.learn()
        if np.mean(steps) < 120:
            break
    timer = time.time() - timer
    print(f"\nTIME_ELAPSED: {timer:.2f}")

def test(test_episode):
    env = gym.make(env_name, render_mode = "human")
    print("---TEST---")
    episode = 0
    for episode in range(test_episode):
        step = 0
        total_reward = 0
        state, _ = env.reset()
        while True:
            step += 1
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            if done or truncated:
                break
        print(f"episode:{episode+1}  reward:{total_reward:.2f}   step:{step}")

train()
test(5)