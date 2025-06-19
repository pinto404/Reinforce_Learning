import gymnasium as gym
import torch
from torch import nn
import numpy as np
from collections import deque
from torch.distributions import Categorical

class Policy_net(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Policy_net, self).__init__()
        self.hidden_nodes = 48
        self.fc1 = nn.Linear(observation_dim, self.hidden_nodes)
        self.fc2 = nn.Linear(self.hidden_nodes, action_dim)
    
    def forward(self,x):
        x = torch.tanh(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)
    
class Agent():
    def __init__(self, observation_dim, action_dim):
        self.policy_net = Policy_net(observation_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.005)
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
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss)   

        episode_entropy = torch.tensor(0.0)
        entropies = [Categorical(logits=all_prob.squeeze(0)).entropy() for all_prob in self.all_log_probs]
        episode_entropy = torch.stack(entropies).sum()

        final_loss = policy_loss.sum() - self.entropy_coeff * episode_entropy

        self.optimizer.zero_grad()
        final_loss.backward()
        self.optimizer.step()

        self.log_probs , self.reward_history, self.all_log_probs = [], [], []

env_name = "CartPole-v1"
env = gym.make(env_name)
agent = Agent(env.observation_space.shape[0], env.action_space.n)

def train():
    rewards = deque(maxlen=100)
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
            if done or truncated:
                break
        rewards.append(total_reward)
        print(f"\repisode:{episode}  reward:{total_reward:.2f}  mean:{np.mean(rewards):.2f}  step:{step}", end="")
        agent.learn()
        if np.mean(rewards) > 490:
            break

def test(test_episode):
    env = gym.make(env_name, render_mode = "human")
    print(f"\n---TEST---")
    episode = 0
    for episode in range(test_episode):
        step = 0
        total_reward = 0
        state, _ = env.reset()
        while True:
            step += 1
            action = agent.test_action(state)
            if step % 100 >= 96:action = step // 100 % 2
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            if done or truncated:
                break
        print(f"episode:{episode+1}  reward:{total_reward:.2f}   step:{step}")

train()
test(5)