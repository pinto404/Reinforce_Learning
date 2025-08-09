import gymnasium as gym
import torch
from torch import nn
from collections import deque
import numpy as np

class Actor_net(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_nodes=128):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, action_dim)

    def forward(self, x):
        x = torch.rrelu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
    

class Critic_net(nn.Module):
    def __init__(self, obs_dim, hidden_nodes=128):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class Agent():
    def __init__(self, obs_dim, action_dim):
        self.actor_net = Actor_net(obs_dim, action_dim)
        self.critic_net = Critic_net(obs_dim)

        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=0.0005)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=0.001)

        self.memory = deque(maxlen=1000)
        self.gamma = 0.99
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.actor_net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def store_memory(self, state, action, reward, next_state, log_prob, done):
        self.memory.append((state, action, reward, next_state, log_prob, done))

    def learn(self, done=False):
        if len(self.memory) < 20 and not done:
            return
        
        states, actions, rewards, next_states, log_probs, dones = zip(*self.memory)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        dones = torch.tensor(dones, dtype=torch.float32)

        critic_current = self.critic_net(states).squeeze()
        with torch.no_grad():
            critic_next = self.critic_net(next_states).squeeze()
            td_target = rewards + self.gamma * critic_next * (1 - dones)

        critic_loss = nn.MSELoss()(critic_current, td_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
        self.critic_optim.step()


        advantage = td_target - critic_current.detach()
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        actor_loss = (-log_probs * advantage).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        self.actor_optim.step()

        self.memory.clear()

env_name = "LunarLander-v3"

def train():
    env = gym.make(env_name)
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    
    rewards = deque(maxlen=100)
    episode = 0
    while(True):
        episode += 1
        episode_reward = 0
        step = 0
        state, _ = env.reset()
        while(True):
            step += 1
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            agent.store_memory(state, action, reward, next_state, log_prob, done or truncated)
            state = next_state  
            if done or truncated:
                break
            agent.learn()

        rewards.append(episode_reward)
        print(f"\repisode:{episode: 4d} step:{step: 5d} episode_reward{episode_reward: 6.1f} mean_reward:{np.mean(rewards): 6.1f}", end="")
        agent.learn(done=True)
        if np.mean(rewards) > 260:
            torch.save(agent.actor_net.state_dict(), "model/LL_A2C.pth")
            break

def test():
    env = gym.make(env_name, render_mode="human")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    agent.actor_net.load_state_dict(torch.load("model/LL_A2C.pth"))
  
    episode = 0
    for episode in range(1, 6):
        episode_reward = 0
        step = 0
        state, _ = env.reset()
        while(True):
            step += 1
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state  
            if done or truncated:
                break

        print(f"episode:{episode: 4d} step:{step: 5d} episode_reward{episode_reward: 6.1f} ")

#train()
test()

