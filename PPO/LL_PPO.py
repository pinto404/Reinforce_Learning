import gymnasium as gym
import torch 
from torch import nn
from collections import deque
import numpy as np
import os


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

class actor_net(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(actor_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.rrelu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
    
class critic_net(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super(critic_net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class Agent():
    def __init__(self, obs_dim, action_dim):
        self.device = device
        self.actor_net = actor_net(obs_dim, action_dim).to(self.device)
        self.critic_net = critic_net(obs_dim).to(self.device)

        self.actor_net_old = actor_net(obs_dim, action_dim).to(self.device)
        self.actor_net_old.load_state_dict(self.actor_net.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=0.0003)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=0.001)

        self.memory = deque()
        self.gamma = 0.99
        self.policy_clip = 0.3
        self.k_epochs = 4

    def select_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            probs = self.actor_net_old(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def store_memory(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def learn(self):
        if not self.memory:
            return

        states, actions, rewards, next_states, dones, log_probs = zip(*self.memory)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        log_probs = torch.stack(list(log_probs), dim=0).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            critic_current = self.critic_net(states).squeeze()
            critic_next = self.critic_net(next_states).squeeze()

            deltas = rewards + self.gamma * critic_next * (1-dones) - critic_current
            advantages = torch.zeros_like(rewards).to(self.device)
            last_gae_lambda = 0
            for t in reversed(range(len(rewards))):
                last_gae_lambda = deltas[t] + self.gamma * 0.95 * (1-dones[t]) * last_gae_lambda
                advantages[t] = last_gae_lambda
            
            td_targets = advantages + critic_current
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        

        for _ in range(self.k_epochs):
            probs_new = self.actor_net(states)
            dist_new = torch.distributions.Categorical(probs_new)
            log_probs_new = dist_new.log_prob(actions)
            dist_entropy = dist_new.entropy()

            ratio = torch.exp(log_probs_new - log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
            
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            critic_current = self.critic_net(states).squeeze()
            critic_loss = nn.MSELoss()(critic_current, td_targets)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            

            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
            

        self.actor_net_old.load_state_dict(self.actor_net.state_dict())
        self.memory.clear()


env_name = "LunarLander-v3"

def train():
    env = gym.make(env_name)
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    learn_interval = 2048
    T_step = 0
    
    if not os.path.exists("model"):
        os.makedirs("model")

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
            agent.store_memory(state, action, reward, next_state, done or truncated, log_prob)
            T_step += 1
            if T_step >= learn_interval:
                agent.learn()
                T_step = 0
            state = next_state 
            if done or truncated:
                break
            

        rewards.append(episode_reward)
        print(f"\repisode:{episode: 4d} step:{step: 5d} episode_reward{episode_reward: 6.1f} mean_reward:{np.mean(rewards): 6.1f}", end="")
        if np.mean(rewards) > 260:
            torch.save(agent.actor_net_old.state_dict(), "model/LL_PPO.pth")
            break

def test():
    env = gym.make(env_name, render_mode="human")
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    agent.actor_net_old.load_state_dict(torch.load("model/LL_PPO.pth", map_location=device))

    episode = 0
    for episode in range(1, 6):
        episode_reward = 0
        step = 0
        state, _ = env.reset()
        while(True):
            step += 1
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = torch.argmax(agent.actor_net_old(state_tensor)).item()
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state 
            if done or truncated:
                break

        print(f"episode:{episode: 4d} step:{step: 5d} episode_reward{episode_reward: 6.1f} ")

#train()
test()
