import gymnasium as gym
from   gymnasium import Wrapper
import torch
from   torch     import nn
from   collections import deque
import random

    
class Model(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Model, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, state):
        return self.network(state)

''' 
model = Model(5,4)
state = [1,2,3,4,5]
state_tensor = torch.tensor(state, dtype=torch.float32)
print(model(state_tensor), torch.argmax(model(state_tensor)), torch.argmax(model(state_tensor)).item())
'''

class Agent():
    def __init__(self, observation_dim, action_dim):
        self.action_dim = action_dim
        self.model = Model(observation_dim, action_dim)
        self.target_model = Model(observation_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.memory = deque(maxlen=10000)

        self.epsilon = 1.0
        self.min_epsilon = 0.02
        self.epsilon_decay = 0.9
        self.gamma = 0.99

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()
        
    def store_exprience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self, batch_size = 32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tenor        = torch.tensor(states,          dtype=torch.float32)
        actions_tensor      = torch.tensor(actions,         dtype=torch.int64  ).unsqueeze(1)
        rewards_tensor      = torch.tensor(rewards,         dtype=torch.float32)      
        next_states_tensor = torch.tensor(next_states,    dtype=torch.float32)
        dones_tensor        = torch.tensor(dones,           dtype=torch.float32)  

        q_tensor = self.model(states_tenor).gather(1, actions_tensor).squeeze() #(batchsize, )

        next_q_values = self.target_model(next_states_tensor).max(1)[0]
        target_q_tensor = rewards_tensor + self.gamma * next_q_values * (1 - dones_tensor)

        loss = nn.MSELoss()(q_tensor, target_q_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

batch_size = 32
env_name = "MountainCar-v0"
env = gym.make(env_name)
agent = Agent(env.observation_space.shape[0], env.action_space.n)

def train(train_episode = 1000):
    for episode in range(train_episode):
        step = 0
        total_reward = 0
        state, _ = env.reset()
        while(True):
            step += 1
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            agent.store_exprience(state, action, reward, next_state, done)
            if step % 5 == 1:
                agent.train_model(batch_size)
            state = next_state
            if done or truncated:
                break
        agent.update_target_model()
        agent.epsilon = max(agent.min_epsilon, agent.epsilon*agent.epsilon_decay)
        print(f"\repisode:{episode+1} total_reward:{total_reward:7.2f} step:{step} epsilon:{agent.epsilon:.2f}", end="")

def test(test_episode = 5):
    print(f"\n---TEST---")
    env = gym.make(env_name, render_mode="human")
    agent.epsilon = 0
    for episode in range(test_episode):
        step = 0
        total_reward = 0
        state, _ = env.reset()
        while(True):
            step += 1
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            if done or truncated:
                break
        print(f"episode:{episode+1} total_reward:{total_reward:7.2f} step:{step}")


train()
test()