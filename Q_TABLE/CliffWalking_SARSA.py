import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

env_name = "CliffWalking-v0"
env = gym.make(env_name)
action_dim = env.action_space.n

alpha = 0.05
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.99

print(env.observation_space, env.action_space) # Discreate 48, Discreate 4

q_table = np.zeros((env.observation_space.n, action_dim))

def select_action(state, epsilon = epsilon):
    if random.random() < epsilon or np.sum(q_table[state]) == 0:
        return random.randint(0, action_dim-1)
    return np.argmax(q_table[state])

def update_q_table(state, action , reward, next_state, next_action):
    if next_action != None:
        loss = reward + gamma * (q_table[next_state, next_action] - q_table[state, action])
    else:
        loss = reward + gamma * (0 - q_table[state, action])
    q_table[state, action] += alpha * loss

def train(train_episode = 1000, epsilon = epsilon):
    for episode in range(train_episode):
        step = 0
        total_reward = 0
        state, _  = env.reset()
        action = select_action(state, epsilon)
        while(True):
            step += 1
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated or step>=500: 
                update_q_table(state, action , reward, next_state, None)
                break
            next_action = select_action(next_state, epsilon)
            update_q_table(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
        rewards.append(total_reward)
        print(f"\repisode:{episode+1:4d} total_reward:{total_reward:.2f} final_step:{step:3d}", end="")

rewards = []
def test(test_episode=5):
    print(f"\n---TEST---")
    for episode in range(test_episode):
        step = 0
        total_reward = 0
        state, _  = env.reset()
        while(True):
            step += 1
            action = select_action(state, epsilon=0)
            next_state, reward, done ,truncated, info = env.step(action)
            total_reward += reward
            if done or truncated: 
                break
            state = next_state

        print(f"episode:{episode+1:4d} total_reward:{total_reward:.2f} final_step:{step:3d}") #goal at 17 steps, it's a sucess

train()
plt.plot(range(1,len(rewards)+1), rewards)
plt.show()

env = gym.make(env_name, render_mode = "human")
test()