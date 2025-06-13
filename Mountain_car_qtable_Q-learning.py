import gymnasium as gym
import random
import numpy as np

#wrap step() to difine original reward.
class wrapper_env(gym.Wrapper):    
    def step(self,action):
        state, reward, done, truncated, info = super().step(action)
        reward = -1 + 1*(state[0] + 0.5) ** 2 + done*200
        return state, reward, done, truncated, info
    
env_name = "MountainCar-v0"
env = gym.make(env_name)
env = wrapper_env(env)
observatin_box = env.observation_space  #[-1.2  -0.07] ~ [0.6  0.07]
action_dim     = env.action_space.n     #3

pos_div = 20    #split state[0] every 0.09
velo_div =10   

# Example: (state[0]= -1.2  →  index = 0), (state[0]= 0  → index = 13)
def select_pos_index(pos):
    index = int(min((pos - observatin_box.low[0]) // (abs(observatin_box.low[0] - observatin_box.high[0]) / pos_div), pos_div-1))
    return index

def select_velo_index(velo):
    index = int(min((velo - observatin_box.low[1]) // (abs(observatin_box.low[1] - observatin_box.high[1]) / velo_div), velo_div-1))
    return index

alpha = 0.05
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.99
gamma = 0.99

#shape[pos_div, velo_div, action_dim]
q_table = np.array([[[0.0 for _ in range(action_dim)] for _ in range(velo_div)] for _ in range(pos_div)])

def select_action(state, q_table, epsilon):
    if random.random() < epsilon or sum(q_table[state[0]][state[1]]) == 0:
        return random.randint(0, action_dim - 1)
    action = np.argmax(q_table[state[0]][state[1]])
    return  action

def update_q_table(state, action, next_state,  reward):
    loss = (reward + gamma * (np.max(q_table[next_state[0]][next_state[1]])) - q_table[state[0]][state[1]][action])
    q_table[state[0]][state[1]][action] += alpha * loss
    return loss

#training might not complete in 2000 episodes
for episode in range(2000):
    total_reward = 0
    total_loss   = 0
    step = 0
    state, _ = env.reset()
    state = (select_pos_index(state[0]), select_velo_index(state[1]))
    while(True):
        step += 1
        action = select_action(state, q_table, epsilon)
        next_state, reward, done, truncated, info= env.step(action)
        next_state = (select_pos_index(next_state[0]), select_velo_index(next_state[1]))
        total_loss += update_q_table(state, action, next_state,reward)
        state = next_state
        total_reward += reward

        if done or truncated:
            break
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
    print(f"\repisode: {episode+1:4d}  total_reward: {total_reward:7.2f}  step: {step} total_loss: {total_loss:.2f}", end="")


print(f"\n---TEST---")
env = gym.make(env_name, render_mode="human") 
for test_episode in range(5):
    total_reward = 0
    step = 0
    state, _ = env.reset()
    state = (select_pos_index(state[0]), select_velo_index(state[1]))
    while(True):
        step += 1
        action = select_action(state, q_table, epsilon=0)
        next_state, reward, done, truncated, info= env.step(action)
        next_state = (select_pos_index(next_state[0]), select_velo_index(next_state[1]))
        state = next_state
        total_reward += reward

        if done or truncated:
            break
    print(f"episode: {test_episode+1}  total_reward: {total_reward:.2f}  step: {step}") #end within 200steps, it's a succeess

