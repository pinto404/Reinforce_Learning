import gymnasium as gym
import numpy as np
import random

env_name = "CliffWalking-v0"
env = gym.make(env_name)

print(env.observation_space, env.action_space) # Discreate 48, Discreate 4

q_table = np.array([[0 for _ in range(env.action_space.n)] for _ in range(env.observation_space.n)])

