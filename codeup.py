# Import libraries
import numpy as np # Numpy makes python do better at maths
import gym # OpenAI Gym for our environment
import random # Random number genrator to let the fate decide
import time # Allows our program to sleep so things slow down a little


# Initialise our environment
env = gym.make("FrozenLake-v0")

# Render the current board state to the console
env.render()

# Get the number of possible moves
action_space_size = env.action_space.n
print(action_space_size)

# Get the number of tiles
observation_space_size = env.observation_space.n

# Initialise our Q Table
q_table = np.zeros((observation_space_size, action_space_size))

print(q_table)
