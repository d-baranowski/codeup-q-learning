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

# Get the number of tiles
observation_space_size = env.observation_space.n

# Initialise our Q Table
q_table = np.zeros((observation_space_size, action_space_size))

print(q_table)

# Max Number of turns in a game before terminating the episode
max_steps_per_episode = 50

# Number of games to play
num_episodes = 6000

# Weight used to calculate each new q value based of the previous q value for that state action pair
learning_rate = 0.1
# How much less do we care about the future when calculating the max reward
discount_rate = 0.99

# Initial exploration rate aka How little do we trust our selves how much do we trial and error ?
max_exploration_rate = 1

# Whats the absolute maximum level of trust we're willing to achieve
min_exploration_rate = 0.00001
exploration_rate = max_exploration_rate

# How quickly to we gain trust in ourselves
exploration_decay_rate = 0.001

# Random agent
for episode in range(num_episodes):
    state = env.reset() # Go back to starting state
    done = False
    rewards_current_episode = 0 # Keeps track of rewards in current episode

    for step in range(max_steps_per_episode):
        # Pick a random action
        action = env.action_space.sample()

        # do the action and note the next state, reward etc
        new_state, reward, done, info = env.step(action)

        # Print the board to the console
        env.render()

        # Sleep a little so the board does not print to quickly for us to follow
        time.sleep(0.3)

        # Set the current state to the new state
        state = new_state

        # Finish the episode sooner than max_steps_per_episode is something happened aka agent died/lost already
        if done:
            if reward == 1:
                # Agent reached the goal and won episode
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
            break
