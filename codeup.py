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

# Keep track of all the rewards for each episode
rewards_all_episodes = []

# Random agent
for episode in range(num_episodes):
    state = env.reset() # Go back to starting state
    done = False
    rewards_current_episode = 0 # Keeps track of rewards in current episode

    for step in range(max_steps_per_episode):
        # Pick an action
        # Decide to explore or exploit randomly
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            # Get all estimated rewards for each action of a given state and pick action with the highest reward
            # Numpy argmax returns the indices of the maximum values along an axis.
            available_actions_q_values = q_table[state]
            action = np.argmax(available_actions_q_values)
        else:
            action = env.action_space.sample()

        # do the action and note the next state, reward etc
        new_state, reward, done, info = env.step(action)

        # Q value for current state and action pair
        current_q_value = q_table[state, action]

        # Given the new state find the q value of the best action according to our current predictions
        best_future_q_value = np.max(q_table[new_state])

        # Update the q table for current the state and action we just took
        # Look at the reward we just got for performing this action and add the estimated reward for the best next
        # possible action to the sum. Modify the values by the learning_rate
        q_table[state, action] = \
            current_q_value * (1 - learning_rate) + learning_rate * (reward + discount_rate * best_future_q_value)

        # Set the current state to the new state
        state = new_state

        # Note the reward
        rewards_current_episode += reward

        # Finish the episode sooner than max_steps_per_episode is something happened aka agent died/lost already
        if done:
            break

    # After the episode is done decrease the exploration rate
    exploration_rate = \
        min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
            -exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

# Print updated  q table
print("\n\n***** Q-table *******\n")
print(q_table)

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("**** Average reward per thousand episodes ****\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
