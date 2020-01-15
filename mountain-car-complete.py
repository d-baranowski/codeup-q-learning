import numpy as np
import gym
import random
import time


env = gym.make("MountainCar-v0")
action_space_size = env.action_space.n

# Because there are two values observable in the environment and we want to dissect the ranges into 20 hunks
discrete_os_size = [14] * 2
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

# Shape
q_table = np.zeros((discrete_os_size[0], discrete_os_size[1], action_space_size))

# Max Number of turns in a game before terminating the episode
max_steps_per_episode = 99999

# Functions
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

def watch_x_episodes(x):
    for episode in range(x):
        # initialize new episode params
        state = env.reset()
        discrete_state = get_discrete_state(state)
        available_actions_q_values = q_table[discrete_state]

        print("*****EPISODE ", episode + 1, "*****\n")
        time.sleep(1)

        for step in range(max_steps_per_episode):
            # Show current state of environment on screen
            env.render()

            # Choose action with highest Q-value for current state
            action = np.argmax(available_actions_q_values)

            # Take new action
            new_state, reward, done, info = env.step(action)

            # Agent stepped in a hole and lost episode
            if done:
                break

            # Set new state
            state = new_state
            discrete_state = get_discrete_state(state)
            available_actions_q_values = q_table[discrete_state]
    env.close()

watch_x_episodes(1)

# Number of games to play
num_episodes = 5000


# Weight used to calculate each new q value based of the previous q value for that state action pair
learning_rate = 0.1
# How much less do we care about the future when calculating the max reward
discount_rate = 0.95

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001

# Keep track of all the rewards for each episode
rewards_all_episodes = []

# Train with Q Learning Algorithm
for episode in range(num_episodes):
    discrete_state = get_discrete_state(env.reset())

    done = False
    rewards_current_episode = 0 # Keeps track of rewards in current episode

    if episode % 100 == 0:
        print("Training Progress episode " + str(episode))

    for step in range(max_steps_per_episode):
        # Decide to explore or exploit randomly
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            # Get all estimated rewards for each action of a given state and pick action with the highest reward
            # Numpy argmax returns the indices of the maximum values along an axis.
            available_actions_q_values = q_table[discrete_state]
            action = np.argmax(available_actions_q_values)
        else:
            action = env.action_space.sample()

        new_state, reward, done, test = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q_value = q_table[discrete_state][action]

            # And here's our equation for a new Q value for current state and action
            new_q_value =\
                (1 - learning_rate) * current_q_value + learning_rate * (reward + discount_rate * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state][action] = new_q_value

        discrete_state = new_discrete_state
        # Note the reward
        rewards_current_episode += reward

        if done:
            break

    # After the episode is done decrease the exploration rate
    exploration_rate =\
        min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

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

# Watch our agent play Frozen Lake by playing the best action
# from each state according to the Q-table
watch_x_episodes(3)

