import numpy as np
import gym
import random
import time
from IPython.display import clear_output


env = gym.make("MountainCar-v0")
action_space_size = env.action_space.n

# Because there are two values observable in the environment and we want to disect the ranges into 20 hunks
discrete_os_size = [20] * 2
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

q_table = np.zeros((discrete_os_size[0], discrete_os_size[1] ,action_space_size))


num_episodes = 10000 # Number of games to play
max_steps_per_episode = 150 # Max Number of turns in a game before terminating the episode

learning_rate = 0.1 # Weight used to calculate each new q value based of the previous q value for that state action pair
discount_rate = 0.99 # How much less do we care about the future when calculating the max reward

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.00001
exploration_decay_rate = 0.001

# Keep track of all the rewards for each episode
rewards_all_episodes = []

# Q Learning Algorithm
for episode in range(num_episodes):
    state = env.reset() # Go back to starting state
    done = False
    rewards_current_episode = 0 # Keeps track of rewards in current episode

    for step in range(max_steps_per_episode):
        # Decide to explore or exploit randomly
        exploration_rate_threshold = random.uniform(0, 1)
        if (exploration_rate_threshold > exploration_rate):
            action = np.argmax(q_table[state,:]) # Get all estimated rewards for each action of a given state and pick the one with the highest estimated reward
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action) # do the action and note the next state, reward etc

        # Update the q table for current the state and action we just took
        # Look at the reward we just got for performing this action and add the estimated reward for the best next
        # possible action to the sum. Modify the values by the learning_rate
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * \
                                 (reward + discount_rate * np.max(q_table[new_state, :]))

        # Set the current state to the new state
        state = new_state
        # Note the reward
        rewards_current_episode += reward

        # Finish the episode sooner than max_steps_per_episode is something happened aka agent died/lost already
        if done:
            break


    # After the episode is done decrease the exploration rate
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards_all_episodes.append(rewards_current_episode)


rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("**** Average reward per thousand episodes ****\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated  q table
print("\n\n***** Q-table *******\n")
print(q_table)

