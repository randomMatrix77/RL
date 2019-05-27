import numpy as np
import gym
import random

env = gym.make("Taxi-v2").env

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

print(qtable.shape)

total_eps = 15000
lr = 0.8
max_steps = 99
gamma = 0.95 #discount factor

# Exploration parameters

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay = 0.01

rewards = []

for ep in range(total_eps):

    penalties = 0

    state = env.reset()
    step = 0
    done = False
    tot_rewards = 0

    for step in range(max_steps):

        tradeoff = random.uniform(0, 1)

        if tradeoff > epsilon:

            action = np.argmax(qtable[state,:])

        else:

            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        if reward == -10:
            
            penalties += 1

        qtable[state, action] += lr * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        tot_rewards += reward

        state = new_state

        if done == True:

            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*ep)

    rewards.append(tot_rewards)

    if ep % 1000 == 0:

        print('Score : {}, Penalties: {}'.format(tot_rewards, penalties))
        env.render()

print('Final score : {}'.format(sum(rewards)/total_eps))
