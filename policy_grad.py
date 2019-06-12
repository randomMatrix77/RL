import gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)


## Hyperparameters:

#Enviroment
state_size = 4
action_size = env.action_space.n

#Model
max_episodes = 1
lr = 0.001
gamma = 0.95


## Discounted reward function:

def discounted_rewards_calc(episode_rewards):

    discounted_rewards = np.zeros_like(episode_rewards)
    add = 0

    for i in reversed(range(len(episode_rewards))):

        add = add * gamma + episode_rewards[i]
        discounted_rewards[i] = add

    mean = np.mean(discounted_rewards)
    std = np.std(discounted_rewards)

    discounted_rewards = (discounted_rewards - mean) / std

    return discounted_rewards


## Neural net for policy gradient(i.e. policy function!)

class Policy(nn.Module):

    def __init__(self):

        super(Policy, self).__init__()

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 2)

        self.act = nn.ReLU()
        self.soft = nn.Softmax()

        self.CE = nn.CrossEntropyLoss()

    def forward(self, inp):

        out = self.act(self.fc1(inp))
        out = self.act(self.fc2(inp))
        out = self.soft(self.fc3(inp))

        return out

    def loss(self, out, actions, discounted_rewards):

        nll = self.CE(out, actions)
        loss = nll * torch.FloatTensor(discounted_rewards)

        return loss


oracle = Policy()


for episode in range(max_episodes):

    total_reward = 0

    episode_states, episode_actions, apisode_rewards = [], [], []
    episode_action_probs = []

    state = env.reset()

    #env.render()

    while True:

        action_probs = oracle.forward(state)
        episode_action_probs.append(action_probs)
        
        action_probs = action_probs.detach().numpy()

        action = np.random.choice(range(action_probs.shape[1]), p = action_probs[0])

        new_state, reward, done, info = env.step(action)

        episode_states.append(state)

        action_ = np.zeros(action_size)
        action_[action] = 1

        episode_actions.append(action_)

        episode_rewards.append(reward)

        if done:

            total_reward = np.sum(episode_rewards)
            max_rewards = np.amax(episode_rewards)

            discounted_rewards = discounted_rewards_calc(episode_rewards)

            optim.zero_grad()
            loss = oracle.loss(episode_action_probs, episode_actions, discounted_rewards)
            loss.backward()
            optim.step()

            break
        
        state = new_state

        if episode % 100 == 0:

            print(episode, total_reward)

    

        
