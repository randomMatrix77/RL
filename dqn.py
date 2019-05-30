from vizdoom import *
import numpy as np
from skimage import transform
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import deque
import random
import time
import matplotlib.pyplot as plt

memory_size = 1000
batch_size = 64
lr = 1e-5
pre_train = batch_size
render_frame = False
stack_size = 4
episodes = 500
sim_games = 10
max_steps = 100
train = True
gamma = 0.95
decay_rate = 0.0001
explore_start = 1.0
explore_stop = 0.01


def create_env(render_frame):

    game = DoomGame()

    game.load_config("torch/RL/doom_cfg/basic.cfg")

    game.set_doom_scenario_path("torch/RL/doom_cfg/basic.wad")

    game.init()

    game.set_window_visible(render_frame)

    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]

    possible_actions = [left, right, shoot]

    return game, possible_actions

def test_environment():
    game = DoomGame()
    game.load_config("/home/kinngmax/Documents/torch/RL/doom_cfg/basic.cfg")
    game.set_doom_scenario_path("/home/kinngmax/Documents/torch/RL/doom_cfg/basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print ("\treward:", reward)
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()

def preprocess(frame):

    if len(frame.shape) == 3:
        
        frame = np.mean(frame, 0)

    cropped = frame[30:-10, 30:-30]

    frame = transform.resize(cropped, [84, 84])

    return frame/255

def stack_frames(stacked_frames, state, is_new_episode):

    frame = preprocess(state)

    if is_new_episode:

        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        for i in range(4):
            stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis = 0)

    else:

        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis = 0)

    return stacked_state, stacked_frames


class DQN(nn.Module):

    def __init__(self, batch_size):

        super(DQN, self).__init__()

        self.batch_size = batch_size
        
        self.conv1 = nn.Conv2d(4, 32, 8, stride = 2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 4, stride = 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.act = nn.ELU()

        self.l1 = nn.Sequential(self.conv1, self.bn1, self.act)
        self.l2 = nn.Sequential(self.conv2, self.bn2, self.act)
        self.l3 = nn.Sequential(self.conv3, self.bn3, self.act)

        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 3)

        self.sig = nn.Sigmoid()

        self.criterion = nn.MSELoss()

    def forward(self, inp):

        inp = torch.Tensor(inp)

        if len(inp.shape) == 3:
            inp = inp.view(1, 4, 84, 84)

        out = self.l1(inp)
        out = self.l2(out)
        out = self.l3(out)

        if inp.shape[0] == 1:
            flat = out.view(1, -1)
        else:
            flat = out.view(self.batch_size, -1)
        
        out = self.act(self.fc1(flat))
        out = self.sig(self.fc2(out))

        return out

    def train(self, pred, target, optim):

        target = torch.FloatTensor(target)
        target = target.view(self.batch_size, -1)

        optim.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        optim.step()

        return loss.item()


class Memory():

    def __init__(self, max_size = 64):

        self.buffer = deque(maxlen = max_size)

    def add(self, experience):

        self.buffer.append(experience)

    def sample(self, batch_size):

        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)

        return [self.buffer[i] for i in index]

def pretrain(pre_tran = pre_train):
    
    for i in range(pre_train):

        if i == 0:

            state = game.get_state().screen_buffer
            stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        action = random.choice(possible_actions)
        reward = game.make_action(action)
        done = game.is_episode_finished()

        if done:

            next_state = np.zeros(state.shape)
            memory.add((state, action, reward, next_state, done))

            game.new_episode()

            state = game.get_state().screen_buffer

            state, stacked_frames = stack_frames(stacked_frames, state, done)

        else:

            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            memory.add((state, action, reward, next_state, done))

            state = next_state

def predict_action(explore_start, explore_stop, decay, decay_rate, state, possible_actions, net):

    exp_tradeoff = np.random.rand()

    exp_prob = explore_stop + (explore_stop - explore_start) * np.exp(-decay_rate * decay)

    if exp_prob > exp_tradeoff:

        action = random.choice(possible_actions)

    else:

        Qs = net.forward(state)
        action = possible_actions[int(Qs.argmax())]

    return action, exp_prob

def sim_game(sim_games):

    game, possible_actions = create_env(rende_frame)

    game.init()

    net = torch.load('')

    for i in range(sim_games):

        total_score = 0

        game.new_episode()

        state = game.get_state().screen_buffer
        
        stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not game.is_episode_finished():

            Qs = net.forward(state)

            action = possible_actions[int(Qs.argmax())]
            game.make_action(action)

            score = game.get_total_reward()

            total_score += score


        print('Game {} ended. Total score : {}'.format(i+1, total_score))



net = DQN(batch_size)
optim = torch.optim.Adam(net.parameters(), lr)

memory = Memory(max_size = memory_size)

game, possible_actions = create_env(render_frame)
game.new_episode()
pretrain(pre_train)

def agent(train):

    if train == True:

        decay = 0

        game.init()

        for ep in range(episodes):

            step = 0

            ep_reward = 0

            game.new_episode()

            state = game.get_state().screen_buffer
            stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:

                step += 1

                decay += 1

                action, exp_prob = predict_action(explore_start, explore_stop, decay, decay_rate, state, possible_actions, net)

                reward = game.make_action(action)

                done = game.is_episode_finished()

                ep_reward += reward

                if done:

                    next_state = np.zeros(state.shape)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, done)

                    step = max_steps

                    memory.add((state, action, reward, next_state, done))

                else:


                    next_state = game.get_state().screen_buffer

                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, done)

                    memory.add((state, action, reward, next_state, done))

                    state = next_state


                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin = 3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_state_mb = np.array([each[3] for each in batch], ndmin = 3)
                done_mb = np.array([each[4] for each in batch])

                Qs_next_state = net.forward(next_state_mb)

                Qs_next_state = Qs_next_state.detach().numpy()
                
                target_Qs_mb = []

                for i in range(len(batch)):

                    terminal = done_mb[i]

                    if terminal:

                        target_Qs_mb.append(np.ones(Qs_next_state[i].shape)*rewards_mb[i])

                    else:

                        target = rewards_mb[i] + gamma * Qs_next_state[i]
                        
                        target_Qs_mb.append(target)

                targets_mb = np.array([each for each in target_Qs_mb])

                Qs_current_state = net.forward(states_mb)

                loss = net.train(Qs_current_state, targets_mb, optim)

            if ep % 50 == 0:

                #torch.save(net, '.pt')
                print('Episode : {}, Reward : {}, Loss : {}'.format(ep, ep_reward, loss))

    else:

        sim_game(sim_games)
    

                
