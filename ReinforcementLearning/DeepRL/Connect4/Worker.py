from kaggle_environments import evaluate, make, utils

import termcolor
import colorama
import torch.multiprocessing as mp
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
device = torch.device('cpu')
import numpy as np
import random

class ValueModel(nn.Module):
    def __init__(self, n_outputs, lr=0.01):
        super(ValueModel, self).__init__()
        self.lr = lr
        self.n_outputs = n_outputs

        self.cnn = nn.Sequential(
            nn.Linear(42, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, n_outputs)
        ).to(device=device)


        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, eps=0.001)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.criterion = torch.nn.SmoothL1Loss()

    def forwards(self, X):
        state_t = torch.tensor(X, dtype=torch.float32).to(device=device)
        return self.cnn(state_t)

    def predict(self, X):
        return self.forwards(X).cpu().detach().numpy()

    def partial_fit(self, X, Y, i):
        self.optimizer.zero_grad()

        prediction = self.forwards(X)
        prediction = prediction[torch.arange(len(prediction)), i]

        target = torch.tensor(Y, dtype=torch.float32).to(device=device)
        loss = self.criterion(prediction, target)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        self.optimizer.step()

    def copy(self, copy):
        for (i, (me, them)) in enumerate(zip(self.parameters(), copy.parameters())):
            me.data = them.data.clone()


class ExperienceReplay:
    def __init__(self, max_size, min_size, batch_size):
        self.max_size = max_size
        self.min_size = min_size
        self.batch_size = batch_size
        self.n = 0
        self.current = 0

        self.action_buffer = np.zeros(max_size)
        self.reward_buffer = np.zeros(max_size)
        self.done_buffer = np.zeros(max_size)
        self.frame_buffer = np.zeros((max_size, 42))

    def add(self, frame, action, reward, done):
        self.action_buffer[self.current] = action
        self.frame_buffer[self.current, ...] = frame
        self.reward_buffer[self.current] = reward
        self.done_buffer[self.current] = done
        self.n = max(self.n, self.current + 1)
        self.current = (self.current + 1) % self.max_size

    def get_batch(self):
        if self.n < self.min_size:
            return None
        elif self.n == self.min_size:
            print("\n\n\n\{} frames in {} s\n\n\n".format(self.n, 0))

        indx = np.random.choice(np.arange(start=0, stop=self.n - 1), replace=False, size=self.batch_size)

        actions = self.action_buffer[indx]
        rewards = self.reward_buffer[indx]
        dones = self.done_buffer[indx]
        states = self.frame_buffer[indx, :]
        next_states = self.frame_buffer[indx + 1, :]

        return actions, rewards, dones, states, next_states

    def get_last_state(self):
        return self.frame_buffer[self.current]


class ConnectAgent:
    def __init__(self, env, value, experience, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.V = value
        self.experience = experience

    def chose_action(self, s, e):
        if random.random() < e:
            return random.randint(1,6)
        else:
            state = self.experience.get_last_state()
            if state is not None:
                pred =self.V.predict(state)
                return np.argmax(pred)
            else:
                return  random.randint(1,6)

    def train(self, target_network):

        batch = self.experience.get_batch()

        if batch is None:
            return

        actions, rewards, dones, states, next_states = batch
        next_q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self.gamma * next_q if not done else r for (r, next_q, done) in zip(rewards, next_q, dones)]
        self.V.partial_fit(states, targets, actions)


    def play_one(self, target_network, t, min_size, update_freq=200, epsilon=0, epsilon_change=0, epsilon_min=0,
                 train_freq=1):
        episode_reward = 0
        training_time = 0
        image_time = 0

        s = self.env.reset().board
        while True:

            a = int(self.chose_action(s, epsilon))

            s_, r, done, info = self.env.step(a)

            if r is None:
                r = -10


            episode_reward += r

            self.experience.add(s, a, r, done)

            if t % train_freq == 0:
                self.train(target_network)
            if t > min_size and t % update_freq == 0:
                target_network.copy(self.V)
            s = s_.board

            epsilon = max(epsilon - epsilon_change, epsilon_min)

            if done:
                break
            t += 1

        return episode_reward, training_time * 1e-6, image_time * 1e-6, epsilon, t

def train(agent, target_network, min_size, episodes=1000, update_freq=1000, pritn_freq=1, train_freq=1):
    av_reward = 0
    av_train_time = 0
    av_img_time = 0
    av_steps = 0
    total_t = 0

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) /  650000

    reward_log = np.zeros((episodes, 7))
    for i in range(episodes):
        old_steps = total_t
        episode_reward, training_time, image_time, epsilon, total_t = agent.play_one(target_network=target_network,
                                                                                     t=total_t,
                                                                                     epsilon=epsilon,
                                                                                     epsilon_change=epsilon_change,
                                                                                     epsilon_min=epsilon_min,
                                                                                     update_freq=update_freq,
                                                                                     min_size=min_size,
                                                                                     train_freq=train_freq)
        av_reward += episode_reward
        av_train_time += training_time
        av_img_time += image_time
        av_steps += (total_t - old_steps)

        # display message
        if i % pritn_freq == 0:
            reward = av_reward / pritn_freq
            steps = av_steps / pritn_freq

            print(
                "Episode {} Averages({}) - reward : {} | e: {} | steps : {} | total_steps: {} ".format(
                    i + 1,
                    pritn_freq,
                    reward,
                    round(epsilon, 2),
                    round(steps, 2),
                    round(total_t, 2)))

            reward_log[i] = [0, 0, image_time, reward, 0, steps, epsilon]

            if i % (pritn_freq * 100) == 0:
                np.save("reward-log-final.npy", reward_log)
                torch.save(agent.V, "agent-final-{}.pt".format(i))

            av_reward = 0
            av_train_time = 0
            av_img_time = 0
            av_steps = 0

def run_connect():
    env = make("connectx", debug=True)
    trainer = env.train([None, "random"])
    K = 7
    lr = 0.00025

    V = ValueModel(lr=lr,n_outputs=K)
    target_net = ValueModel(lr=lr, n_outputs=K)

    experience = ExperienceReplay(max_size=300000,  # 700000,
                                  min_size=50000,
                                  batch_size=32)

    agent = ConnectAgent(trainer, V, experience)

    train(agent,
          target_net,
          min_size=5000,
          episodes=500000,
          pritn_freq=1000,
          update_freq=1000,
          train_freq=4)  # 1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    run_connect()