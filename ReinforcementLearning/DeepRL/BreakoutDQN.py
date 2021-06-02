import gym
import numpy as np
import random
import torch
from sklearn.utils import shuffle
from PIL import Image
import torch.nn as nn
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ValueModel(nn.Module):
    def __init__(self, image_dim, color_dim, n_outputs, lr=0.01):
        super(ValueModel, self).__init__()
        self.activations = []
        self.layers = []
        self.params = []
        self.lr = lr

        self.color_dim = color_dim
        self.n_outputs = n_outputs
        self.cnn = nn.Sequential(
            nn.Conv2d(color_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        ).to(device=device)

        self.fc_layer_inputs = self.cnn_out_dim(image_dim)

        self.fully_connected = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs)).to(device=device)

        # self.optimizer = torch.optim.Adam(self.parameters(),
        #                                   lr=lr)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, eps=0.001)
        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.SmoothL1Loss()

    def cnn_out_dim(self, input_dim):
        return self.cnn(torch.zeros(1, self.color_dim, *input_dim).to(device=device)).flatten().shape[0]

    def forwards(self, X):
        state_t = torch.tensor(X, dtype=torch.float32).to(device=device)
        cnn_out = self.cnn(state_t).reshape(-1, self.fc_layer_inputs)
        return self.fully_connected(cnn_out)

    def predict(self, X):
        return self.forwards(X).cpu().detach().numpy()

    def partial_fit(self, X, Y, i):
        self.optimizer.zero_grad()
        prediction = self.forwards(X)
        prediction = prediction[torch.arange(len(prediction)), i]
        target = torch.tensor(Y, dtype=torch.float32).to(device=device)
        loss = self.criterion(prediction, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
        self.optimizer.step()

    def copy(self, copy):
        for (i, (me, them)) in enumerate(zip(self.parameters(), copy.parameters())):
            me.data = them.data.clone()


class ExperienceReplay:
    def __init__(self, max_size, min_size, batch_size, frame_dimensions):
        self.max_size = max_size
        self.min_size = min_size
        self.frame_dimensions = frame_dimensions
        self.batch_size = batch_size
        self.n = 0
        self.current = 0
        self.start = datetime.datetime.now()

        self.action_buffer = np.zeros(max_size)
        self.reward_buffer = np.zeros(max_size)
        self.done_buffer = np.zeros(max_size)
        self.frame_buffer = np.zeros([max_size] + frame_dimensions)

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
            print("{} frames in {} s".format(self.n, (datetime.datetime.now() - self.start).seconds))

        indx = np.random.choice(np.arange(start=4, stop=self.n - 1), replace=False, size=self.batch_size)
        for i in range(-4, 1):
            t = self.done_buffer[indx + i]
            dones = t[t == 1]
            if dones.shape[0] != 0:
                # print("Frame overlap!, calling recursively to generate new index")
                return self.get_batch()

        actions = self.action_buffer[indx - 1]
        rewards = self.reward_buffer[indx - 1]
        dones = self.done_buffer[indx - 1]

        cur_idx = np.dstack((indx - 4, indx - 3, indx - 2, indx - 1))[0]
        next_idx = np.dstack((indx - 3, indx - 2, indx - 1, indx))[0]
        states = self.frame_buffer[cur_idx, :, :]
        next_states = self.frame_buffer[next_idx, :, :]

        return actions, rewards, dones, states, next_states

    def get_last_state(self):
        if self.n < 4:
            return None

        idx = np.dstack((self.current - 4, self.current - 3, self.current - 2, self.current - 1))[0]
        return self.frame_buffer[idx]


class BreakoutAgent:
    def __init__(self, env, value, experience, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.V = value
        self.experience = experience

    def chose_action(self, s, e):
        if random.random() < e:
            return self.env.action_space.sample()
        else:
            state = self.experience.get_last_state()
            if state is not None:
                return np.argmax(self.V.predict(state))
            else:
                return self.env.action_space.sample()

    def train(self, target_network):

        batch = self.experience.get_batch()

        if batch is None:
            return

        actions, rewards, dones, states, next_states = batch
        next_q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self.gamma * next_q if not done else r for (r, next_q, done) in zip(rewards, next_q, dones)]
        self.V.partial_fit(states, targets, actions)

    def state_to_frame(self, state):
        # state = 210 x 160 x 3

        # 164 x 160 x 3
        state = state[31:195]

        # 164 x 160
        state = np.mean(state, axis=2)

        # 80 x 80
        state = Image.fromarray(state).resize(size=(84, 84), resample=Image.NEAREST)

        state = np.array(state)
        state = state / state.max()

        return state

    def play_one(self, target_network, t, min_size, update_freq=200, epsilon=0, epsilon_change=0, epsilon_min=0,
                 train_freq=1):
        episode_reward = 0
        training_time = 0
        image_time = 0

        s = self.env.reset()
        while True:
            f = self.state_to_frame(s)

            a = self.chose_action(f, epsilon)

            s_, r, done, info = self.env.step(a)
            episode_reward += r

            t_0 = datetime.datetime.now()
            self.experience.add(f, a, r, done)
            image_time += (datetime.datetime.now() - t_0).microseconds

            t_0 = datetime.datetime.now()
            if t % train_freq == 0:
                self.train(target_network)
            if t > min_size and t % update_freq == 0:
                target_network.copy(self.V)
            training_time += (datetime.datetime.now() - t_0).microseconds

            s = s_

            epsilon = max(epsilon - epsilon_change, epsilon_min)

            if done:
                break
            t += 1

        return episode_reward, training_time * 1e-6, image_time * 1e-6, epsilon, t


def train(agent, target_network, min_size, episodes=1000, update_freq=1000, pritn_freq=1, train_freq=1):
    av_reward = 0
    av_train_time = 0
    av_img_time = 0
    av_play_time = datetime.datetime.now()
    av_steps = 0
    total_t = 0

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 400000  # 650000

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
            total_time = (datetime.datetime.now() - av_play_time).seconds
            train_time = av_train_time / pritn_freq
            image_time = av_img_time / pritn_freq
            reward = av_reward / pritn_freq
            play_time = (total_time / pritn_freq) - train_time - image_time
            steps = av_steps / pritn_freq

            print(
                "Episode {} Averages({}) - reward : {} | e: {} | steps : {} | total_steps: {} | train_time: {} | image_time: {} | play_time: {}".format(
                    i + 1,
                    pritn_freq,
                    reward,
                    round(epsilon, 2),
                    round(steps, 2),
                    round(total_t, 2),
                    round(train_time, 2),
                    round(image_time, 2),
                    round(play_time, 2)))

            reward_log[i] = [total_time, train_time, image_time, reward, play_time, steps, epsilon]

            if i % (pritn_freq * 100) == 0:
                np.save("reward-log-final.npy", reward_log)
                torch.save(agent.V, "agent-final-{}.pt".format(i))

            av_reward = 0
            av_train_time = 0
            av_img_time = 0
            av_play_time = datetime.datetime.now()
            av_steps = 0


def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def plot_smoothed_return():
    reward_log = np.load("reward-log-final.npy")
    episode_rewards = reward_log[:, 3]

    # Plot the smoothed returns
    y = smooth(episode_rewards)
    plt.plot(episode_rewards, label='orig')
    plt.plot(y, label='smoothed')
    plt.legend()
    plt.show()


def run_random(episodes=20):
    env_name = "Breakout-v0"
    env = gym.make(env_name)
    for i in range(episodes):
        done = False
        s = env.reset()

        while not done:
            env.render()

            a = env.action_space.sample()

            s_, r, done, info = env.step(a)

            s = s_


def run_from_file(episodes=20):
    env_name = "Breakout-v0"
    env = gym.make(env_name)
    img_dimensions = [84, 84]

    V = torch.load("agent.pt")
    experience = ExperienceReplay(max_size=300000,  # 700000,
                                  min_size=50000,
                                  batch_size=32,
                                  frame_dimensions=img_dimensions)

    agent = BreakoutAgent(env, V, experience)
    e = 0

    for i in range(episodes):
        done = False

        s = env.reset()
        action = None
        count = 0
        while not done:
            env.render()

            f = agent.state_to_frame(s)

            if count % 100 == 0:
                a = 1
            elif random.random() < e:
                a = agent.env.action_space.sample()
            else:
                a = agent.chose_action(f, 0)


            s_, r, done, info = agent.env.step(a)

            agent.experience.add(f, a, r, done)

            s = s_

            count += 1




def run_breakout():
    print('Breakout DQN')

    env_name = "Breakout-v0"
    env = gym.make(env_name)

    img_dimensions = [84, 84]
    K = env.action_space.n
    lr = 0.00025

    V = ValueModel(image_dim=img_dimensions,
                   color_dim=4,
                   lr=lr,
                   n_outputs=K)
    target_net = ValueModel(image_dim=img_dimensions,
                            color_dim=4,
                            lr=lr,
                            n_outputs=K)

    experience = ExperienceReplay(max_size=300000,  # 700000,
                                  min_size=50000,
                                  batch_size=32,
                                  frame_dimensions=img_dimensions)

    agent = BreakoutAgent(env, V, experience)

    train(agent,
          target_net,
          min_size=50000,
          episodes=16000,
          pritn_freq=10,
          update_freq=10000,
          train_freq=4)  # 1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # run_breakout()

    # plot_smoothed_return()
    #
    run_from_file()
    # run_random()
