import gym
import numpy as np
import random
import torch
from sklearn.utils import shuffle
from PIL import Image
import torch.nn as nn


class ValueModel(nn.Module):
    def __init__(self, image_dim, color_dim, n_outputs, device="cpu", lr=0.01):
        super(ValueModel, self).__init__()

        self.activations = []
        self.layers = []
        self.params = []
        self.lr = lr
        self.device = device
        self.color_dim = color_dim
        self.n_outputs = n_outputs
        self.cnn = nn.Sequential(
            nn.Conv2d(color_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc_layer_inputs = self.cnn_out_dim(image_dim)

        self.fully_connected = nn.Sequential(
            nn.Linear(self.fc_layer_inputs, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs))

        # Set device for GPU's
        if self.device == 'cuda':
            self.cnn.cuda()
            self.fully_connected.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=lr)
        self.criterion = torch.nn.MSELoss()

    def cnn_out_dim(self, input_dim):
        return self.cnn(torch.zeros(1, self.color_dim, *input_dim)).flatten().shape[0]

    def forwards(self, X):
        try:
            state_t = torch.tensor(X, dtype=torch.float32).to(device=self.device)
            cnn_out = self.cnn(state_t).reshape(-1, self.fc_layer_inputs)
            return self.fully_connected(cnn_out)
        except Exception as e:
            g = 0

    def predict(self, X):
        return self.forwards(X).detach().numpy()

    def partial_fit(self, X, Y, i):
        self.optimizer.zero_grad()

        prediction = self.forwards(X)
        prediction = prediction[torch.arange(len(prediction)), i]
        target = torch.tensor(Y, dtype=torch.float32)
        loss = self.criterion(prediction, target)
        loss.backward()

        self.optimizer.step()

    def copy(self, copy):
        for (i, (target, source)) in enumerate(zip(self.params, copy.params)):
            self.params[i] = copy.params[i].clone()


class ExperienceReplay:
    def __init__(self, max_size, min_size, batch_size, frame_dimensions):
        self.max_size = max_size
        self.min_size = min_size
        self.frame_dimensions = frame_dimensions
        self.batch_size = batch_size
        self.n = 0

        self.action_buffer = np.zeros(max_size)
        self.reward_buffer = np.zeros(max_size)
        self.done_buffer = np.zeros(max_size)
        self.frame_buffer = np.zeros([max_size] + frame_dimensions)

    def add(self, frame, action, reward, done):
        self.frame_buffer[1:] = self.frame_buffer[:-1]
        self.frame_buffer[0] = frame

        self.done_buffer[1:] = self.done_buffer[:-1]
        self.done_buffer[0] = done

        self.action_buffer[1:] = self.action_buffer[:-1]
        self.action_buffer[0] = action

        self.reward_buffer[1:] = self.reward_buffer[:-1]
        self.reward_buffer[0] = reward

        self.n += 1

    def get_batch(self):
        if self.n < self.min_size:
            return None

        indx = np.random.choice(self.n, replace=False, size=self.batch_size)
        for i in range(-4, 1):
            t = self.done_buffer[indx + i]
            dones = t[t == 1]
            if dones.shape[0] != 0:
                print("Frame overlap!, calling recursively to generate new index")
                return self.get_batch()

        actions = self.action_buffer[indx]
        rewards = self.reward_buffer[indx]
        dones = self.done_buffer[indx]

        cur_idx = np.dstack((indx - 4, indx - 3, indx - 2, indx - 1))[0]
        next_idx = np.dstack((indx - 3, indx - 2, indx - 1, indx))[0]
        states = self.frame_buffer[cur_idx, :, :]
        next_states = self.frame_buffer[next_idx, :, :]

        return actions, rewards, dones, states, next_states

    def get_last_state(self):
        if self.n < 4:
            return None

        if self.n < self.max_size:
            idx = np.dstack((self.n - 1, self.n - 2, self.n - 3, self.n - 4))[0]
        else:
            idx = np.dstack((self.max_size - 1, self.max_size - 2, self.max_size - 3, self.max_size - 4))[0]

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
        state = Image.fromarray(state).resize(size=(80, 80), resample=Image.NEAREST)

        state = np.array(state)
        state = state / state.max()

        return state

    def play_one(self, e, target_network, max_iter=2000, update_freq=50):
        episode_reward = 0

        s = self.env.reset()

        for t in range(max_iter):
            f = self.state_to_frame(s)

            a = self.chose_action(f, e)

            s_, r, done, info = self.env.step(a)
            episode_reward += r

            self.experience.add(f, a, r, done)

            self.train(target_network)
            if t % update_freq == 0:
                target_network.copy(self.V)

            s = s_

            if done:
                break

        return episode_reward


def train(agent, target_network, episodes=1000, max_iter=1000, pritn_freq=1):
    av_reward = 0

    for i in range(episodes):
        eps = 1.0 / np.sqrt(1 + (i))  # /10))

        episode_reward = agent.play_one(target_network=target_network, e=eps, max_iter=max_iter)
        av_reward += episode_reward

        # display message
        if i % pritn_freq == 0:
            print("Episode {} - Av reward : {} | e:{}".format(i + 1, av_reward / pritn_freq, eps))
            av_reward = 0


def run_breakout():
    print('Breakout DQN')

    env_name = "Breakout-v0"
    env = gym.make(env_name)

    actions = env.action_space.n

    img_dimentions = [80, 80]
    K = env.action_space.n
    lr = 0.01

    V = ValueModel(image_dim=img_dimentions, color_dim=4, lr=lr, n_outputs=K)
    target_net = ValueModel(image_dim=img_dimentions, color_dim=4, lr=lr, n_outputs=K)
    experience = ExperienceReplay(max_size=10000, min_size=100, batch_size=32, frame_dimensions=img_dimentions)
    agent = BreakoutAgent(env, V, experience)

    train(agent, target_net, episodes=100000, pritn_freq=1, max_iter=100)


if __name__ == "__main__":
    run_breakout()
