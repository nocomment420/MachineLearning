import gym
import numpy as np
import random
import torch
from sklearn.utils import shuffle
from DeepQNetwork import ValueModel
from PIL import Image



class ExperienceReplay:
    def __init__(self, max_size, min_size, batch_size, frame_dimensions):
        self.max_size = max_size
        self.min_size = min_size
        self.frame_dimensions = self.frame_dimensions
        self.batch_size = batch_size
        self.n = 0

        self.action_buffer = np.zeros(max_size)
        self.reward_buffer = np.zeros(max_size)
        self.done_buffer = np.zeros(max_size)
        self.frame_buffer = np.zeros([max_size] + frame_dimensions)

    def add(self, frame, action, reward, done):
        if self.n >= self.max_size:
            self.done_buffer = np.roll(self.done_buffer, 1, axis=0)
            self.reward_buffer = np.roll(self.reward_buffer, 1, axis=0)
            self.action_buffer = np.roll(self.action_buffer, 1, axis=0)
            self.frame_buffer = np.roll(self.frame_buffer, 1, axis=0)

        self.done_buffer[0] = done
        self.action_buffer[0] = action
        self.reward_buffer[0] = reward
        self.frame_buffer[0] = frame

        self.n += 1

    def get_batch(self):
        if self.n < self.min_size:
            return None

        indx = np.random.choice(self.n, replace=False, size=self.batch_size)

        for i in range(indx - 4, indx + 1):
            if self.done_buffer[i]:
                print("Frame overlap!, calling recursively to generate new index")
                return self.get_batch()

        actions = self.action_buffer[indx]
        rewards = self.reward_buffer[indx]
        dones = self.done_buffer[indx]
        states = self.frame_buffer[indx - 4:indx, :, :]
        next_states = self.frame_buffer[indx - 3:indx + 1, :, :]

        return actions, rewards, dones, states, next_states


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
            return np.argmax(self.V.predict([s]))

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
        state = state[31, 195]

        # 164 x 160 x 3
        state = np.mean(state, axis=2)

        # 80 x 80
        state = Image.fromarray(state).resize(size=(80, 80), resample=Image.NEAREST)

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
    H = 200
    K = len(actions)
    lr = 0.01
    X = [4, 80, 80]

    v_layers = [[X, H, torch.tanh, True],
                [H, H, torch.tanh, True],
                [H, K, None, True]]
    V = ValueModel(v_layers, lr=lr)
    target_net = ValueModel(v_layers, lr=lr)
    experience = ExperienceReplay(max_size=10000, min_size=100, batch_size=32, frame_dimensions=img_dimentions)
    agent = BreakoutAgent(env, V, experience)

    train(agent, target_net, episodes=100000, pritn_freq=4, max_iter=2000)


if __name__ == "__main__":
    run_breakout()
