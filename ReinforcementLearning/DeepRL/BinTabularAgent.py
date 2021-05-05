import gym
import random
import numpy as np
from matplotlib import pyplot as plt


class BinTabularAgent():
    def __init__(self, env_name, state_bins, actions, lr=0.01, gamma=0.9, e=1):
        self.actions = actions
        self.state_dims = (state_bins.shape[1] + 1) ** state_bins.shape[0]
        self.stat_bins = state_bins
        self.lr = lr
        self.gamma = gamma
        self.e = e
        self.env = gym.make(env_name)

        # Build Q table
        self.Q = np.random.uniform(-1, 1, (self.state_dims, len(actions)))

    def find_q_index(self, state):
        index_list = [np.digitize(x=s, bins=self.stat_bins[i]) for (i, s) in enumerate(state)]
        string_index = ""
        for index in index_list:
            string_index += str(index)
        return int(string_index)

    def chose_action(self, s, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.find_max_action(s)

    def find_max_action(self, s):
        x = self.find_q_index(s)
        try:
            return np.argmax(self.Q[x])
        except Exception as e:
            t = 9

    def update_q(self, s, s_, a, r):
        x = self.find_q_index(s)
        x_ = self.find_q_index(s_)

        try:
            self.Q[x, a] += self.lr * (r + (self.gamma * np.max(self.Q[x_]) - self.Q[x, a]))
        except Exception as e:
            t = 7

    def train(self, episodes=1000, render=False, print_freq=100):
        episode_rewards = []
        cumulative_rewards = 0
        longest = 0
        for i_episode in range(episodes):
            epsilon = self.e / np.sqrt(i_episode + 1)
            s = self.env.reset()
            episode_reward = 0

            for t in range(500):
                if render:
                    self.env.render()

                a = self.chose_action(s, epsilon)

                s_, r, done, info = self.env.step(a)

                if done and t < 499:
                    r = -500

                episode_reward += r
                if episode_reward > longest:
                    longest = episode_reward

                self.update_q(s, s_, a, r)

                s = s_

                if done:
                    cumulative_rewards += episode_reward
                    episode_rewards.append(cumulative_rewards)

                    break

            if i_episode % print_freq == 0:
                self.print_progress(i_episode, cumulative_rewards / print_freq, epsilon, longest)
                cumulative_rewards = 0
                longest = 0
        plt.plot(episode_rewards)
        plt.show()

    def print_progress(self, i, average_reward, epsilon, longest):
        print("Episode {}, av reward: {} | lr: {} | e: {} | longest: {}".format(i, average_reward, self.lr, epsilon,
                                                                                longest))




if __name__ == '__main__':
    env_name = "CartPole-v1"
    dim = 9
    state_bins = np.array([np.linspace(-2.4, 2.4, dim), np.linspace(-2, 2, dim), np.linspace(-0.4, 0.4, dim),
                           np.linspace(-3.5, 3.5, dim)])

    actions = [0, 1]
    agent = BinTabularAgent(env_name, state_bins, actions, lr=0.01)
    agent.train(episodes=10000, print_freq=1000, render=False)
