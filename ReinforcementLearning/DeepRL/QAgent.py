import gym
import random
import numpy as np


class QAgent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
    def chose_action(self, s):
        pass

    def update_epsilon(self, i):
        pass

    def update_lr(self, i):
        pass

    def update_q(self, s, s_, a, r):
        pass

    def print_progress(self, i, average_reward):
        pass

    def train(self, episodes=1000, render=False, print_freq=100):

        cumulative_rewards = 0

        for i_episode in range(episodes):

            self.update_epsilon(i_episode)
            self.update_lr(i_episode)

            s = self.env.reset()
            episode_reward = 0

            for t in range(200):

                if render:
                    self.env.render()

                s_t = [s[0], s[1]]
                if s[2]:
                    s_t.append(1)
                else:
                    s_t.append(0)
                s = s_t

                a = self.chose_action(s)

                s_, r, done, info = self.env.step(a)

                s_t = [s_[0], s_[1]]
                if s_[2]:
                    s_t.append(1)
                else:
                    s_t.append(0)
                s_ = s_t


                episode_reward += r

                if r == -1:
                    r  =-10

                self.update_q(s, s_, a, r)


                if done:
                    cumulative_rewards += episode_reward
                    # if i_episode % print_freq == 0:
                    #     print(s)
                    #     print(a)
                    #     print(s_)
                    #     print("\n")
                    break
                s = s_



            if i_episode % print_freq == 0:
                self.print_progress(i_episode, cumulative_rewards / print_freq)
                cumulative_rewards = 0






