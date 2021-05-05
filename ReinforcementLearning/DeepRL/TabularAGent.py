import gym
import random
import numpy as np
from DeepRL.QAgent import QAgent


class TabularAgent(QAgent):
    def __init__(self, env_name, state_dims, actions, k_lr=10000, k_e=10000, lr=1, e=1, gamma=0.9):
        super().__init__(env_name)
        self.state_dims = state_dims
        self.actions = actions
        dims = np.append(state_dims, len(actions))

        # Build Q table
        self.Q = np.random.uniform(-1, 1, dims)

        # lr
        self.lr = lr
        self.k_lr = k_lr

        # epsilon
        self.e = e
        self.k_e = k_e

        # gamma
        self.gamma = gamma

    def chose_action(self, s):
        if random.random() < self.e:
            return self.env.action_space.sample()
        else:
            q = self.Q
            for i in s:
                q = q[i]

            return np.argmax(q)

    def update_epsilon(self, i):
        # self.e = self.k_e / (self.k_e + i)
        self.e = 1 / np.sqrt(1 + i)

    def update_lr(self, i):
        # self.lr = self.k_lr / (self.k_lr + i)
        pass

    def update_q(self, s, s_, a, r):
        q = self.Q
        for i in range(len(self.state_dims)):
            try:
                q = q[s[i]]
            except Exception as e:
                t = 8

        q_ = self.Q
        for i in range(len(self.state_dims)):
            q_ = q_[s[i]]


        q[a] += self.lr * (r + (self.gamma * np.max(q_) - q[a]))

    def print_progress(self, i, average_reward):
        print("Episode {}, av reward: {} | lr: {} | e: {}".format(i, average_reward, self.lr, self.e))

    def show_games(self, n):
        wins = 0
        draws = 0
        for i_episode in range(n):
            print("\n\n\n GAME {}".format(i_episode + 1))
            s = self.env.reset()

            for t in range(200):

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

                print("{} + {} -> {} ({})".format(s, a, s_, r))

                if done:
                    if r == 1:
                        wins += 1
                    elif r == 0:
                        draws += 1
                    break
                s = s_
        print("Win rate : {}/{} | draw rate : {}/{}".format(wins, n, draws, n))


if __name__ == '__main__':
    env_name = "Blackjack-v0"
    state_dims = [32, 11, 2]
    actions = [0, 1]
    agent = TabularAgent(env_name, state_dims, actions, lr=0.01)
    print(agent.env.observation_space)
    agent.train(episodes=100000, print_freq=1000)
    agent.show_games(10)
