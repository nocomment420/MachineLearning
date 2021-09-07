import numpy as np
from PIL import Image
import datetime
import termcolor
import colorama
import torch.multiprocessing as mp
import torch
import gym

class Step:
    def __init__(self, state, action, reward, next_state, done):
        colorama.init()
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class StateActionReward:
    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward


class Worker(mp.Process):
    def __init__(self, id, env_name, network, n_steps, gamma, color, step_queue):
        super(Worker, self).__init__()

        self.id = id
        self.color = color
        self.step_queue = step_queue
        self.network = network

        self.env = gym.make(env_name)

        self.gamma = gamma
        self.n_steps = n_steps

        self.episode_count = 0

    def store_progress(self, steps):
        state_action_rewards = []

        G = 0.0
        if not steps[-1].done:
            _, G = self.network.forward([steps[-1].next_state])
            G = G.detach().numpy()

        for step in reversed(steps):
            G = step.reward + self.gamma * G
            sar = StateActionReward(step.state, step.action, G)
            state_action_rewards.append(sar)

        state_action_rewards.reverse()

        self.step_queue.put(state_action_rewards)

        self.network.update_weights()



    def process_image(self, state):
        # state = 210 x 160 x 3

        # 164 x 160 x 3
        state = state[31:195]

        # 164 x 160
        state = np.mean(state, axis=2)

        # 80 x 80
        state = Image.fromarray(state).resize(size=(84, 84), resample=Image.NEAREST)

        state = np.array(state)
        state = state / 255.0

        return state

    def repeat_frame(self, frame):
        return np.stack([frame] * 4, axis=0)

    def frame_to_state(self, state, next_frame):
        return np.append(state[1:, :, :], np.expand_dims(next_frame, 0), axis=0)

    def run(self):
        episode_reward = 0
        episode_steps = 0
        episode_start = datetime.datetime.now()
        episode_train_time = 0
        rewards = []

        current_state = self.repeat_frame(self.process_image(self.env.reset()))
        steps = []
        while True:
            for t in range(self.n_steps):
                # play game
                action = self.network.chose_action([current_state])

                s_, r, done, info = self.env.step(action)

                episode_reward += r
                episode_steps += 1

                next_state = self.frame_to_state(current_state, self.process_image(s_))

                steps.append(Step(current_state, action, r, next_state, done))

                # if game over break
                if done:
                    self.episode_count += 1
                    total_time = (datetime.datetime.now() - episode_start).seconds

                    # print(termcolor.colored(
                    #     "{} (E{}) - Reward: {} | Steps: {} | Total Time: {} | Train Time: {}".format(self.id,
                    #                                                                                  self.episode_count,
                    #                                                                                  episode_reward,
                    #                                                                                  episode_steps,
                    #                                                                                  total_time,
                    #                                                                                  round(
                    #                                                                                      episode_train_time * 1e-6,
                    #                                                                                      2)),
                    #     color=self.color))

                    if self.episode_count != 0 and self.episode_count % 20 == 0:
                        av_reward = sum(rewards[self.episode_count - 20:self.episode_count]) / 20
                        print(termcolor.colored(
                            "\n-----------------{} Past 20 Av Reward: {} -------------------\n".format(self.id,
                                                                                                       round(av_reward,
                                                                                                             2)),
                            color=self.color))

                    # save reward
                    rewards.append(episode_reward)
                    if len(rewards) % 50 == 0:
                        np.save("{}-rewards.npy".format(self.id), np.array(rewards))

                    episode_reward = 0
                    episode_steps = 0
                    episode_start = datetime.datetime.now()
                    episode_train_time = 0

                    current_state = self.repeat_frame(self.process_image(self.env.reset()))

                    break
                else:
                    current_state = next_state

            # train global network
            t_0 = datetime.datetime.now()
            self.store_progress(steps)
            episode_train_time += (datetime.datetime.now() - t_0).microseconds

            steps = []