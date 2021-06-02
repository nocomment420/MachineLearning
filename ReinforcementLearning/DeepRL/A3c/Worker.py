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

class WorkerOld(mp.Process):
    def __init__(self, id, env, global_network, local_network, optimiser, global_counter, max_total_steps, n_steps,
                 gamma, color,
                 is_ghost=False):
        super(Worker, self).__init__()
        self.id = id
        self.optimiser = optimiser
        self.color = color
        self.is_ghost = is_ghost

        self.global_network = global_network
        self.global_counter = global_counter
        self.max_total_steps = max_total_steps

        self.env = env
        self.gamma = gamma
        self.n_steps = n_steps

        self.local_counter = 0
        self.local_network = local_network
        self.episode_count = 0

    def train_global(self, steps):
        returns = []
        actions = []
        states = []

        G = 0.0
        if not steps[-1].done:
            G = self.local_network.V_predict(steps[-1].next_state)

        for step in reversed(steps):
            G = step.reward + self.gamma * G

            returns.append(G)
            actions.append(step.action)
            states.append(step.state)

        returns.reverse()
        actions.reverse()
        states.reverse()

        total_loss = self.local_network.calc_loss(states, actions, returns)

        self.optimiser.zero_grad()

        for param in self.local_network.parameters():
            if param.grad is not None:
                param.grad.data.zero_()

        total_loss.backward()

        for (old, new) in zip(self.global_network.parameters(), self.local_network.parameters()):
            old.grad_ = new.grad

        torch.nn.utils.clip_grad_norm_(self.global_network.parameters(), 5)

        self.optimiser.step()

        self.local_network.load_state_dict(self.global_network.state_dict())

    def train_global_old(self, steps):
        value_targets = []
        advantages = []
        actions = []
        states = []

        G = 0.0
        if not steps[-1].done:
            G = self.local_network.V_predict(steps[-1].next_state)

        for step in reversed(steps):
            value_targets.append(G)
            advantages.append(G - self.local_network.V_predict(step.state))
            G = step.reward + self.gamma * G
            actions.append(step.action)
            states.append(step.state)

        value_targets.reverse()
        advantages.reverse()
        actions.reverse()
        states.reverse()

        self.local_network.fit_gradients(advantages, value_targets, states, actions)

        self.global_network.apply_gradients(self.local_network)

        # copy from global
        self.local_network.copy(self.global_network)

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
                a = self.local_network.chose_action([current_state])

                s_, r, done, info = self.env.step(a)

                episode_reward += r
                episode_steps += 1

                next_state = self.frame_to_state(current_state, self.process_image(s_))

                steps.append(Step(current_state, a, r, next_state, done))

                # increase counter + check
                with self.global_counter.get_lock():
                    self.global_counter.value += 1
                    if self.global_counter.value >= self.max_total_steps:
                        print(termcolor.colored("{} DONE".format(self.id), color=self.color))
                        return

                # if game over break
                if done:
                    self.episode_count += 1
                    total_time = (datetime.datetime.now() - episode_start).seconds

                    print(termcolor.colored(
                        "{} (E{}) - Reward: {} | Steps: {} | Total Time: {} | Train Time: {}".format(self.id,
                                                                                                     self.episode_count,
                                                                                                     episode_reward,
                                                                                                     episode_steps,
                                                                                                     total_time,
                                                                                                     round(
                                                                                                         episode_train_time * 1e-6,
                                                                                                         2)),
                        color=self.color))

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
            self.train_global(steps)
            episode_train_time += (datetime.datetime.now() - t_0).microseconds

            steps = []


class Worker(mp.Process):
    def __init__(self, id, env_name, global_network, local_network, optimiser, global_counter, max_total_steps, n_steps,
                 gamma, color):
        super(Worker, self).__init__()
        self.id = id
        self.optimiser = optimiser
        self.color = color

        self.global_network = global_network
        self.global_counter = global_counter
        self.max_total_steps = max_total_steps

        self.env = gym.make(env_name)

        self.gamma = gamma
        self.n_steps = n_steps

        self.local_counter = 0
        self.local_network = local_network
        self.episode_count = 0

    def train_global(self, steps):
        returns = []
        states = []
        actions = []

        G = 0.0
        if not steps[-1].done:
            _, G = self.local_network.forward([steps[-1].next_state])

        for step in reversed(steps):
            G = step.reward + self.gamma * G

            returns.append(G)
            states.append(step.state)
            actions.append(step.action)

        returns.reverse()
        states.reverse()
        actions.reverse()
        self.optimiser.zero_grad()

        total_loss = self.local_network.calc_loss(states, actions, returns)
        total_loss.backward()

        for (old, new) in zip(self.global_network.parameters(), self.local_network.parameters()):
            old._grad = new.grad
        torch.nn.utils.clip_grad_norm_(self.global_network.parameters(), 40)
        self.optimiser.step()

        self.local_network.load_state_dict(self.global_network.state_dict())

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
        count = 0
        while True:
            for t in range(self.n_steps):
                # play game
                count += 1
                if count + 1 % 100 == 0:
                    count = 0
                    action = 1
                else:
                    action = self.local_network.chose_action([current_state])

                s_, r, done, info = self.env.step(action)

                episode_reward += r
                episode_steps += 1

                next_state = self.frame_to_state(current_state, self.process_image(s_))

                steps.append(Step(current_state, action, r, next_state, done))

                # increase counter + check
                with self.global_counter.get_lock():
                    self.global_counter.value += 1
                    if self.global_counter.value >= self.max_total_steps:
                        print(termcolor.colored("{} DONE".format(self.id), color=self.color))
                        return

                # if game over break
                if done:
                    self.episode_count += 1
                    total_time = (datetime.datetime.now() - episode_start).seconds

                    print(termcolor.colored(
                        "{} (E{}) - Reward: {} | Steps: {} | Total Time: {} | Train Time: {}".format(self.id,
                                                                                                     self.episode_count,
                                                                                                     episode_reward,
                                                                                                     episode_steps,
                                                                                                     total_time,
                                                                                                     round(
                                                                                                         episode_train_time * 1e-6,
                                                                                                         2)),
                        color=self.color))

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
            self.train_global(steps)
            episode_train_time += (datetime.datetime.now() - t_0).microseconds

            steps = []
