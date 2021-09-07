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
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class GameImageProcessor:
    def __init__(self, crop, image_dimensions):
        self.crop = crop
        self.image_dimensions = image_dimensions

    def process_image(self, state):
        # state = 210 x 160 x 3
        # 164 x 160 x 3
        state = state[self.crop[0]:self.crop[1]]
        # 164 x 160
        state = np.mean(state, axis=2)
        # 80 x 80
        state = Image.fromarray(state).resize(size=self.image_dimensions, resample=Image.NEAREST)
        state = np.array(state)

        state = state / 255.0
        # Image.fromarray(state).show()

        return state

    def repeat_frame(self, frame):
        return np.stack([frame] * 4, axis=0)

    def frame_to_state(self, state, next_frame):
        return np.append(state[1:, :, :], np.expand_dims(next_frame, 0), axis=0)


class Worker(mp.Process):
    def __init__(self, id, env_name, network, n_steps, gamma, color, grad_queue, rewards_list, image_processor):
        super(Worker, self).__init__()
        colorama.init()

        self.id = id
        self.color = color
        self.grad_queue = grad_queue
        self.rewards_list = rewards_list
        self.network = network
        self.image_processor = image_processor
        self.env = gym.make(env_name)

        self.gamma = gamma
        self.n_steps = n_steps

        self.episode_count = 0

    def calc_grads(self, steps):
        returns = []
        states = []
        actions = []

        G = 0.0
        if not steps[-1].done:
            _, G = self.network.forward([steps[-1].next_state])
            G = G.detach().numpy()

        for step in reversed(steps):
            G = step.reward + self.gamma * G

            returns.append(G)
            states.append(step.state)
            actions.append(step.action)

        returns.reverse()
        states.reverse()
        actions.reverse()

        self.network.calc_loss(states, actions, returns)
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.1)
        grads = [param.grad.data.cpu().numpy() if param.grad is not None else 0 for param in self.network.parameters()]
        self.grad_queue.put(grads)

    def run(self, print_freq=20):
        episode_reward = 0
        rewards = []

        current_state = self.image_processor.repeat_frame(self.image_processor.process_image(self.env.reset()))
        steps = []
        while True:
            for t in range(self.n_steps):
                # play game
                action = self.network.chose_action([current_state])

                s_, r, done, info = self.env.step(action)

                episode_reward += r

                next_state = self.image_processor.frame_to_state(current_state, self.image_processor.process_image(s_))

                steps.append(Step(current_state, action, r, next_state, done))

                # if game over break
                if done:
                    current_state = self.image_processor.repeat_frame(self.image_processor.process_image(self.env.reset()))
                    self.episode_count += 1

                    rewards.append(episode_reward)
                    self.rewards_list.put(episode_reward)
                    episode_reward = 0

                    # print av reward
                    if self.episode_count != 0 and self.episode_count % print_freq == 0:
                        av_reward = sum(rewards) / len(rewards)
                        print(termcolor.colored("\n{} Past {} av reward: {} ".format(self.id, print_freq, round(av_reward, 2)), color=self.color))
                        rewards = []
                    break

                else:
                    current_state = next_state

            # calculate gradients and put in queue
            self.calc_grads(steps)

            steps = []
