import gym
import numpy as np
import random
import torch
import torch.nn as nn
from PIL import Image
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ActorCritc(nn.Module):
    def __init__(self, image_dim, color_dim, n_classes, C, lr):
        super(ActorCritc, self).__init__()
        self.C = C

        self.shared = nn.Sequential(
            nn.Conv2d(color_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_output = self.shared(torch.zeros(1, color_dim, *image_dim)).flatten().shape[0]
        self.shared.add_module("7", nn.Linear(cnn_output, 512, bias=True))
        self.shared.add_module("8", nn.ReLU())

        self.V = nn.Sequential(nn.Linear(512, 1))

        self.P = nn.Sequential(nn.Linear(512, n_classes),
                               nn.Softmax(dim=1))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def shared_forwards(self, X):
        state_t = torch.tensor(X, dtype=torch.float32, device=device)
        shared_out = self.shared(state_t).reshape(-1, 512)
        return shared_out

    def V_forwards(self, X):
        shared_out = self.shared_forwards(X)
        V_out = self.V(shared_out)
        return V_out

    def P_forwards(self, X):
        shared_out = self.shared_forwards(X)
        P_out = self.P(shared_out)
        return P_out

    def V_predict(self, X):
        return self.V_forwards([X]).cpu().detach().numpy()[0, 0]

    def P_predict(self, X):
        return self.P_forwards(X).cpu().detach().numpy()

    def chose_action(self, s):
        p_actions = self.P_predict(s)[0]
        return np.random.choice(p_actions.shape[0], p=p_actions)

    def fit_gradients(self, advantages, states, actions):

        advantage = torch.tensor(advantages, device=device)  # T x 1
        actions = torch.tensor(actions, device=device)  # T x 1

        P_prediction = self.P_forwards(states)  # T x K
        P_prediction = P_prediction[torch.arange(len(actions)), actions]  # T x 1

        H = -torch.sum(P_prediction * torch.log(P_prediction))  # CHECK THIS

        actor_loss = (advantage * torch.log(P_prediction)) + (self.C * H)

        critic_loss = advantage.pow(2)

        total_loss = (- actor_loss + critic_loss).mean()

        self.optimizer.zero_grad()

        total_loss.backward()

        self.optimizer.step()


class Step:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Worker:
    def __init__(self, env, network, gamma):
        super(Worker, self).__init__()
        self.network = network
        self.env = env
        self.gamma = gamma

    def train(self, step_batch):
        advantages = []
        actions = []
        states = []

        for steps in step_batch:
            G = 0

            for step in reversed(steps):
                # G = step.reward + self.gamma * G
                G = step.reward + self.gamma * self.network.V_predict(step.next_state)
                advantages.append(G - self.network.V_predict(step.state))
                actions.append(step.action)
                states.append(step.state)

        advantages.reverse()
        actions.reverse()
        states.reverse()

        self.network.fit_gradients(advantages, states, actions)

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

    def run(self, episodes, train_freq):
        episode_rewards = np.zeros(episodes)

        for e in range(episodes):
            step_batch = []
            episode_reward = 0
            episode_steps = 0
            episode_start = datetime.datetime.now()
            episode_train_time = 0

            for t in range(train_freq):
                steps = []


                current_state = self.repeat_frame(self.process_image(self.env.reset()))
                done = False
                while not done:
                    # play game
                    a = self.network.chose_action([current_state])

                    s_, r, done, info = self.env.step(a)

                    episode_reward += r
                    episode_steps += 1

                    next_state = self.frame_to_state(current_state, self.process_image(s_))

                    steps.append(Step(current_state, a, r, next_state, done))

                    current_state = next_state

                step_batch.append(steps)

                episode_rewards[e] = episode_reward

            total_time = (datetime.datetime.now() - episode_start).seconds / train_freq
            print("(B{}) - Reward: {} | Steps: {} | Total Time: {} | Train Time: {}".format(e + 1, round(episode_reward / train_freq,2),
                                                                                            round(episode_steps / train_freq,2),
                                                                                            round(total_time / train_freq,2),
                                                                                            round(episode_train_time * 1e-6 / train_freq,2)))
            if e != 0 and e % 100 == 0:
                print("\nAverage reward over 100 episodes: {}\n".format(episode_rewards[e-100:e].mean()))



            t_0 = datetime.datetime.now()
            self.train(step_batch)
            episode_train_time = (datetime.datetime.now() - t_0).microseconds

def train_a2c():
    image_dim = [84, 84]
    color_dim = 4
    n_classes = 4
    lr = 2e-4
    C = 0.001
    gamma = 0.99

    env_name = "Breakout-v0"
    env = gym.make(env_name)

    global_network = ActorCritc(image_dim, color_dim, n_classes, C, lr).to(device)
    target_network = ActorCritc(image_dim, color_dim, n_classes, C, lr).to(device)

    worker = Worker(env, global_network, gamma)

    worker.run(episodes=5000, train_freq=6)


if __name__ == "__main__":
    train_a2c()
