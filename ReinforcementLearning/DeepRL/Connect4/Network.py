from kaggle_environments import evaluate, make, utils


# g = env.render(mode="html")
# with open("test.html","w") as f:
#     f.write(g)
#     f.close()
import termcolor
import colorama
import torch.multiprocessing as mp
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
device = torch.device('cpu')


class Step:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Worker(mp.Process):
    def __init__(self, id, network, n_steps, gamma, color, grad_queue):
        super(Worker, self).__init__()
        colorama.init()

        self.id = id
        self.color = color
        self.grad_queue = grad_queue
        self.network = network
        env = make("connectx", debug=True)
        self.env = env.train([None, "random"])

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
        grads = [param.grad.data.cpu().numpy() if param.grad is not None else 0 for param in
                 self.network.parameters()]
        self.grad_queue.put(grads)

    def run(self, print_freq=20):
        episode_reward = 0
        rewards = []

        current_state = self.env.reset()
        steps = []
        while True:
            for t in range(self.n_steps):

                action = self.network.chose_action([current_state])

                s_, r, done, info = self.env.step(action)

                episode_reward += r

                next_state = s_

                steps.append(Step(current_state, action, r, next_state, done))

                # if game over break
                if done:
                    current_state = self.env.reset()
                    self.episode_count += 1

                    rewards.append(episode_reward)
                    episode_reward = 0

                    # print av reward
                    if self.episode_count != 0 and self.episode_count % print_freq == 0:
                        av_reward = sum(rewards) / len(rewards)
                        print(termcolor.colored(
                            "\n{} Past {} av reward: {} ".format(self.id, print_freq, round(av_reward, 2)),
                            color=self.color))
                        rewards = []
                    break

                else:
                    current_state = next_state

            # calculate gradients and put in queue
            self.calc_grads(steps)

            steps = []


class ActorCriticModel(nn.Module):
    def __init__(self, n_classes, C, lr):
        super(ActorCriticModel, self).__init__()
        self.C = C

        self.shared = nn.Sequential(
            nn.Linear(42, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 250, bias=True),
            nn.ReLU()
        ).to(device=device)

        self.V = nn.Sequential(nn.Linear(512, 1)).to(device=device)

        self.P = nn.Sequential(nn.Linear(512, n_classes),
                               nn.Softmax(dim=1)).to(device=device)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, X):
        state_t = torch.tensor(X, dtype=torch.float32, device=device)
        shared_out = self.shared(state_t).reshape(-1, 512)

        v = self.V(shared_out)
        pi = self.P(shared_out)
        return pi, v.squeeze()

    def chose_action(self, X):
        pi, v = self.forward(X)

        dist = Categorical(pi)
        action = dist.sample()

        return action.numpy()[0]

    def calc_loss(self, states, actions, returns):
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, device=device)

        self.optim.zero_grad()

        pi, values = self.forward(states)

        critic_loss = F.mse_loss(values, returns.squeeze())

        advantages = returns - values.detach()
        H = -torch.sum(pi * torch.log(pi), dim=1)

        actor_loss = advantages * torch.log(pi[torch.arange(len(actions)), actions]) + (self.C * H)

        total_loss = critic_loss - actor_loss.mean()

        total_loss.backward()

    def apply_grads(self, grads):
        for (grad, param) in zip(grads, self.parameters()):
            param.grad = torch.FloatTensor(grad)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        self.optim.step()
