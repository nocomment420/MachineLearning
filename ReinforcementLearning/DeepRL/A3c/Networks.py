import gym
import numpy as np
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class PolicyValueModel(nn.Module):
    def __init__(self, image_dim, color_dim, n_classes, C, lr, global_optimiser):
        super(PolicyValueModel, self).__init__()
        self.C = C

        self.shared = nn.Sequential(
            nn.Conv2d(color_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(device=device)
        cnn_output = self.shared(torch.zeros(1, color_dim, *image_dim).to(device=device)).flatten().shape[0]
        self.shared.add_module("7", nn.Linear(cnn_output, 512, bias=True))
        self.shared.add_module("8", nn.ReLU())

        self.V = nn.Sequential(nn.Linear(512, 1)).to(device=device)

        self.P = nn.Sequential(nn.Linear(512, n_classes),
                               nn.Softmax(dim=1)).to(device=device)

        self.local_optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, alpha=0.99, eps=1e-6, weight_decay=0,
                                                   momentum=0)
        self.optimizer = global_optimiser
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def shared_forwards(self, X):
        state_t = torch.tensor(X, dtype=torch.float32).to(device=device)
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

    def calc_loss(self, states, actions, returns):
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, device=device)  # T x 1

        cnn_out = self.shared_forwards(states)

        V_prediction = self.V(cnn_out)
        V_prediction = torch.reshape(V_prediction, [len(V_prediction)])

        advantages = returns - V_prediction
        critic_loss = advantages.pow(2)

        P_prediction = self.P(cnn_out)  # T x K

        H = -torch.sum(P_prediction * torch.log(P_prediction), dim=1)

        P_prediction = P_prediction[torch.arange(len(actions)), actions]  # T x 1

        actor_loss = (advantages * torch.log(P_prediction)) + (self.C * H)

        return torch.sum(critic_loss) - torch.sum(actor_loss)

    def fit_gradients(self, advantages, value_targets, states, actions):
        self.local_optimizer.zero_grad()

        target = torch.tensor(value_targets, dtype=torch.float32).to(device=device)
        advantage = torch.tensor(advantages).to(device=device)  # T x 1
        actions = torch.tensor(actions).to(device=device)  # T x 1

        V_prediction = self.V_forwards(states)
        V_prediction = torch.reshape(V_prediction, [len(V_prediction)])

        critic_loss = (target - V_prediction).pow(2)

        P_prediction = self.P_forwards(states)  # T x K
        H = -torch.sum(P_prediction * torch.log(P_prediction), dim=1)

        P_prediction = P_prediction[torch.arange(len(actions)), actions]  # T x 1

        actor_loss = (advantage * torch.log(P_prediction)) + (self.C * H)

        total_loss = torch.sum(critic_loss) - torch.sum(actor_loss)

        total_loss.backward()

    def copy(self, copy):
        for (target, source) in zip(self.parameters(), copy.parameters()):
            target.data = source.data.clone()

    def apply_gradients(self, total_loss, target_network):
        self.optimizer.zero_grad()
        total_loss.backward()

        for (old, new) in zip(self.parameters(), target_network.parameters()):
            old.grad_ = new.grad

        torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

        self.optimizer.step()


class ActorCriticModel(nn.Module):
    def __init__(self, image_dim, color_dim, n_classes, C):
        super(ActorCriticModel, self).__init__()
        self.C = C
        #
        # self.shared = nn.Sequential(
        #     nn.Conv2d(color_dim, 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # ).to(device=device)
        self.shared = nn.Sequential(
            nn.Conv2d(color_dim, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        ).to(device=device)

        cnn_output = self.shared(torch.zeros(1, color_dim, *image_dim).to(device=device)).flatten().shape[0]

        self.shared.add_module("7", nn.Linear(cnn_output, 512, bias=True))
        self.shared.add_module("8", nn.ReLU())


        self.shared2 = nn.Sequential(
            nn.Conv2d(color_dim, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        ).to(device=device)
        self.shared2.add_module("7", nn.Linear(cnn_output, 512, bias=True))
        self.shared2.add_module("8", nn.ReLU())


        self.V = nn.Sequential(nn.Linear(512, 1)).to(device=device)

        self.P = nn.Sequential(nn.Linear(512, n_classes),
                               nn.Softmax(dim=1)).to(device=device)


    def forward(self, X):
        state_t = torch.tensor(X, dtype=torch.float32, device=device)
        shared_out = self.shared(state_t).reshape(-1, 512)

        shared_out2 = self.shared2(state_t).reshape(-1, 512)

        v = self.V(shared_out)
        pi = self.P(shared_out2)
        return pi, v.squeeze()

    def chose_action(self, X):

        pi, v = self.forward(X)

        dist = Categorical(pi)
        action = dist.sample()


        return action.numpy()[0]

    def calc_loss(self, states, actions, returns):

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, device=device)

        pi, values = self.forward(states)
        dist = Categorical(pi)
        log_probs = dist.log_prob(actions)

        advantages = returns - values

        # critic_loss = 0.5 * advantages.pow(2)
        critic_loss = advantages.pow(2)
        actor_loss = (advantages * log_probs) + (self.C * dist.entropy())
        total_loss = torch.sum(critic_loss) - torch.sum(actor_loss)

        return total_loss
