import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class ActorCriticModel(nn.Module):
    def __init__(self, image_dim, color_dim, n_classes, C, lr, device, weight_storage):
        super(ActorCriticModel, self).__init__()
        self.device = device
        self.weight_storage = weight_storage
        self.C = C

        self.shared = nn.Sequential(
            nn.Conv2d(color_dim, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        ).to(device=self.device)

        cnn_output = self.shared(torch.zeros(1, color_dim, *image_dim).to(device=self.device)).flatten().shape[0]

        self.shared.add_module("7", nn.Linear(cnn_output, 512, bias=True))
        self.shared.add_module("8", nn.ReLU())

        self.V = nn.Sequential(nn.Linear(512, 1)).to(device=self.device)

        self.P = nn.Sequential(nn.Linear(512, n_classes),
                               nn.Softmax(dim=1)).to(device=self.device)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, X):
        state_t = torch.tensor(X, dtype=torch.float32, device=self.device)
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

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)

        self.optim.zero_grad()

        pi, values = self.forward(states)

        critic_loss = F.mse_loss(values, returns.squeeze())

        advantages = returns - values.detach()
        H = -torch.sum(pi * torch.log(pi), dim=1)

        actor_loss = advantages * torch.log(pi[torch.arange(len(actions)), actions]) + (self.C * H)

        total_loss = critic_loss - actor_loss.mean()

        total_loss.backward()

        self.optim.step()

        return total_loss


    def update_weights(self):
        for (param, storage) in zip(self.parameters(), self.weight_storage.parameters()):
            param.data = storage.data.clone()

    def share_weights(self):
        for (param, storage) in zip(self.parameters(), self.weight_storage.parameters()):
            storage.data = param.data.detach().cpu()



