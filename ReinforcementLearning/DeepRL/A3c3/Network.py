import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

device = torch.device('cpu')


class ActorCriticModel(nn.Module):
    def __init__(self, image_dim, color_dim, n_classes, C, lr):
        super(ActorCriticModel, self).__init__()
        self.C = C

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
        for (grad, param) in zip (grads, self.parameters()):
            param.grad = torch.FloatTensor(grad)
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        self.optim.step()
