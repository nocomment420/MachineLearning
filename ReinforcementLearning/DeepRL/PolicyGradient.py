import torch
import numpy as np
import gym


class PolicyModel:
    def __init__(self, layers, lr=0.01):

        self.activations = []
        self.layers = []
        self.params = []
        self.lr = lr

        for (dim_in, dim_out, activation, bias) in layers:
            layer = torch.nn.Linear(dim_in, dim_out, bias=bias).to(torch.device(device))
            self.params += layer.parameters()
            self.layers.append(layer)
            self.activations.append(activation)

        assert len(self.activations) == len(self.layers)

        self.optim = torch.optim.Adam(self.params, lr=lr)

    def forwards(self, state):
        h = torch.tensor(state, dtype=torch.float32).to(torch.device(device))
        for (layer, activation) in zip(self.layers, self.activations):
            h = layer(h)
            if activation is not None:
                h = activation(h)
        return h

    def chose_action(self, s):
        p_actions = self.predict(s)[0]
        return np.random.choice(p_actions.shape[0], p=p_actions)

    def fit(self, advantages, states, actions):
        advantage = torch.tensor(advantages).to(torch.device(device))  # T x 1
        actions = torch.tensor(actions).to(torch.device(device))  # T x 1

        self.optim.zero_grad()

        prediction = self.forwards(states)  # T x K
        h = prediction[torch.arange(len(actions)), actions]  # T x 1
        loss = - torch.sum(self.lr * advantage * torch.log(h))

        loss.backward()

        self.optim.step()

    def predict(self, state):
        return self.forwards(state).cpu().detach().numpy()


class ContinuousActionPolicyModel:
    def __init__(self, layers, lr=0.01, using_gd=True):
        self.layer_constructor = layers
        self.activations = []
        self.layers_mu = []
        self.layers_v = []
        self.params = []
        self.lr = lr
        self.H = layers[-1][1]

        mu_init = np.random.randn(self.H, 1)
        self.mu = torch.tensor(mu_init.astype(np.float32)).to(torch.device(device))
        self.params.append(self.mu)

        v_init = np.random.randn(self.H, 1)
        self.v = torch.tensor(v_init.astype(np.float32)).to(torch.device(device))
        self.params.append(self.v)

        for (dim_in, dim_out, activation, bias) in layers:
            layer = torch.nn.Linear(dim_in, dim_out, bias=bias).to(torch.device(device))
            self.params += layer.parameters()
            self.layers_mu.append(layer)

        for (dim_in, dim_out, activation, bias) in layers:
            layer = torch.nn.Linear(dim_in, dim_out, bias=bias).to(torch.device(device))
            self.params += layer.parameters()
            self.layers_v.append(layer)
            self.activations.append(activation)

        assert len(self.activations) == len(self.layers_mu) == len(self.layers_v)

        if using_gd:
            self.mu.requires_grad = True
            self.v.requires_grad = True
            self.optim = torch.optim.Adam(self.params, lr=lr)

    def forwards(self, state):
        # find mu
        h_mu = torch.tensor(state, dtype=torch.float32).to(torch.device(device))
        for (layer, activation) in zip(self.layers_mu, self.activations):
            h_mu = layer(h_mu)
            if activation is not None:
                h_mu = activation(h_mu)

        # h_mu = feature vector of size T x D
        mu = torch.matmul(h_mu, self.mu)  # T x 1

        # find v
        h_v = torch.tensor(state, dtype=torch.float32).to(torch.device(device))
        for (layer, activation) in zip(self.layers_v, self.activations):
            h_v = layer(h_v)
            if activation is not None:
                h_v = activation(h_v)

        # h_v = feature vector of size T x D
        v = torch.exp(torch.matmul(h_v, self.v))  # T x 1


        Z = torch.normal(mu, v)  # T x 1
        Z = torch.clip(Z, min=-1, max=-1)

        return Z

    def fit(self, advantages, states, actions):
        advantage = torch.tensor(advantages).to(torch.device(device))  # T x 1

        self.optim.zero_grad()

        prediction = self.forwards(states)  # T x 1
        loss = - torch.sum(self.lr * advantage * torch.log(prediction))

        loss.backward()

        self.optim.step()

    def predict(self, state):
        return self.forwards(state).cpu().detach().numpy()

    def chose_action(self, s):
        t = self.predict(s)
        return t[0]

    def mutate_weights(self, factor=0.05):
        for weight in self.params:
            noise = torch.rand(weight.shape) * factor
            weight += noise

    def create_copy(self):
        copy = ContinuousActionPolicyModel(self.layer_constructor, self.lr)
        for (i, (old, new)) in enumerate(zip(self.params, copy.params)):
            copy.params[i] = old.clone()
        return copy


class ValueModel:
    def __init__(self, layers, lr):
        self.activations = []
        self.layers = []
        self.params = []
        self.lr = lr

        for (dim_in, dim_out, activation, bias) in layers:
            layer = torch.nn.Linear(dim_in, dim_out, bias=bias).to(torch.device(device))
            self.params += layer.parameters()
            self.layers.append(layer)
            self.activations.append(activation)

        assert len(self.activations) == len(self.layers)

        self.criterion = torch.nn.MSELoss().to(torch.device(device)).to(torch.device(device))
        self.optim = torch.optim.Adam(self.params, lr=lr)

    def forwards(self, X):
        h = torch.tensor(X, dtype=torch.float32).to(torch.device(device))
        for (layer, activation) in zip(self.layers, self.activations):
            h = layer(h)
            if activation is not None:
                h = activation(h)
        return h

    def predict(self, X):
        return self.forwards(X).cpu().detach().numpy()

    def partial_fit(self, X, Y):
        self.optim.zero_grad()

        target = torch.tensor([Y], dtype=torch.float32).to(torch.device(device))
        target = torch.transpose(target, 0, 1)

        prediction = self.forwards(X)

        loss = self.criterion(prediction, target)
        loss.backward()

        self.optim.step()


class Agent:
    def __init__(self, env, policy, value, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.P = policy
        self.V = value

    def play_one(self, max_iter=10000, overide_done=2000):
        episode_reward = 0

        states = []
        actions = []
        rewards = []

        s = self.env.reset()

        for t in range(max_iter):
            a = self.P.chose_action([s])

            s_, r, done, info = self.env.step(a)
            episode_reward += r

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = s_

            if done and t > overide_done - 2:
                return states, actions, rewards

    def train_mc(self, episodes=1000, max_iter=2000, pritn_freq=100, overide_done=0):
        av_reward = 0
        for i in range(episodes):

            states, actions, rewards = self.play_one(max_iter, overide_done)
            av_reward += sum(rewards)

            returns = []
            advantages = []
            G = 0

            for s, r in zip(reversed(states), reversed(rewards)):
                returns.append(G)
                advantages.append(G - self.V.predict(s))
                G = r + self.gamma * G

            returns.reverse()
            advantages.reverse()

            self.P.fit(advantages, states, actions)
            self.V.partial_fit(states, returns)

            if i % pritn_freq == 0:
                print("Episode {}, Total reward : {}".format(i + 1, av_reward / pritn_freq))
                av_reward = 0

    def train_td(self, episodes=1000, max_iter=1000, pritn_freq=100):
        av_reward = 0
        for i in range(episodes):
            s = self.env.reset()
            episode_reward = 0
            for t in range(max_iter):
                a = self.P.chose_action([s])

                s_, r, done, info = self.env.step(a)
                episode_reward += r

                G = r + self.gamma * self.V.predict(s_)
                advantage = G - self.V.predict(s)

                self.P.fit([advantage], [s], [a])
                self.V.partial_fit([s], G)

                s = s_

                if done:
                    av_reward += episode_reward

                    if i % pritn_freq == 0:
                        print("Episode {} - Average reward : {}".format(i + 1, av_reward / pritn_freq))
                        av_reward = 0

                    break


def run_discrete_cartpole(use_td=False):
    if use_td:
        method = "TD"
    else:
        method = "MC"
    print('Cartpole Discerete Action space using {}'.format(method))

    env_name = "CartPole-v1"
    env = gym.make(env_name)

    actions = [0, 1]
    X = env.observation_space.shape[0]
    H = 500
    K = len(actions)
    lr = 0.01

    v_layers = [[X, H, torch.tanh, True],
                [H, 1, None, False]]
    V = ValueModel(v_layers, lr=lr)

    p_layers = [[X, K, lambda x: torch.softmax(x, 1), False]]
    P = PolicyModel(p_layers, lr=lr)

    agent = Agent(env, P, V)

    if use_td:
        agent.train_td()
    else:
        agent.train_mc()


def run_discrete_mountain_car(use_td=False):
    if use_td:
        method = "TD"
    else:
        method = "MC"
    print('Mountain Car Discerete Action space using {}'.format(method))

    env_name = "MountainCar-v0"
    env = gym.make(env_name)

    actions = [0, 1, 2]
    X = env.observation_space.shape[0]
    H = 100
    K = len(actions)
    lr = 0.1

    v_layers = [[X, H, torch.tanh, True],
                [H, 1, None, False]]
    V = ValueModel(v_layers, lr=lr)

    p_layers = [[X, K, lambda x: torch.softmax(x, 1), False]]
    P = PolicyModel(p_layers, lr=lr)

    agent = Agent(env, P, V)

    if use_td:
        agent.train_td(max_iter=1000, episodes=1000, pritn_freq=10)
    else:
        agent.train_mc(max_iter=20000, episodes=1000, pritn_freq=10, overide_done=20000)


def run_continous_mountain_car(use_td=False, use_gd=True):
    if use_td:
        method = "TD"
    else:
        method = "MC"
    if use_gd:
        optim = "Gradient Descent"
    else:
        optim = "Hill Climb"

    print('Mountain Car Continuous Action space using {} - {}'.format(method, optim))

    env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name)

    X = env.observation_space.shape[0]
    H = 500
    H_ = 100
    lr = 0.01

    v_layers = [[X, H, torch.tanh, True],
                [H, 1, None, False]]
    V = ValueModel(v_layers, lr=lr)

    p_layers = [[X, H_, torch.tanh, False]]
    P = ContinuousActionPolicyModel(p_layers, lr=lr)

    agent = Agent(env, P, V)

    if use_gd:
        if use_td:
            agent.train_td(max_iter=2000, episodes=10000, pritn_freq=10)
        else:
            agent.train_mc(max_iter=2000, episodes=10000, overide_done=2000, pritn_freq=10)
    else:
        epochs = 100
        episodes = 5
        max_iter = 100000

        best_policy = agent.P
        best_score = -10000000

        for e in range(epochs):
            av_reward = 0
            e_states = []
            e_actions = []
            e_rewards = []

            for i in range(episodes):
                states, actions, rewards = agent.play_one(max_iter)
                av_reward += sum(rewards)

                e_states.append(states)
                e_actions.append(actions)
                e_rewards.append(rewards)

            # check for new best
            av_reward = av_reward / episodes
            if av_reward >= best_score:
                best_score = av_reward
                best_policy = agent.P

            # clone and mutate
            new_policy = best_policy.create_copy()
            new_policy.mutate_weights()
            agent.P = new_policy

            print("Epoch {} - Best av reward : {}".format(e + 1, best_score))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # run_discrete_cartpole(use_td=False)
    # run_continous_mountain_car(use_gd=True, use_td=False)
    run_discrete_mountain_car(use_td=False)
