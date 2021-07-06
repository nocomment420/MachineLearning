import torch
import numpy as np
import torch.nn as nn
import gym


class ReplayBuffer:
    def __init__(self, size, min_size, state_dims, num_actions):
        self.n = 0
        self.current = 0
        self.size = size
        self.min_size = min_size

        self.action_buffer = np.zeros((size, num_actions))
        self.reward_buffer = np.zeros(size)
        self.done_buffer = np.zeros(size)
        self.state_buffer = np.zeros((size, *state_dims))
        self.next_state_buffer = np.zeros((size, *state_dims))

    def add(self, state, action, reward, next_state, done):
        self.action_buffer[self.current] = action
        self.state_buffer[self.current] = state
        self.next_state_buffer[self.current] = next_state
        self.reward_buffer[self.current] = reward
        self.done_buffer[self.current] = done

        self.n = max(self.n, self.current + 1)
        self.current = (self.current + 1) % self.size

    def sample_buffer(self, batch_size):
        if self.n < self.min_size:
            return None

        idx = np.random.choice(np.arange(start=0, stop=self.n - 1), replace=False, size=batch_size)

        actions = self.action_buffer[idx]
        rewards = self.reward_buffer[idx]
        dones = self.done_buffer[idx]
        states = self.state_buffer[idx]
        next_states = self.next_state_buffer[idx]

        return states, actions, rewards, dones, next_states


class ActorNetwork(torch.nn.Module):
    def __init__(self, input_size, fc1, fc2, n_outputs, device):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(input_size, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, n_outputs),
            nn.Tanh()
        ).to(device=self.device)

    def forwards(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        output = self.network(state)

        return output

    def copy(self, source, T):
        for (source, target) in zip(source.parameters(), self.parameters()):
            target.data = (T * source.data.clone()) + ((1 - T) * target.data.clone())


class CriticNetwork(torch.nn.Module):
    def __init__(self, input_size, fc1, fc2, device):
        super(CriticNetwork, self).__init__()
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(input_size, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, 1)
        ).to(device=self.device)

    def forwards(self, state, action):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        # action = torch.tensor(action, dtype=torch.float32, device=self.device)

        input = torch.cat([state, action], dim=1)
        output = self.network(input)

        return output

    def copy(self, source, T):
        for (source, target) in zip(source.parameters(), self.parameters()):
            target.data = (T * source.data.clone()) + ((1 - T) * target.data.clone())


class Agent:
    def __init__(self, env, actor, critic, target_actor, target_critic, critic_optim, actor_optim, replay_buffer, tao,
                 batch_size, gamma, nois_std, device):
        self.env = env
        self.nois_std = nois_std
        self.target_critic = target_critic
        self.target_actor = target_actor
        self.critic = critic
        self.critic_optim = critic_optim
        self.actor = actor
        self.actor_optim = actor_optim
        self.max_action = self.env.action_space.high[0]
        self.min_action = self.env.action_space.low[0]
        self.replay_buffer = replay_buffer
        self.T = tao
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

    def chose_action(self, state, add_noise=True):
        action = self.target_actor.forwards(state).detach().cpu().numpy()
        if add_noise:
            action += np.random.normal(0, self.nois_std, [action.shape[0]])
        action = np.clip(action, self.min_action, self.max_action)

        return action

    def train(self):
        sample = self.replay_buffer.sample_buffer(self.batch_size)
        if sample is None:
            return

        states, actions, rewards, dones, next_states = sample

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

        # critic network
        self.critic_optim.zero_grad()
        target_actor_actions = self.target_actor.forwards(next_states)
        future_reward = torch.squeeze(self.target_critic.forwards(next_states, target_actor_actions))
        R = rewards + self.gamma * future_reward * (1 - dones)
        critic_val = torch.squeeze(self.critic.forwards(states, actions))
        critic_loss = torch.mean(torch.pow(R - critic_val, 2))
        critic_loss.backward()
        self.critic_optim.step()

        # actor network
        self.actor_optim.zero_grad()
        actor_actions = self.actor.forwards(states)
        actor_loss = -torch.mean(self.critic.forwards(states, actor_actions))
        actor_loss.backward()
        self.actor_optim.step()

    def update_target(self):
        self.target_actor.copy(self.actor, self.T)
        self.target_critic.copy(self.critic, self.T)

    def play_one(self, t, is_eval):
        episode_reward = 0

        s = self.env.reset()
        while True:
            if is_eval:
                self.env.render()
            a = self.chose_action(s)

            s_, r, done, info = self.env.step(a)
            episode_reward += r

            self.replay_buffer.add(s, a, r, s_, done)

            s = s_
            if not is_eval:
                self.train()
                self.update_target()

            if done:
                break
            t += 1

        return episode_reward, t


def train(agent, episodes, is_eval, save_freq=None):
    t = 0
    rewards = np.zeros(episodes)
    for e in range(episodes):
        reward, t = agent.play_one(t, is_eval)
        rewards[e] = reward
        average = rewards.sum() / (e + 0.000001)
        print("Episode {} : {} | Av : {}".format(e + 1, reward, round(average, 2)))
        if not is_eval and save_freq is not None and e % save_freq == 0:
            torch.save(agent.actor, "pendulum-actor.pt")
            torch.save(agent.critic, "pendulum-critic.pt")


def run(env_name, eval_filename=None):
    env = gym.make(env_name)

    num_inputs = env.observation_space.shape
    num_outputs = env.action_space.shape[0]
    fc1 = 400
    fc2 = 300
    lr1 = 0.0035
    lr2 = 0.0095
    buffer_max = 500000
    buffer_min = 100
    T = 0.005
    batch_size = 64
    gamma = 0.99
    num_episodes = 250
    noise_std = 0.1

    replay_buffer = ReplayBuffer(size=buffer_max,
                                 min_size=buffer_min,
                                 state_dims=num_inputs,
                                 num_actions=num_outputs)

    if eval_filename is not None:
        actor = torch.load("{}-actor-net.pt".format(eval_filename), map_location=torch.device('cpu'))
    else:
        actor = ActorNetwork(input_size=num_inputs[0],
                             fc1=fc1,
                             fc2=fc2,
                             n_outputs=num_outputs,
                             device=device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr1)
    target_actor = ActorNetwork(input_size=num_inputs[0],
                                fc1=fc1,
                                fc2=fc2,
                                n_outputs=num_outputs,
                                device=device)
    target_actor.copy(actor, 1)

    if eval_filename is not None:
        critic = torch.load("{}-critic-net.pt".format(eval_filename), map_location=torch.device('cpu'))
    else:
        critic = CriticNetwork(input_size=num_inputs[0] + num_outputs,
                               fc1=fc1,
                               fc2=fc2,
                               device=device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr2)
    target_critic = CriticNetwork(input_size=num_inputs[0] + num_outputs,
                                  fc1=fc1,
                                  fc2=fc2,
                                  device=device)
    target_critic.copy(critic, 1)

    agent = Agent(env=env,
                  actor=actor,
                  critic=critic,
                  target_actor=target_actor,
                  target_critic=target_critic,
                  critic_optim=critic_optim,
                  actor_optim=actor_optim,
                  replay_buffer=replay_buffer,
                  tao=T,
                  batch_size=batch_size,
                  gamma=gamma,
                  nois_std=noise_std,
                  device=device)

    train(agent=agent,
          episodes=num_episodes,
          save_freq=40,
          is_eval=eval_filename is not None)


def run_pendulum(is_eval):
    env_name = "Pendulum-v0"
    env = gym.make(env_name)

    num_inputs = env.observation_space.shape
    num_outputs = env.action_space.shape[0]
    fc1 = 400
    fc2 = 300
    lr1 = 0.0035
    lr2 = 0.0095
    buffer_max = 500000
    buffer_min = 100
    T = 0.005
    batch_size = 64
    gamma = 0.99
    num_episodes = 250
    noise_std = 0.1

    replay_buffer = ReplayBuffer(size=buffer_max,
                                 min_size=buffer_min,
                                 state_dims=num_inputs,
                                 num_actions=num_outputs)

    if is_eval:
        actor = torch.load("pendulum-actor.pt")
    else:
        actor = ActorNetwork(input_size=num_inputs[0],
                             fc1=fc1,
                             fc2=fc2,
                             n_outputs=num_outputs,
                             device=device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr1)
    target_actor = ActorNetwork(input_size=num_inputs[0],
                                fc1=fc1,
                                fc2=fc2,
                                n_outputs=num_outputs,
                                device=device)
    target_actor.copy(actor, 1)

    if is_eval:
        critic = torch.load("pendulum-critic.pt")
    else:
        critic = CriticNetwork(input_size=num_inputs[0] + num_outputs,
                               fc1=fc1,
                               fc2=fc2,
                               device=device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr2)
    target_critic = CriticNetwork(input_size=num_inputs[0] + num_outputs,
                                  fc1=fc1,
                                  fc2=fc2,
                                  device=device)
    target_critic.copy(critic, 1)

    agent = Agent(env=env,
                  actor=actor,
                  critic=critic,
                  target_actor=target_actor,
                  target_critic=target_critic,
                  critic_optim=critic_optim,
                  actor_optim=actor_optim,
                  replay_buffer=replay_buffer,
                  tao=T,
                  batch_size=batch_size,
                  gamma=gamma,
                  nois_std=noise_std,
                  device=device)

    train(agent=agent,
          episodes=num_episodes,
          save_freq=40,
          is_eval=is_eval)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # run_pendulum(is_eval=True)
    run("BipedalWalker-v3", eval_filename="bipedal")
