from ReinforcementLearning.TabularRL.ValueEquationSolver import ValueEquationSolver
import random as rnd
import numpy as np

class MonteCarloLearningAlgorithm(ValueEquationSolver):
    def __init__(self, grid):
        super().__init__(grid)

    """
    Records [state, reward] for each play,
    Then returns [state, return] (return(G) = R(t) + yR(t+1) + y^2R(t+2)...    
    gamma - dicount rate
    """

    def play_episode(self, gamma):
        self.grid.reset_game()

        # Initialise [state, reward] for starting pos
        state_reward = [[self.grid.player_pos, self.grid.get_reward(self.grid.player_pos)]]

        while not self.grid.game_over():
            state = self.grid.get_current_state()

            # Chooses a random action 30% of the time and policy the rest of the time
            if rnd.random() < 0.3:
                action = rnd.choice(self.grid.get_actions(state))
            else:
                action = self.policy[state[0]][state[1]]

            reward = self.grid.play_turn(action)
            state_reward.append([self.grid.player_pos, reward])

        G = 0
        state_return = []
        # for each [state, reward], calculate [state, return]
        for s, r in reversed(state_reward):
            state_return.append([s, G])

            G = r + (gamma * G)

        return reversed(state_return)

    """
    Monty Carlo Algorithm to solve Value function
    V(s) = E[G(t)|s]
           1/n sum(G(t))
    N - num iterations
    gamma - Discount rate
    """

    def first_visit_monte_carlo(self, N, gamma):
        self.title = "MC first visit prediction"

        # Initialise count table to calculate sample mean in-place
        counts = [[0 for y in range(self.grid.rows)] for x in range(self.grid.cols)]

        for _ in range(N):
            state_return = self.play_episode(gamma)

            for sr in state_return:
                old_mean = self.values[sr[0][0]][sr[0][1]]
                old_count = counts[sr[0][0]][sr[0][1]]

                new_mean = ((old_mean * old_count) + sr[1]) / (old_count + 1)

                counts[sr[0][0]][sr[0][1]] += 1
                self.values[sr[0][0]][sr[0][1]] = new_mean

    """
    Plays episode from random start position,
    Records [state, action, reward] for each play,
    Then returns [state, action, return] (return(G) = R(t) + yR(t+1) + y^2R(t+2)...
    gamma - Discount rate
    """

    def play_episode_exploring_starts(self, gamma):
        self.grid.reset_game()

        # Initialise all states and random starting position
        S = self.grid.get_all_states()
        start_state = rnd.choice(S)
        # Ensure starting pos not wall or terminal
        while self.grid.is_wall(start_state) or self.grid.is_terminal(start_state):
            start_state = rnd.choice(S)

        # Set starting pos on grid
        self.grid.player_pos = start_state

        # Set random first action
        action = rnd.choice(self.grid.get_actions(start_state))
        self.policy[start_state[0]][start_state[1]] = action

        # initialise sar to start pos
        state_action_rewards = [[start_state, action, 0]]

        seen_states = []
        num_steps = 0

        # While game is ongoing, play action and store [state, action, reward]
        while True:
            state = self.grid.get_current_state()
            action = self.policy[state[0]][state[1]]
            reward = self.grid.play_turn(action)
            num_steps += 1

            # Cell already visited -> end game ( to avoid infinite loops)
            if state in seen_states:
                reward = -10. / num_steps
                state_action_rewards.append([state, None, reward])
                break
            elif self.grid.game_over():
                state_action_rewards.append([state, None, reward])
                break
            else:
                state_action_rewards.append([state, action, reward])
                seen_states.append(state)

        G = 0
        state_action_return = []
        first = True
        # for each [state, action, reward] calculate [state, action, return]
        # Return G(t) = R(t) + yR(t+1) + y^2R(t+2)...
        for s, a, re in reversed(state_action_rewards):
            # Ignore first entry as was random starting position
            if first:
                first = False
            else:
                state_action_return.append([s, a, G])
            G = re + (gamma * G)

        return reversed(state_action_return)

    """
        Finds Optimal policy and values by solving Action-Value and State-Value functions
         Q(s,a) = (( Q(s,a) * counts(s,a) ) + return ) /(counts(s,a) + 1 )
                        (old total + new Val) / old count + 1
        N - num iterations
        gamma - discount rate
    """

    def monte_carlo_exploring_starts(self, N, gamma):
        self.title = "MC Exploring starts"

        # Create Q Table, initialise  Q(s,a) = 0 for all s,a
        # Create counts table to calculate simulation means in-place
        Q = [[{} for y in range(self.grid.rows)] for x in range(self.grid.cols)]
        counts = [[{} for y in range(self.grid.rows)] for x in range(self.grid.cols)]
        for x in range(self.grid.cols):
            for y in range(self.grid.rows):
                for action in self.grid.get_actions([x, y]):
                    Q[x][y][action] = 0
                    counts[x][y][action] = 0

        # For each iteration simulate game, get state action returns and update Q(s,a)
        for _ in range(N):
            state_action_returns = self.play_episode_exploring_starts(gamma)

            for sar in state_action_returns:
                s = sar[0]
                a = sar[1]
                r = sar[2]

                # Update Q Table mean and increase count
                Q[s[0]][s[1]][a] = ((Q[s[0]][s[1]][a] * counts[s[0]][s[1]][a]) + r) / (counts[s[0]][s[1]][a] + 1)
                counts[s[0]][s[1]][a] += 1

            # For each state find optimal policy and value and store
            for s in self.grid.get_all_states():
                if not self.grid.is_terminal(s) and not self.grid.is_wall(s):
                    # Find best action and value for state
                    action, max = self.max_q_table(Q, s)

                    self.policy[s[0]][s[1]] = action
                    self.values[s[0]][s[1]] = max


    """
    Monte Carlo Algorithm using linear approximation
    Will perform gradient descent to find weights vector that best approximates v(s)
    At each iteartion O = O + alpha( G - V(s,O))x
                        = O + alpha( g - O'x)x
    
    N - num iterations
    gamma - Dicount Rate
    alpha - Learning Rate
    """

    def MC_linear_approx_prediction(self, N, gamma, alpha):
        self.title="MC linear approx prediction"

        # Calculate dimentions for theta
        dimentions = self.map_state_to_x(self.grid.starting_pos).size

        # Initialise theta with random values
        theta = np.random.randn(dimentions) / (dimentions / 2)

        # T controls probability of exploring 1 -> 100% explore, 0 -> 100% exploit
        t = 1.0
        for it in range(N):
            # Decrease t every 100th iteration to exploit more and more
            if it % 100 == 0:
                t += 0.01

            state_returns = self.play_episode(gamma)

            # For each step in the episode, perform gradient descent to improve estimate of theta
            # O = O + alpha(g - O'x)x
            seen_states = []
            for sg in state_returns:
                s = sg[0]
                g = sg[1]

                if s not in seen_states:
                    x = self.map_state_to_x(s)
                    v_hat = theta.dot(x)

                    scaler = alpha / t
                    theta += scaler*(g - v_hat)*x

                    seen_states.append(s)

        # Estimate value function
        self.estimate_value_from_approx(theta)

