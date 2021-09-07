from ReinforcementLearning.TabularRL.ValueEquationSolver import ValueEquationSolver
import random as rnd
import numpy as np

class TD0(ValueEquationSolver):
    def __init__(self, grid):
        super().__init__(grid)

    def solve(self):
        pass

    """
        Plays an episode, returns [state, reward] list for each state encountered
    """

    def play_episode(self, epsilon):
        self.grid.reset_game()

        # stores [state, reward], initialise to starting position
        state_rewards = [[self.grid.player_pos, 0]]

        s = self.grid.player_pos

        # play episode, store state reward each time
        while not self.grid.game_over():
            a = self.epsilon_greedy_action(epsilon, s)
            r = self.grid.play_turn(a)
            s = self.grid.get_current_state()
            state_rewards.append([s, r])

        return state_rewards

    def epsilon_greedy_action(self, e, s):
        rand = rnd.random()
        if rand < e:
            a = rnd.choice(self.grid.get_actions(s))
        else:
            a = self.policy[s[0]][s[1]]
        return a

    """
        Temporal Difference 0 Learning Algorithm,
        Sets values and policy by solving State-Value function
        V(s) = v(s) + a[ r + gamma*V(s') - V(S) ]  
        N - num iterations
        alpha - learning rate
        gamma - discount rate
        epsilon - for epsilon greedy
    """

    def TD0(self, N, alpha, gamma, epsilon):
        self.title="TD0"

        for _ in range(N):
            state_rewards = self.play_episode(epsilon)

            # for each s, s' in state rewards:
            # V(s) = v(s) + a[ r + gamma*V(s') - V(S) ]
            for i in range(len(state_rewards) - 1):
                s, _ = state_rewards[i]
                s_2, r = state_rewards[i + 1]

                v_1 = self.values[s[0]][s[1]]
                v_2 = self.values[s_2[0]][s_2[1]]

                self.values[s[0]][s[1]] = v_1 + (alpha * (r + (gamma * v_2) - v_1))

    """
      Temporal Difference 0 Learning Algorithm using linear approximation
      solves State-Value function by gradient descent
      O = O + alpha* (r + (gamma * (O' . x') - (O' . x))x)
      
      N - num iterations
      alpha - learning rate
      gamma - discount rate
      epsilon - for epsilon greedy
    """

    def TD0_semi_gradient_prediction(self, N, alpha, gamma, epsilon):
        self.title = "TD0_semi_gradient_prediction"

        # Calculate dimentions for theta
        dimentions = self.map_state_to_x(self.grid.starting_pos).size

        # Initialise theta with random values
        theta = np.random.randn(dimentions) / (dimentions / 2)

        for _ in range(N):
            state_rewards = self.play_episode(epsilon)

            # for each s, s' in state rewards:
            # O = O + alpha* (r + (gamma * (O' . x') - (O' . x))x)
            for i in range(len(state_rewards) - 1):
                s, _ = state_rewards[i]
                s_2, r = state_rewards[i + 1]

                x = self.map_state_to_x(s)
                x_2 = self.map_state_to_x(s_2)

                if self.grid.is_terminal(s_2):
                    target = r
                else:
                    target = r + (gamma * theta.dot(x_2))

                theta += alpha * (target - theta.dot(x)) * x

        self.estimate_value_from_approx(theta)