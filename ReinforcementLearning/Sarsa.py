from ReinforcementLearning.ValueEquationSolver import ValueEquationSolver
import random as rnd


class Sarsa(ValueEquationSolver):
    def __init__(self, grid):
        super().__init__(grid)

    """
    Exploits -> max(a)[Q(s,a)] / Explore -> random a
    Controlled by epsilon
    q - Q Table
    s - State
    e - Epsilon
    """

    def epsilon_greedy_action(self, q, s, e):
        r = rnd.random()
        if r < e:
            return rnd.choice(self.grid.get_actions(s))
        else:
            a, _ = self.max_q_table(q, s)
            return a

    """
    SARSA Learning Algorithm,
    Sets values and policy by solving State-Value and Action-Value Equations.
    Q(s,a) = Q(s,a) + a[ R(t+1) + yQ(s',a') - Q(s,a)]
               s a       r           s'  a' -> SARSA :) 

    N - num iterations
    alpha - learning rate
    gamma - discount rate
    scale_alpha - will decrease alpha by 1/ num times calculated
    e - chance of exploring = e/t
    t0 - starting t value (chance of exploring = e/t)
    ti - t increases by ti every 100th iteration (chance of exploring = e/t)
    """

    def SARSA(self, N, alpha, gamma, scale_aplha=False, e=0.5, t0=1.0, ti=0.01):
        self.title = "SARSA"

        # Create Q Table, initialise  Q(s,a) = 0 for all s,a
        # Create count table to scale alpha and set to 1
        Q = [[{} for _ in range(self.grid.rows)] for _ in range(self.grid.cols)]
        counts = [[{} for _ in range(self.grid.rows)] for _ in range(self.grid.cols)]
        for x in range(self.grid.cols):
            for y in range(self.grid.rows):
                for action in self.grid.get_actions([x, y]):
                    Q[x][y][action] = 0
                    counts[x][y][action] = 1

        t = t0
        # T controls probability of exploring 1 -> 100% explore, 0 -> 100% exploit
        for iteration in range(1, N):
            # Decrease t every 100th iteration to exploit more and more
            if iteration % 100 == 0:
                t += ti

            # Initialise game, first state and first action
            self.grid.reset_game()
            s = self.grid.starting_pos
            a = self.epsilon_greedy_action(Q, s, e / t)

            # Play an episode, after each action update Q table:
            # Q(s,a) = Q(s,a) + a[ R(t+1) + yQ(s',a') - Q(s,a)]
            while not self.grid.game_over():
                r = self.grid.play_turn(a)
                s_2 = self.grid.get_current_state()
                a_2 = self.epsilon_greedy_action(Q, s_2, e / t)

                q = Q[s[0]][s[1]][a]
                q_2 = Q[s_2[0]][s_2[1]][a_2]

                # Scale alpha to slow learning rate, increase count
                if scale_aplha:
                    scaled_a = alpha / counts[s[0]][s[1]][a]
                else:
                    scaled_a = alpha

                counts[s[0]][s[1]][a] += 1

                # Update Q Table
                Q[s[0]][s[1]][a] = q + (scaled_a * (r + (gamma * q_2) - q))

                s = s_2
                a = a_2

            # For each state find optimal policy and value and store
            for s in self.grid.get_all_states():
                if not self.grid.is_terminal(s) and not self.grid.is_wall(s):
                    # Find best action and value for state
                    action, max = self.max_q_table(Q, s)

                    self.policy[s[0]][s[1]] = action
                    self.values[s[0]][s[1]] = max

    """
       Semi Gradient SARSA Learning Algorithm,
       Q(s,a) -> w' . X 
       X = linear approximation features from self.state_action_to_x()
       w = weight vector this method finds using gradient descent
       w = w + alpha*[r + gamma*Q(s',a') - Q(s,a)] , where Q(s,a) = w' . X 
    
       N - num iterations
       alpha - learning rate
       gamma - discount rate
       scale_alpha - will decrease alpha by 1/ num times calculated
       e - chance of exploring = e/t
       t0 - starting t value (chance of exploring = e/t)
       ti - t increases by ti every iteration (chance of exploring = e/t)
   """
    def semi_gradient_SARSA(self, N, alpha, gamma, scale_aplha=False, e=0.5, t0=1.0, ti=0.01):
        self.title = "Semi Gradient SARSA"
        t = t0
        # T controls probability of exploring 1 -> 100% explore, 0 -> 100% exploit
        for iteration in range(1, N):
            # Decrease t every 100th iteration to exploit more and more
            if iteration % 100 == 0:
                t += ti

            # Initialise game, first state and first action
            self.grid.reset_game()
            s = self.grid.starting_pos
            a = self.epsilon_greedy_action_linear_approx(s, e / t, w)

            # Play an episode, after each action update w:
            # w = w + alpha*[r + gamma*Q(s',a') - Q(s,a)] , where Q(s,a) = w' . X
            while not self.grid.game_over():
                r = self.grid.play_turn(a)
                s_2 = self.grid.get_current_state()
                a_2 = self.epsilon_greedy_action_linear_approx( s_2, e / t,w)

                x = self.state_action_to_x(s, a)
                x_2 = self.state_action_to_x(s_2, a_2)

                # Scale alpha to slow learning rate, increase count
                if scale_aplha:
                    scaled_a = alpha / counts[s[0]][s[1]][a]
                else:
                    scaled_a = alpha

                counts[s[0]][s[1]][a] += 1

                if self.grid.is_terminal(s_2):
                    target = r
                else:
                    target = r + (gamma * w.dot(x_2))

                w += alpha * (target - w.dot(x)) * x


                s = s_2
                a = a_2

            # For each state find optimal policy and value and store
            for s in self.grid.get_all_states():
                if not self.grid.is_terminal(s) and not self.grid.is_wall(s):
                    # Find best action and value for state
                    action, max = self.max_q_table(Q, s)

                    self.policy[s[0]][s[1]] = action
                    self.values[s[0]][s[1]] = max

    def epsilon_greedy_action_linear_approx(self, s, p, w):
        pass