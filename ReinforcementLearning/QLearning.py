from ReinforcementLearning.ValueEquationSolver import ValueEquationSolver
import random as rnd


class QLearning(ValueEquationSolver):
    def __init__(self, grid):
        super().__init__(grid)

    """
    Q Learning Algorithm,
    Sets values and policy by solving State-Value and Action-Value Equations.
    Q(s,a) = Q(s,a) + a[ R(t+1) + yQ(s',a') - Q(s,a)]
    
    N - num iterations
    alpha - learning rate
    gamma - discount rate
    """

    def Q_learning(self, N, alpha, gamma, scale_alpha=False):
        self.title = "Q - Learning"

        # Create Q Table, initialise  Q(s,a) = 0 for all s,a
        # Create count table to scale alpha and set to 1
        Q = [[{} for _ in range(self.grid.rows)] for _ in range(self.grid.cols)]
        counts = [[{} for _ in range(self.grid.rows)] for _ in range(self.grid.cols)]
        for x in range(self.grid.cols):
            for y in range(self.grid.rows):
                for action in self.grid.get_actions([x, y]):
                    Q[x][y][action] = 0
                    counts[x][y][action] = 1

        # T controls probability of exploring 1 -> 100% explore, 0 -> 100% exploit
        t = 1.0
        for iteration in range(1, N):
            # Decrease t every 100th iteration to exploit more and more
            if iteration % 100 == 0:
                t += 0.08

            # Initialise game, first state and first action
            self.grid.reset_game()
            s = self.grid.starting_pos
            a = self.explore_exploit_action(Q, s, t)

            # Play an episode, after each action update Q table:
            # Q(s,a) = Q(s,a) + a[ R(t+1) + yQ(s',a') - Q(s,a)]
            while not self.grid.game_over():
                r = self.grid.play_turn(a)
                s_2 = self.grid.get_current_state()
                a_2 = self.explore_exploit_action(Q, s_2, t)

                q = Q[s[0]][s[1]][a]
                q_2 = Q[s_2[0]][s_2[1]][a_2]

                # Scale alpha to slow learning rate or keep alpha
                if scale_alpha:
                    scaled_a = alpha / counts[s[0]][s[1]][a]
                    counts[s[0]][s[1]][a] += 1
                else:
                    scaled_a = alpha


                # update Q Table
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
    Exploits -> max(a)[Q(s,a)]
    Or
    Explore -> random a
    
    controlled by t
    """

    def explore_exploit_action(self, Q, s, t):
        if rnd.random() < 1 / t:
            return rnd.choice(self.grid.get_actions(s))
        else:
            a, _ = self.max_q_table(Q, s)
            return a
