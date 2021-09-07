from ReinforcementLearning.TabularRL.ValueEquationSolver import ValueEquationSolver

class DPLearning(ValueEquationSolver):
    def __init__(self, grid):
        super().__init__(grid)

    """
    Uses Dynamic Programing to solve State-Value and Action-Value functions
    repeats iterative policy calculation -> find max a' until convergence
    Q(s,a) = max(a') [r + gamma*Q(s',a')
    
    gamma - Discount Rate
    """

    def policy_evaluation(self, gamma):
        S = self.grid.get_all_states()

        # Update Action-Values by calculating State-Value function and finding max a
        # Repeat until convergence
        while True:
            policy_changed = False
            self.iterative_policy_evaluation(gamma)

            # For each state find max(a') [r + gamma*Q(s',a')
            for s in S:
                old_a = self.policy[s[0]][s[1]]

                # Calculate returns of each s'
                allowable_actions = self.grid.get_actions(s)
                values = []
                for action in allowable_actions:
                    next_state = self.grid.get_next_state(s, action)
                    reward = self.grid.get_reward(next_state)

                    value = reward + (gamma * self.values[next_state[0]][next_state[1]])

                    values.append([value, action])

                # sort values and select a' that maximises return
                values = sorted(values, key=lambda x: x[0], reverse=True)
                new_a = values[0][1]

                # Update Q Table
                if old_a != new_a:
                    policy_changed = True
                    self.policy[s[0]][s[1]] = new_a

            # Policy hasn't changed -> convergence -> return
            if not policy_changed:
                break

    """
        Uses Dynamic Programing to solve State-Value and Action-Value functions
        calates V Table and find max a' until convergence
        Q(s,a) = max(a') [r + gamma*Q(s',a')

        gamma - Discount Rate
    """

    def value_iteration(self, gamma):
        self.title = "DP Value iteration"

        # Initialise states and Value Table(stores array of each [return, action] for each state)
        S = self.grid.get_all_states()
        V = [[[0, "U"] for y in range(self.grid.rows)] for x in range(self.grid.cols)]

        threshold = 0.1

        # Repeat until convergence
        while True:
            d = 0

            # For each state,
            for s in S:
                old_v = V[s[0]][s[1]][0]
                new_v = 0

                if not self.grid.is_terminal(s):

                    # Calculate return for each action
                    allowable_actions = self.grid.get_actions(s)
                    values = []
                    for action in allowable_actions:
                        next_state = self.grid.get_next_state(s, action)
                        reward = self.grid.get_reward(next_state)

                        value = reward + (gamma * V[next_state[0]][next_state[1]][0])

                        values.append([value, action])

                    # Set V Table to max return
                    V[s[0]][s[1]] = sorted(values, key=lambda x: x[0], reverse=True)[0]
                    new_v = V[s[0]][s[1]][0]

                # d = largest step size
                d = max(d, abs(old_v - new_v))

            # if converged -> end loop
            if d <= threshold:
                break

        # For each state find optimal policy and value and store
        for s in S:
            if not self.grid.is_terminal(s):
                self.policy[s[0]][s[1]] = V[s[0]][s[1]][1]
                self.values[s[0]][s[1]] = V[s[0]][s[1]][0]

    """
    Dynamic Programming algorithm to determine State-Values
    V(s) = R(t+1) + gamma*V(s')
    gamma - Discount Rate
    
    """

    def iterative_policy_evaluation(self, gamma):
        self.title = "DP iterative policy evaluation"

        # Initialise all states and Value tables
        S = self.grid.get_all_states()
        V = [[0 for y in range(self.grid.rows)] for x in range(self.grid.cols)]

        threshold = 0.1

        # Loop will break when d < threshold (convergence)
        while True:
            d = 0

            # For each state, calculate  V(s) = R(t+1) + gamma*V(s')
            for s in S:
                old_v = V[s[0]][s[1]]
                new_v = 0

                if not self.grid.is_terminal(s):
                    action = self.policy[s[0]][s[1]]
                    s_2 = self.grid.get_next_state(s, action)
                    reward = self.grid.get_reward(s_2)

                    new_v = reward + (gamma * V[s_2[0]][s_2[1]])

                V[s[0]][s[1]] = new_v

                # d = largest step
                d = max(d, abs(old_v - new_v))

            # if values have converged -> break
            if d <= threshold:
                break

        self.values = V
