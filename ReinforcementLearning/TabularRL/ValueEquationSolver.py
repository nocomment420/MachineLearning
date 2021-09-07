import random
import numpy as np

class ValueEquationSolver:
    def __init__(self, grid):
        self.grid = grid
        self.policy = [[self.random_move(x, y) for y in range(self.grid.rows)] for x in range(self.grid.cols)]
        self.values = [[0 for _ in range(self.grid.rows)] for _ in range(self.grid.cols)]
        self.init_means_std()
        self.path = None
        self.title = ""

    """Generates a legal random move from pos [x,y]"""
    def random_move(self, x, y):
        if self.grid.is_wall([x, y]):
            return "-"
        elif self.grid.is_terminal([x, y]):
            return "!"

        loc = [x, y]
        actions = self.grid.get_actions(loc)
        index = random.randint(0, len(actions) - 1)
        return actions[index]

    def solve(self):
        pass

    """
        Finds the mean and stdev of x1, x2 and interaction term
        """

    def init_means_std(self):
        x1_s = [i for i in range(self.grid.cols)]
        x2_s = [i for i in range(self.grid.rows)]
        int = []
        for x in x1_s:
            for y in x2_s:
                int.append(x * y)

        self.means_std = np.mean(x1_s), np.mean(x2_s), np.mean(int), np.std(x1_s), np.std(x2_s), np.std(int)

    """
    Maps state s to feature vector x
    x = [1, x-cord, y-cord, x-cord*y-cord]
    """

    def map_state_to_x(self, s):
        x1 = s[0]
        x2 = s[1]
        x1_m, x2_m, int_m, x1_d, x2_d, int_d = self.means_std
        return np.array([(x1 - x1_m) / x1_d, (x2 - x2_m) / x2_d, (x1 * x2 - int_m) / int_d, 1])

    """
       Approximates Value Function by linear approximation using theta estimate
       V(s) = O' . x
       """

    def estimate_value_from_approx(self, theta):
        S = self.grid.get_all_states()

        # For each state approximate value
        for s in S:
            if not self.grid.is_terminal(s) and not self.grid.is_wall(s):
                x = self.map_state_to_x(s)
                self.values[s[0]][s[1]] = theta.dot(x)


    """Returns max(a)[Q(s,a)]"""
    def max_q_table(self, Q, s):
        qs = Q[s[0]][s[1]]
        max = -1000000
        max_a = ""
        for a in qs.keys():
            val = qs[a]
            if val > max:
                max_a = a
                max = val
        if max_a == "":
            raise Exception("no max action found in q table")
        return max_a, max


    def __str__(self):
        self.find_path()
        return self.print_all()

    def print_all(self):
        output = "\n\n\n\n\n{}\n\n".format(self.title)

        # ---- Labels ----
        output += " " * ((self.grid.cols * 2) - 1)
        output += "Path  "
        output += " " * (((self.grid.cols * 2) - 1) + 4)
        output += " " * ((self.grid.cols * 4) - 1)
        output += "Values"
        output += " " * (((self.grid.cols * 4) - 1) + 3)
        output += " " * ((self.grid.cols * 2) - 1)

        output += "Policies\n"

        # ----- Top Bar -----
        # Left
        output += "  "
        for _ in range(self.grid.cols):
            output += "    "

        # Middle
        output += "     "

        # Right
        output += "  "

        for _ in range(self.grid.cols):
            output += " _______"

        # Middle2
        output += "     "

        # Right2
        output += "  "
        for _ in range(self.grid.cols):
            output += " ___"
        output += "\n"

        # ----- Middle -----
        for r in range(self.grid.rows - 1, -1, -1):

            # Row 1
            # Left
            output += "{} ".format(r)
            for c in range(self.grid.cols):
                output += "  {} ".format(self.format_path(c,r))
            # Middle
            output += "     "
            # Right
            output += "{} ".format(r)
            for c in range(self.grid.cols):
                output += "|{}".format(self.format_value(c, r))

            # Middle2
            output += "|    "
            # Right2
            output += "{} ".format(r)
            for c in range(self.grid.cols):
                output += "| {} ".format(self.policy[c][r])
            output += "|\n"

            # Row 2
            # Left
            output += "  "
            for c in range(self.grid.cols):
                output += "    "
            # Middle
            output += "     "
            # Right
            output += "  "
            for c in range(self.grid.cols):
                output += "|_______"

            # Middle2
            output += "|    "
            # Right2
            output += "  "
            for c in range(self.grid.cols):
                output += "|___"
            output += "|\n"

        # X - axis
        output += "  "

        for i in range(self.grid.cols):
            output += "  {} ".format(i)
        output += "     "
        output += "  "
        for i in range(self.grid.cols):
            output += "    {}   ".format(i)

        output += "     "
        output += "  "
        for i in range(self.grid.cols):
            output += "  {} ".format(i)

        return output

    """Formats cell ocuupier to print"""
    def format_value(self, x, y):
        value = self.values[x][y]
        value_string = str(value)

        if len(value_string) == 7:
            return value_string
        elif len(value_string) > 7:
            return value_string[0:7]

        while len(value_string) < 7:
            value_string = " {}".format(value_string)

        return value_string

    def find_path(self):
        path = []
        next_node = self.grid.starting_pos

        while not self.grid.is_terminal(next_node):
            path.append(next_node)
            if self.policy[next_node[0]][next_node[1]] == "U":
                next_node = [next_node[0], next_node[1] + 1]
            elif self.policy[next_node[0]][next_node[1]] == "D":
                next_node = [next_node[0], next_node[1] - 1]
            elif self.policy[next_node[0]][next_node[1]] == "R":
                next_node = [next_node[0] + 1, next_node[1]]
            elif self.policy[next_node[0]][next_node[1]] == "L":
                next_node = [next_node[0] - 1, next_node[1]]

            if len(path) > 100:
                return None

        self.path = path

    """Formats cell for path printing"""
    def format_path(self, c, r):
        if self.grid.is_wall([c, r]):
            return "+"
        elif self.grid.is_terminal([c, r]):
            return "!"
        elif self.grid.starting_pos == [c, r]:
            return "X"
        elif self.path is not None and [c, r] in self.path:
            if self.policy[c][r] == "U" or self.policy[c][r] == "D":
                return "|"
            elif self.policy[c][r] == "L" or self.policy[c][r] == "R":
                return "-"
        else:
            return " "




