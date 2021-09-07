class Grid:
    def __init__(self, rows, cols, start_pos, max_moves=300):
        self.max_moves = max_moves
        self.rows = rows
        self.cols = cols
        self.rewards = [[0 for _ in range(rows)] for _ in range(cols)]
        self.walls = [[False for _ in range(rows)] for _ in range(cols)]
        self.terminals = []

        self.starting_pos = start_pos
        self.player_pos = start_pos
        self.moves = 0


    """
        Makes cell at loc a terminal cell
    """
    def set_as_terminal(self, loc):
        self.terminals.append(loc)

    """
    Makes cell at loc a wall
    """
    def set_as_wall(self, loc):
        self.walls[loc[0]][loc[1]] = True

    """
    Reset game
    """
    def reset_game(self):
        self.player_pos = self.starting_pos
        self.moves = 0

    """
    Makes cell at loc have reward 
    """
    def set_reward(self, loc, reward):
        self.rewards[loc[0]][loc[1]] = reward

    """
    Play Turn
    returns reward for next state
    """
    def play_turn(self, action):
        new_pos = self.get_next_state(self.player_pos, action)
        self.player_pos = new_pos
        self.moves += 1

        return self.get_reward(new_pos)

    """
    Returns current state
    """
    def get_current_state(self):
        return self.player_pos

    """
    Returns state if action is taken
    """
    def get_next_state(self, state, action):
        x = state[0]
        y = state[1]

        if action not in self.get_actions(state):
            raise Exception("{} is not allowed".format(action))

        if action == "L":
            x -= 1
        elif action == "R":
            x += 1
        elif action == "D":
            y -= 1
        elif action == "U":
            y += 1

        return [x, y]

    """
    Returns legal actions at state
    """
    def get_actions(self, pos):
        actions = []
        x = pos[0]
        y = pos[1]

        # left
        if x - 1 >= 0 and not self.walls[x - 1][y]:
            actions.append("L")
        # right
        if x + 1 <= self.cols - 1 and not self.walls[x + 1][y]:
            actions.append("R")
        # down
        if y - 1 >= 0 and not self.walls[x][y - 1]:
            actions.append("D")
        # up
        if y + 1 <= self.rows - 1 and not self.walls[x][y + 1]:
            actions.append("U")

        return actions

    """
    returns list of [x,y] states
    """
    def get_all_states(self):
        states = []
        for x in range(self.cols):
            for y in range(self.rows):
                if not self.walls[x][y]:
                    states.append([x, y])

        return states

    """
    returns reward at loc
    """
    def get_reward(self, loc):
        return self.rewards[loc[0]][loc[1]]

    """
    True is cell at loc is a wall
    """
    def is_wall(self, loc):
        return self.walls[loc[0]][loc[1]]

    """
    True is cell at loc is a terminal cell
    """
    def is_terminal(self, loc):
        return loc in self.terminals

    """
    :returns True if game is over
    """
    def game_over(self):
        return self.player_pos in self.terminals or self.moves >= self.max_moves

    # --------- PRIVATES -------------
    def find_occupier(self, loc):
        x = loc[0]
        y = loc[1]

        if self.player_pos == loc:
            return "X"

        if self.walls[x][y]:
            return "_"

        if loc in self.terminals:
            return "T"

        return " "

    def __str__(self):
        output = ""

        output += "  "
        for _ in range(self.cols):
            output += " ___"
        output += "\n"

        for r in range(self.rows - 1, -1, -1):
            output += "{} ".format(r)
            for c in range(self.cols):
                output += "| {} ".format(self.find_occupier([c, r]))
            output += "|\n"

            output += "  "
            for c in range(self.cols):
                output += "|___"
            output += "|\n"

        output += "  "
        for i in range(self.cols):
            output += "  {} ".format(i)
        output += "\n"
        return output
