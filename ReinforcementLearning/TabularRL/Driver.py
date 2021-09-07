from ReinforcementLearning.TabularRL.GridWorld import Grid
from ReinforcementLearning.TabularRL.QLearning import QLearning
from ReinforcementLearning.TabularRL.Sarsa import Sarsa
from ReinforcementLearning.TabularRL.MonteCarloLearning import MonteCarloLearningAlgorithm
from ReinforcementLearning.TabularRL.DPLearning import DPLearning
from ReinforcementLearning.TabularRL.TemporalDifferenceLearning import TD0

def hard_grid():
    g = Grid(10, 10, [0, 4])
    g.set_as_terminal([2, 9])

    for i in range(10):
        for j in range(10):
            g.set_reward([i, j], -0.1)
    g.set_reward([2, 9], 10000)

    wall_up(g, [4, 7], [4, 9])
    wall_up(g, [6, 1], [6, 8])
    wall_up(g, [0, 5], [6, 5])
    wall_up(g, [0, 3], [3, 3])
    wall_up(g, [4, 1], [6, 1])
    wall_up(g, [1, 7], [4, 7])

    return g

def easy_grid():
    grid = Grid(5, 5, [0, 0])
    grid.set_as_terminal([2, 4])
    for i in range(5):
        for j in range(5):
            grid.set_reward([i, j], -1)
    grid.set_reward([2, 4], 100)

    for i in range(4):
        grid.set_as_wall([i, 3])

    return grid

def wall_up(g, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    if x1 != x2 and y1 != y2:
        raise Exception("points not lined up")

    if x1 == x2:
        for i in range(y1, y2 + 1, 1):
            g.set_as_wall([x1, i])
    elif y1 == y2:
        for i in range(x1, x2 + 1, 1):
            g.set_as_wall([i, y1])

def medium_grid():
    g = Grid(10, 10, [5, 0])
    for i in range(10):
        for j in range(10):
            g.set_reward([i, j], -1)
    g.set_as_terminal([5, 4])
    g.set_reward([5, 4], 100)

    wall_up(g, [3, 3], [3, 7])
    wall_up(g, [3, 3], [7, 3])
    wall_up(g, [7, 3], [7, 8])

    return g

def run_on_grid(g, file_name=None, verbose=True):
    file_string = ""

    next_string = str(g)
    file_string += next_string

    q = QLearning(g)
    q.Q_learning(1000, 0.1, 0.9)
    next_string = str(q)
    file_string += next_string

    s = Sarsa(g)
    s.SARSA(1000, 0.1, 0.9)
    next_string = str(s)
    file_string += next_string

    d = DPLearning(g)
    d.value_iteration(0.9)
    next_string = str(d)
    file_string += next_string
    d = DPLearning(g)
    d.policy_evaluation(0.9)
    next_string = str(d)
    file_string += next_string

    m = MonteCarloLearningAlgorithm(g)
    m.monte_carlo_exploring_starts(3000, 0.9)
    next_string = str(m)
    file_string += next_string

    if file_name is not None:
        with open("{}.txt".format(file_name), "w") as file:
            file.write(file_string)
            file.close()

    if verbose:
        print(file_string)

run_on_grid(hard_grid(),"RL-algorithms")