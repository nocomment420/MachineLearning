def generate(id, topic):
    return "UPDATE `cms_training`  SET `CrmLeadTopic` = '{}' where (`Id` = {})".format(topic, id)

def generate_whitepaper(id, topic):
    return "UPDATE `cms_whitepaper`  SET `CrmLeadTopic` = '{}' where (`Id` = {})".format(topic, id)

def read(filename):
    df = pd.read_csv(filename)
    for id,topic in zip(df.Id, df.Topic):
        if not pd.isna(id) and not pd.isna(topic) :
            print(generate_whitepaper(int(id),topic))

class Process:
    def __init__(self, t, d):
        self. t = t
        self.d = d
    def __str__(self):
        return "T: {} | D: {}\n".format(self.t, self.d)

def printmemo(memo):
    for row in memo:
        print(row)

def greedy():
    processes = [Process(2,4), Process(3,6),Process(5,9),Process(5,9),Process(9,12),Process(7,14), Process(2,16),Process(3,22),Process(4,15),Process(2,9),Process(6,19),Process(3,26)]
    processes = sorted(processes, key=lambda x: x.d)
    print(processes)
    time = 0
    steps = []
    for process in processes:
        if time + process.t <= process.d:
            time += process.t
            steps.append(process)

    for step in steps:
        print(str(step))


def dp():

    T = 15
    processes = [Process(2,4), Process(3,6),Process(5,9),Process(5,9),Process(9,12),Process(7,14), Process(2,16),Process(3,22),Process(4,15),Process(2,9),Process(6,19),Process(3,26)]

    memo = [[0 for _ in range(8)] for _ in range(8)]

    for i in range(8):
        if processes[i].t > processes[i].d:
            memo[0][i] = 10000000
        else:
            memo[0][i] = processes[i].t
    #printmemo(memo)

    for i in range(1,8):
        for j in range(8):
            if memo[i-1][j] + processes[j].t > processes[j].d:
                memo[i][j] = memo[i][j-1]
            else:
                if j ==0:
                    memo[i][j] =memo[i-1][j] +  processes[j].t

                else:
                    memo[i][j] = max(memo[i-1][j], memo[i][j-1] +  processes[j].t)


    printmemo(memo)


greedy()
