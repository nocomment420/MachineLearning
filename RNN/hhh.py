class ll(object):
    def __init__(self, value):
        self.value = value
        self.link = None


class RMAD(object):
    """Recursive Moving Average Delta
    period: number of datapoints to average over
    depth: number of derivations + 1"""

    def __init__(self, period, depth):
        self.period = period
        self.llHead = None
        self.llTail = None
        self.average = None
        if depth > 0:
            self.delta = RMAD(period, depth - 1)
        else:
            self.delta = None
        self.dps = 0  # datapoints

    def addVal(self, price):
        if self.dps == 0:  # instantiation
            dp = ll(price)
            self.llHead = dp
            self.llTail = dp

        if self.dps > 0:
            dp = ll(price)  # create the new head
            self.llHead.link = dp  # link the current head to the new head
            self.llHead = dp  # set the new head

        if self.dps == self.period - 1:  # calculate the average conventionally
            sum = 0
            item = self.llTail
            while item.link:
                sum += item.value
                item = item.link
            sum += item.value  # add the head, as it will fail the while item.link check
            self.average = sum / self.period

        if self.dps >= self.period:  # Update the average and Linked List
            prevAvg = self.average
            self.average -= (self.llTail.value / self.period)  # subtract the oldest value
            self.llTail = self.llTail.link  # update the tail
            self.average += (price / self.period)  # add the newest value
            if self.delta:
                self.delta.addVal(self.average - prevAvg)

        self.dps += 1


a = RMAD(5, 3)
for i in range(1, 25):
    print(i ** 2)
    a.addVal(i ** 2)
    print("a.average", a.average)
    print("a.delta.average", a.delta.average)
    print("a.delta.delta.average", a.delta.delta.average)
    print("a.delta.delta.delta.average", a.delta.delta.delta.average)
