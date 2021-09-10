import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms.functional as transforms

class Model(nn.Module):
    def __init__(self, img_dimention, num_classes, hidden_size):
        super(Model, self).__init__()
        self.rnn_len = nn.LSTM(img_dimention, hidden_size, batch_first=True, bidirectional=True)
        self.rnn_dep = nn.LSTM(img_dimention, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 4, num_classes)
        self.soft = nn.Softmax()

    def forward(self, x):
        # lengthwise
        len_output, (hn, cn) = self.rnn_len(x)
        len_output = len_output.view(len_output.shape[0], len_output.shape[1], 2, int(len_output.shape[2] / 2))
        len_output = torch.cat((len_output[:, -1, 0, :], len_output[:, -1, 1, :]), dim=1)

        # depthwise
        dep = torch.transpose(x, 1, 2)
        dep_output, (hn, cn) = self.rnn_dep(dep)
        dep_output = dep_output.view(dep_output.shape[0], dep_output.shape[1], 2, int(dep_output.shape[2] / 2))
        dep_output = torch.cat((dep_output[:, -1, 0, :], dep_output[:, -1, 1, :]), dim=1)

        concated = torch.cat((len_output, dep_output), dim=1)

        h = self.linear(concated)
        h = self.soft(h)

        return h


def evaluate_model(X_test, Y_test, model, e):
    with torch.no_grad():
        out = model(X_test)
        out = out.detach().numpy()
        out = out.argmax(axis=1)
        label = Y_test.detach().numpy()
        correct = out[out == label]
        print("Epoch: {}, accuracy: {}%".format(e, round(correct.shape[0] * 100 / label.shape[0])))


if __name__ == "__main__":
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)


    model = Model(mnist_trainset.train_data.shape[1], len(mnist_trainset.classes), 100)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    # train loop
    for e in range(10000):
        # mini batch sample
        idx = np.random.randint(0, mnist_trainset.train_data.shape[0], (128,))
        x_b = mnist_trainset.train_data[idx]
        y_b = mnist_trainset.train_labels[idx]

        # gradient descent
        optim.zero_grad()
        out = model(x_b.float())
        loss = crit(out, y_b.long())
        loss.backward()
        optim.step()

        # evaluation
        if e % 100 == 0:
            evaluate_model(mnist_testset.test_data.float(), mnist_testset.test_labels.long(), model, e)