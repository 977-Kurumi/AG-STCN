import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_size, time_steps, hidden_size1,hidden_size2 ,output_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size*time_steps, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1],-1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        out = self.fc3(x)
        return out
if __name__ == '__main__':

    test = torch.rand(32,207, 12, 2)

    net = FNN(input_size=test.shape[3], time_steps=test.shape[2],hidden_size1=32, hidden_size2=64, output_size=3)


    out = net(test)