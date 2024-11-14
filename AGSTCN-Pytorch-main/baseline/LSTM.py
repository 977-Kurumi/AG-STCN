import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*12, output_size)

    def forward(self, x):
        batch,num_nodes,time_steps,num_sp = x.shape
        x = x.reshape(x.shape[0] * x.shape[1],
                  x.shape[2], x.shape[3])
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(batch,num_nodes,-1)
        out = self.fc(out)
        return out



if __name__ == '__main__':

    test = torch.rand(32,207, 12, 2)

    net = LSTM(input_size=test.shape[3], hidden_size=64, num_layers=2, output_size=3)


    out = net(test)