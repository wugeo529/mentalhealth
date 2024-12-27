import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = nn.init.zeros_(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        c0 = nn.init.zeros_(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 最后一时间步输出
        out = self.fc(out)
        return out
