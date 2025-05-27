import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden1=128, hidden2=64, hidden3=32):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1),
            nn.Sigmoid()  # Binary classification
        )

    def forward(self, x):
        return self.model(x)
