import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
