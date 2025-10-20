import torch.nn as nn


class MultiHeadClassifier(nn.Module):
    def __init__(self, in_features, *num_classes):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(in_features, n) for n in num_classes
        ])

    def forward(self, x):
        return [head(x) for head in self.heads]  # returns a list of outputs
