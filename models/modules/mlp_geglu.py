import torch.nn as nn


class GEGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activate=nn.GELU,  drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.dual_fc = nn.ModuleList([nn.Linear(in_features, hidden_features) for _ in range(2)])
        self.activate = activate()
        self.fc = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.activate(self.dual_fc[0](x)) * self.dual_fc[1](x)
        x = self.dropout(x)
        x = self.fc(x)

        return x
