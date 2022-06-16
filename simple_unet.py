import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden=256, sigmoid_act=True):
        super().__init__()
        self.entry_layer = nn.Conv1d(in_channel, hidden, 1)
        self.down1 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.down2 = nn.Conv1d(hidden, hidden, 5, padding=2)

        self.down3 = nn.Conv1d(hidden, hidden, 5, padding=2)

        self.down_sample = nn.MaxPool1d(2, stride=2)

        self.up2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.up1 = nn.Conv1d(hidden, out_channel, 5, padding=2)

        self.sigmoid_act = sigmoid_act

    def forward(self, x):
        x = self.entry_layer(x)

        d1 = self.down1(x)
        d1 = self.down_sample(d1)

        d2 = self.down2(d1)
        d2 = self.down_sample(d2)

        d3 = self.down3(d2)

        u2 = F.interpolate(d3, size=d1.shape[-1])
        u2 = self.up2(u2 + d1)

        u1 = F.interpolate(u2, size=x.shape[-1])
        u1 = self.up1(u1 + x)

        if self.sigmoid_act:
            u1 = u1.sigmoid()

        return u1
