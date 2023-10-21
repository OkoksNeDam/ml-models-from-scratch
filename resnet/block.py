import torch.nn as nn


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(res_block, self).__init__()
        # how much more channels are there at the output than at the input?
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                               stride=(stride, stride), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, X):
        # save X for its further addition with F(X).
        identity = X

        X = self.conv1(X)
        X = self.relu(self.bn1(X))
        X = self.conv2(X)
        X = self.relu(self.bn2(X))
        X = self.conv3(X)
        X = self.bn3(X)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # adding the result of the layers and the saved X.
        X += identity
        X = self.relu(X)

        return X
