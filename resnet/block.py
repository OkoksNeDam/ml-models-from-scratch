import torch.nn as nn


class res_block(nn.Module):
    """
    A class that is used as a block for the ResNet architecture.
    """
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        """
        Method for initializing res_block parameters.
        Depending on the stride, the spatial dimension of the tensor changes.

        :param in_channels: number of channels on the input tensor.
        :param out_channels: number of channels on the output of res_block.
        :param identity_downsample: layer in which the size of the input tensor is adjusted to the size of the output.
        :param stride: stride for convolutional layer. If stride is not specified,
        then the spatial dimension does not change.
        """
        super(res_block, self).__init__()
        # how much more channels are there at the output than at the input.
        self.expansion = 4

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(out_channels / self.expansion),
            kernel_size=(1, 1)
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=int(out_channels / self.expansion)
        )

        self.conv2 = nn.Conv2d(
            in_channels=int(out_channels / self.expansion),
            out_channels=int(out_channels / self.expansion),
            kernel_size=(3, 3),
            # here, if necessary, the spatial dimension changes.
            stride=(stride, stride),
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=int(out_channels / self.expansion)
        )
        self.conv3 = nn.Conv2d(
            in_channels=int(out_channels / self.expansion),
            out_channels=out_channels,
            kernel_size=(1, 1)
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, X):
        # save X for its further addition with result of res_block.
        identity = X

        X = self.relu(self.bn1(self.conv1(X)))
        X = self.relu(self.bn2(self.conv2(X)))
        X = self.bn3(self.conv3(X))

        # If the layer involves changing spatial dimensions or the number of channels,
        # then use identity_downsample.
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # adding the result of the res_block and the saved X.
        X += identity
        X = self.relu(X)

        return X
