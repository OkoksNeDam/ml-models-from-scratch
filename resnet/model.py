import torch.nn as nn
from block import res_block


class ResNet(nn.Module):
    """
    A class that describes the ResNet architecture.
    """

    def __init__(self, layers, image_channels, num_classes, blocks_out_channels):
        """
        Method for initializing ResNet neural network.
        :param layers: layers: how many times we want to use res_block. Ex: for resnet50 it's [3, 4, 6, 3].
        First, 3 blocks are used, then 4, etc.
        :param image_channels: image_channels: number of channels for input image. For RGB it's 3.
        :param num_classes: what number of classes is used for classification task.
        :param blocks_out_channels: list of number of output channels for each block.
        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=64
        )

        # this number of channels describes the current state of the tensor.
        # after the input image passed through conv1, the number of channels became 64.
        self.in_channels = 64

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # creating the blocks from which the resnet architecture is built.
        self.layer1 = self._make_layer(num_residual_blocks=layers[0], out_channels=blocks_out_channels[0], stride=1)
        self.layer2 = self._make_layer(num_residual_blocks=layers[1], out_channels=blocks_out_channels[1], stride=2)
        self.layer3 = self._make_layer(num_residual_blocks=layers[2], out_channels=blocks_out_channels[2], stride=2)
        self.layer4 = self._make_layer(num_residual_blocks=layers[3], out_channels=blocks_out_channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, num_residual_blocks, out_channels, stride):
        """
        A function that creates a block from several res_blocks.
        :param num_residual_blocks: number of res_blocks in current block.
        :param out_channels: number of channels in the output from each res_block.
        :param stride: stride for conv layer.
        :return: nn.Sequential() of res_blocks.
        """
        identity_downsample = None
        layers = []

        # If the spatial dimension changes or the number of channels in the block changes, then use identity_downsample.
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=(1, 1),
                          stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

        # identity_downsample is used only in the first layer,
        # since this is where the number of channels and spatial dimension (if necessary) change.
        # In other layers, the number of input and output channels remains unchanged.
        layers += [
            res_block(in_channels=self.in_channels, out_channels=out_channels, identity_downsample=identity_downsample,
                      stride=stride)
        ]
        self.in_channels = out_channels

        # Merging the remaining res_blocks.
        for _ in range(num_residual_blocks - 1):
            layers += [
                nn.Sequential(
                    res_block(in_channels=self.in_channels, out_channels=out_channels),
                    nn.BatchNorm2d(num_features=out_channels)
                )
            ]

        return nn.Sequential(*layers)

    def forward(self, X):
        X = self.maxpool(self.relu(self.bn1(self.conv1(X))))

        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)

        X = self.avgpool(X)
        X = nn.Flatten(1)(X)
        return self.fc(X)


def ResNet50(in_channels=3, num_classes=1000, blocks_out_channels=None):
    """
    Function, than returns ResNet50 model.
    :param in_channels: number of channels of the input image.
    :param num_classes: number of classes in the classification task.
    :param blocks_out_channels: list of number of output channels for each block.
    :return: nn.Module() model ResNet50.
    """
    if blocks_out_channels is None:
        blocks_out_channels = [256, 512, 1024, 2048]
    return ResNet(
        layers=[3, 4, 6, 3],
        image_channels=in_channels,
        num_classes=num_classes,
        blocks_out_channels=blocks_out_channels
    )


def ResNet101(in_channels=3, num_classes=1000, blocks_out_channels=None):
    """
        Function, than returns ResNet50 model.
        :param in_channels: number of channels of the input image.
        :param num_classes: number of classes in the classification task.
        :param blocks_out_channels: list of number of output channels for each block.
        :return: nn.Module() model ResNet101.
        """
    if blocks_out_channels is None:
        blocks_out_channels = [256, 512, 1024, 2048]
    return ResNet(
        layers=[3, 4, 23, 3],
        image_channels=in_channels,
        num_classes=num_classes,
        blocks_out_channels=blocks_out_channels
    )


def ResNet152(in_channels=3, num_classes=1000, blocks_out_channels=None):
    """
        Function, than returns ResNet50 model.
        :param in_channels: number of channels of the input image.
        :param num_classes: number of classes in the classification task.
        :param blocks_out_channels: list of number of output channels for each block.
        :return: nn.Module() model ResNet152.
        """
    if blocks_out_channels is None:
        blocks_out_channels = [256, 512, 1024, 2048]
    return ResNet(
        layers=[3, 8, 36, 3],
        image_channels=in_channels,
        num_classes=num_classes,
        blocks_out_channels=blocks_out_channels
    )
