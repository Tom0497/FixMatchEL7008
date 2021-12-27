import torch.nn as nn


class WideDropoutBlock(nn.Module):
    """
    Class representing a basic block for Wide ResNet.

    In ResNet there were two types of basic blocks, called basic
    and bottleneck. In <https://arxiv.org/pdf/1605.07146.pdf> the
    authors define two more called basic-wide and wide-dropout.

    This class represents a wide-dropout block and can easily
    emulate a basic-wide block using `dropout=0`.

    This block is the basic unit to construct a WideResNetBlock,
    and its structure is determined by the `in_planes` and
    `out_planes`, dropout probability and whether it performs
    down sampling.
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 dropout: float,
                 down_sample: bool):
        """
        Constructor of WideDropoutBlock.

        :param in_planes:
            number of channels of wide dropout block input.
        :param out_planes:
            number of channels of wide dropout block output.
        :param dropout:
            dropout probability for model training.
        :param down_sample:
            whether the wide dropout block performs down sampling.
        """

        super(WideDropoutBlock, self).__init__()

        # block structure parameters
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.dropout_prob = dropout
        self.down_sample = down_sample
        self.stride = 2 if self.down_sample else 1

        # block layers BN->RELU->CONV->DROPOUT->BN->RELU->CONV
        self.bn1 = nn.BatchNorm2d(num_features=in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.in_planes,
                               out_channels=self.out_planes,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=(self.stride, self.stride))
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.bn2 = nn.BatchNorm2d(num_features=self.out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=self.out_planes,
                               out_channels=self.out_planes,
                               kernel_size=(3, 3),
                               padding=1,  # 3//2 = 1
                               stride=(1, 1))

        # skip-connection parallel to previous block
        self.skip_connection = self.__resolve_skip_connection()

    def forward(self, x):
        """
        Forward pass through WideDropout block.

        :param x:
            input matrix, usually an image.
        :return:
            WideDropout block output.
        """

        # first stage
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        # dropout layer
        out = self.dropout(out)

        # second stage
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        # adding of skip-connection
        out += self.skip_connection(x)

        return out

    def __resolve_skip_connection(self):
        """
        Determine whether a layer is needed for skip-connection.

        :return:
            layer required for skip-connection.
        """

        # down sampling and changing number of channels require adaptation
        if self.stride != 1 or self.in_planes != self.out_planes:
            return nn.Conv2d(in_channels=self.in_planes,
                             out_channels=self.out_planes,
                             kernel_size=(3, 3),
                             padding=1,  # 3//2 = 1
                             stride=(self.stride, self.stride))
        # otherwise, identity layer is used
        return nn.Identity()


class WideResNetBlock(nn.Module):
    """
    Class representing main block of Wide ResNets.

    As shown in the paper <https://arxiv.org/pdf/1605.07146.pdf>,
    Wide ResNets has four main convolutional blocks, named convx,
    where x goes from 1 to 4. Blocks 2 to 4 have very similar
    structure, therefore, this class seeks to represents it while
    giving all variants required through initialization parameters.

    A block is defined by its `in_planes` and `out_planes`. And
    most important, whether it performs down sampling.
    """

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 depth: int,
                 dropout: float,
                 down_sample: bool):
        """
        Constructor of WideResNetBlock.

        :param in_planes:
            number of channels of block input.
        :param out_planes:
            number of channels of block output.
        :param depth:
            number of wide dropout blocks.
        :param dropout:
            dropout probability for model training.
        :param down_sample:
            whether the block performs down sampling.
        """

        super(WideResNetBlock, self).__init__()

        # block structure parameters
        self.N = depth
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.dropout_prob = dropout
        self.down_sample = down_sample

        # first wide-dropout block takes in in_planes and outputs out_planes
        self.wide_dropouts_blocks = [WideDropoutBlock(in_planes=self.in_planes,
                                                      out_planes=self.out_planes,
                                                      dropout=self.dropout_prob,
                                                      down_sample=self.down_sample)]

        # all other wide-dropout blocks have out_planes as input and output
        for _ in range(self.N - 1):
            self.wide_dropouts_blocks.append(WideDropoutBlock(in_planes=self.out_planes,
                                                              out_planes=self.out_planes,
                                                              dropout=self.dropout_prob,
                                                              down_sample=False))

        # layers combined in nn.Sequential
        self.wide_resnet_block = nn.Sequential(*self.wide_dropouts_blocks)

    def forward(self, x):
        """
        Forward pass through Wide ResNet block.

        :param x:
            input matrix, generally an image.
        :return:
            block output after every layer and non-linearity.
        """

        out = self.wide_resnet_block(x)
        return out


class WideResNet(nn.Module):
    r"""
    Class representing Wide Residual Network.

    Wide ResNet or Wide Residual Networks, correspond to a variant of
    Residual Networks or ResNet, and build up from them. ResNet was a
    pioneer work showing that with the concept of skip-connections it's
    possible to build networks up to thousands of layers, resolving the
    vanishing gradient problem to some extent. Although it was a great
    improvement, ResNet require of doubling the number of layers to gain
    a fraction of a percentage in accuracy.

    The work of Zagoruyko and komodakis in `"Wide Residual Networks"`
    <https://arxiv.org/pdf/1605.07146.pdf> studied and proposed a novel
    architecture that decrease depth and increase width, hence the name.

    These nets reduce considerably the number of parameters per model,
    and obtain better results than ResNet. A Wide ResNet is described
    by the number of layers it has, the width of every convolutional
    block, and the number of classes it classifies.
    """

    def __init__(self,
                 depth: int,
                 width: int,
                 n_classes: int,
                 dropout: float = .5):
        """
        Constructor of WideResNet model.

        :param depth:
            number of layers of the model (ideally 6*N+4 = depth).
        :param width:
            network width or channel multiplication factor.
        :param n_classes:
            number of classes the model is going to classify.
        :param dropout:
            dropout probability for model training.
        """

        super(WideResNet, self).__init__()

        # model structure parameters
        self.n = depth
        self.k = width
        self.n_classes = n_classes
        self.dropout_prob = dropout

        # number of WideDropoutBlocks per WideResNetBlock
        self.N = int((self.n - 4) / 6)

        # model blocks
        # convolutional stage
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=(3, 3),
                               padding='same',
                               stride=(1, 1))
        self.conv2 = WideResNetBlock(in_planes=16,
                                     out_planes=16 * self.k,
                                     depth=self.N,
                                     dropout=self.dropout_prob,
                                     down_sample=False)
        self.conv3 = WideResNetBlock(in_planes=16 * self.k,
                                     out_planes=32 * self.k,
                                     depth=self.N,
                                     dropout=self.dropout_prob,
                                     down_sample=True)
        self.conv4 = WideResNetBlock(in_planes=32 * self.k,
                                     out_planes=64 * self.k,
                                     depth=self.N,
                                     dropout=self.dropout_prob,
                                     down_sample=True)
        # pooling stage
        self.bn = nn.BatchNorm2d(num_features=64 * self.k)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=8)

        # probabilities estimation
        self.linear = nn.Linear(in_features=64 * self.k,
                                out_features=self.n_classes)

    def forward(self, x):
        """
        Forward an input through the model applying layers.

        :param x:
            input matrix, generally an image.
        :return:
            model output after every layer and non-linearity.
        """

        # convolutional stage
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # pooling stage
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)

        # probabilities estimation
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def num_parameters(self) -> int:
        """
        :return:
            number of parameters the model has.
        """

        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = WideResNet(depth=28, width=2, n_classes=10)
    print('Cantidad de parámetros del modelo: ', model.num_parameters())
