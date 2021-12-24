import torch.nn as nn


class WideDropoutBlock(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 dropout: float,
                 down_sample: bool):
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
        if self.stride != 1 or self.in_planes != self.out_planes:
            return nn.Conv2d(in_channels=self.in_planes,
                             out_channels=self.out_planes,
                             kernel_size=(3, 3),
                             padding=1,  # 3//2 = 1
                             stride=(self.stride, self.stride))
        return nn.Identity()


class WideResNetBlock(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 depth: int,
                 dropout: float,
                 down_sample: bool):
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

        for _ in range(self.N - 1):
            self.wide_dropouts_blocks.append(WideDropoutBlock(in_planes=self.out_planes,
                                                              out_planes=self.out_planes,
                                                              dropout=self.dropout_prob,
                                                              down_sample=False))

        self.wide_resnet_block = nn.Sequential(*self.wide_dropouts_blocks)

    def forward(self, x):
        out = self.wide_resnet_block(x)
        return out


class WideResnet(nn.Module):

    def __init__(self,
                 depth: int,
                 width: int,
                 n_classes: int,
                 dropout: float = .5):
        super(WideResnet, self).__init__()

        # model structure parameters
        self.n = depth
        self.k = width
        self.n_classes = n_classes
        self.dropout_prob = dropout

        # number of WideDropoutBlocks per WideResNetBlock
        self.N = int((self.n - 4) / 6)

        # model blocks
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
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.bn = nn.BatchNorm2d(num_features=64 * self.k)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=64 * self.k,
                                out_features=self.n_classes)

    def forward(self, x):
        # convolutional stage
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # pooling stage
        out = self.avg_pool(out)

        # prediction stage
        out = self.bn(out)
        out = self.relu(out)
        out = self.linear(out)

        return out

    def num_parameters(self) -> int:
        """
        :return: number of parameters the model has
        """
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = WideResnet(depth=28, width=2, n_classes=10)
    print('Cantidad de parámetros del modelo: ', model.num_parameters())
