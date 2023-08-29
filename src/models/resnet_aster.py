import torch
import torch.nn as nn

class AsterBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=nn.Identity()):
        super(AsterBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_ASTER(nn.Module):
    def __init__(self, with_lstm=False, n_group=1):
        super(ResNet_ASTER, self).__init__()
        self.with_lstm = with_lstm
        self.n_group = n_group
        self.inplanes = 32
        self.out_planes = 512

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(32,  3, [2, 2])
        self.layer2 = self._make_layer(64,  4, [2, 2])
        self.layer3 = self._make_layer(128, 6, [2, 1])
        self.layer4 = self._make_layer(256, 6, [2, 1])
        self.layer5 = self._make_layer(512, 3, [2, 1])

        # Add LSTM layer if required
        if with_lstm:
            self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)

        # Initialize weights and biases
        self._initialize_weights()

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = [AsterBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend([AsterBlock(self.inplanes, planes) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        cnn_feat = x5.squeeze(2).transpose(2, 1)

        if self.with_lstm:
            return self.rnn(cnn_feat)[0]
        else:
            return cnn_feat
