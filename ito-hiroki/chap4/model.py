from torch import nn


class LeNet(nn.Module):
    """
    メモ: オリジナル実装の活性化関数はsigmoid squashing function
    f(a) = Atanh(Sa), AとSはそれぞれ係数
    学習も損失関数はMSEでsoftmaxは使われていなかったっぽい
    ref: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    """

    def __init__(self, in_channel=1, out_channel=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, kernel_size=5, stride=1, padding=2)
        self.sigmoid1 = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        # 別に上のsigmoidを使いまわしても良いはず
        self.sigmoid2 = nn.Sigmoid()
        self.pool2 = nn.AvgPool2d(2, 2, 0)

        self.flatten = nn.Flatten()
        self.fcn1 = nn.Linear(400, 120)
        self.sigmoid3 = nn.Sigmoid()
        self.fcn2 = nn.Linear(120, 84)
        self.sigmoid4 = nn.Sigmoid()
        self.fcn3 = nn.Linear(84, out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = self.sigmoid1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.sigmoid2(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.fcn1(out)
        return out


class AlexNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, 2, 0)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3, 2, 0)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, 2, 0)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(6400, 4096)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout()

        self.linear2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout()

        self.linear3 = nn.Linear(4096, out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.relu4(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.pool3(out)

        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu6(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.relu7(out)
        out = self.dropout2(out)

        out = self.linear3(out)
        return out


class SmallAlexNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(SmallAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, 2, 0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3, 2, 0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, 2, 0)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(1600, 1024)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout()

        self.linear2 = nn.Linear(1024, 1024)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout()

        self.linear3 = nn.Linear(1024, out_channel)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.relu4(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.pool3(out)

        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu6(out)
        out = self.dropout1(out)

        out = self.linear2(out)
        out = self.relu7(out)
        out = self.dropout2(out)

        out = self.linear3(out)
        return out


class VGG11(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(VGG11, self).__init__()
        self.conv1 = self.conv_relu(in_channel, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = self.conv_relu(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = self.conv_relu(128, 256)
        self.conv4 = self.conv_relu(256, 256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = self.conv_relu(256, 512)
        self.conv6 = self.conv_relu(512, 512)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = self.conv_relu(512, 512)
        self.conv8 = self.conv_relu(512, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(4608, 4096)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(4096, out_channel)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)

        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool3(out)

        out = self.conv7(out)
        out = self.conv8(out)
        out = self.pool4(out)

        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out

    @staticmethod
    def conv_relu(in_channel, out_channel):
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, 3), nn.ReLU())


class SmallVGG11(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(SmallVGG11, self).__init__()
        self.conv1 = self.conv_relu(in_channel, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = self.conv_relu(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = self.conv_relu(32, 64)
        self.conv4 = self.conv_relu(64, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = self.conv_relu(64, 128)
        self.conv6 = self.conv_relu(128, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = self.conv_relu(128, 128)
        self.conv8 = self.conv_relu(128, 128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(1152, 1024)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(1024, out_channel)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool2(out)

        out = self.conv5(out)
        out = self.conv6(out)
        out = self.pool3(out)

        out = self.conv7(out)
        out = self.conv8(out)
        out = self.pool4(out)

        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        return out

    @staticmethod
    def conv_relu(in_channel, out_channel):
        return nn.Sequential(nn.Conv2d(in_channel, out_channel, 3), nn.ReLU())
