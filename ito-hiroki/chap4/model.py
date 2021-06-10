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
        
