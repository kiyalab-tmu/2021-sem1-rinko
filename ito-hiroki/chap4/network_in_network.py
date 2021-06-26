import torch
import torch.nn as nn


class NiN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(NiN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, (11, 11), (4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(64, 128, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channel, (1, 1)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((6, 6), (1, 1), (0, 0), ceil_mode=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)


# class NiN(nn.Module):
#     def __init__(self, in_channel=1, out_channel=10):
#         super(NiN, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Conv2d(in_channel, 192, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Dropout(0.5),
#             nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Dropout(0.5),
#             nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, out_channel, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),
#             # nn.Flatten(),
#         )

#     def forward(self, x):
#         x = self.classifier(x)
#         x = x.view(x.size(0), 10)
#         return x


if __name__ == "__main__":
    inputs = torch.randn((3, 1, 224, 224))
    model = NiN()
