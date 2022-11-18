import torch.nn as nn
import torch.nn.functional as F
import torch

class B_C2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, pooling=False):
        super(B_C2d, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=False)
        self.pooling = pooling
        if self.pooling:
            self.P = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = F.leaky_relu(self.conv(x), inplace=True)
        if self.pooling:
            x = self.P(x)
        return x


class CNN(nn.Module):
    def __init__(self, class_num=100):
        super(CNN, self).__init__()
        init_c = [32, 64, 128, 256]

        self.conv = nn.Sequential(
            B_C2d(1, init_c[0], 3, 1, False),
            B_C2d(init_c[0], init_c[0], 3, 1, True),
            B_C2d(init_c[0], init_c[1], 3, 1, False),
            B_C2d(init_c[1], init_c[1], 3, 1, True),
            B_C2d(init_c[1], init_c[2], 3, 1, False),
            B_C2d(init_c[2], init_c[2], 3, 1, True),
            B_C2d(init_c[2], init_c[3], 3, 1, False),
            B_C2d(init_c[3], init_c[3], 3, 1, True)
        )
        self.FC = nn.Linear(init_c[3], class_num)
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(-1).mean(-1) + x.max(-1)[0].max(-1)[0]
        x = self.FC(x)
        return x

if __name__ == '__main__':

    x = torch.randn(8, 1, 256, 256)
    net = CNN()
    y = net(x)
    print(y.size())