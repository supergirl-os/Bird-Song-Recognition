# VGG19.py
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=True)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()
        nn.init.kaiming_normal_(self.conv.weight,mode='fan_out',nonlinearity='relu')




    def forward(self, x):
        # print(x.size(),type(x))
        return self.act(self.conv(x))

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.stages = nn.Sequential(*[
            self._make_stage(1, 64, num_blocks=1, max_pooling=True),
            self._make_stage(64, 128, num_blocks=1, max_pooling=True),
            self._make_stage(128, 256, num_blocks=1, max_pooling=True),
            self._make_stage(256, 512, num_blocks=1, max_pooling=True),
            self._make_stage(512, 512, num_blocks=1, max_pooling=True)
        ])
        self.head = nn.Sequential(*[
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(512 * 8 * 9, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        ])
        for layer in self.head:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias,0)

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, max_pooling):
        layers = [Conv(in_channels, out_channels, kernel_size=3, stride=1)]
        for _ in range(1, num_blocks):
            layers.append(Conv(out_channels, out_channels, kernel_size=3, stride=1))
        if max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.stages(x)
        return self.head(x)

# if __name__ == "__main__":
#     inputs = torch.rand((8, 1, 256, 313)).cuda()
#     model = VGG19(num_classes=100).cuda().train()
#     outputs = model(inputs)
#     print(outputs.shape)
