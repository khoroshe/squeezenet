import torch
import torch.nn
import torch.nn.functional


class FireModule(torch.nn.Module):

    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()

        self.s1x1 = torch.nn.Conv2d(in_channels, s1x1, 1)
        self.e1x1 = torch.nn.Conv2d(s1x1, e1x1, 1)
        self.e3x3 = torch.nn.Conv2d(s1x1, e3x3, 3, padding=1)

    def forward(self, x):
        x = self.s1x1(x)  # squeeze
        x = torch.nn.functional.relu(x)  # squeeze ReLU

        x1 = self.e1x1(x)  # expand 1x1
        x2 = self.e3x3(x)  # expand 3x3
        out = torch.cat((x1, x2), dim=1)  # concat expand
        out = torch.nn.functional.relu(out)  # expand ReLU

        return out


class SqueezeNet(torch.nn.Module):

    def __init__(self, input_shape = 224, type="vanilla", num_classes=10):
        super(SqueezeNet, self).__init__()

        assert type == "vanilla" or type == "simple_bypass" or type == "complex_bypass", \
            "SqueezeNet type error, should be either 'vanilla', 'simple_bypass' or 'complex_bypass'"

        self.type = type

        self.conv1 = torch.nn.Conv2d(3, 96, 7, stride=2, padding=2)
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2)
        self.fire2 = FireModule(96, 16, 64, 64)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.maxpool4 = torch.nn.MaxPool2d(3, stride=2)

        self.fire5 = FireModule(256, 32, 128, 128)
        self.fire6 = FireModule(256, 48, 192, 192)
        self.fire7 = FireModule(384, 48, 192, 192)
        self.fire8 = FireModule(384, 64, 256, 256)

        self.maxpool8 = torch.nn.MaxPool2d(3, stride=2)

        self.fire9 = FireModule(512, 64, 256, 256)

        self.dropout = torch.nn.Dropout(p=0.5)

        self.conv10 = torch.nn.Conv2d(512, 1000, 1)

        if input_shape == 224:
            self.avgpool10 = torch.nn.AvgPool2d(13, stride=1)
        elif input_shape == 32:
            self.avgpool10 = torch.nn.AvgPool2d(1, stride=1)

        self.fc = torch.nn.Linear(in_features=1000, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)

        x = self.maxpool4(x)

        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)

        x = self.maxpool8(x)

        x = self.fire9(x)

        x = self.dropout(x)

        x = self.conv10(x)
        x = self.avgpool10(x)

        x = torch.squeeze(x)

        x = self.fc(x)

        return x
