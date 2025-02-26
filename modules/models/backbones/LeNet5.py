from torch import nn

from activation_function import BuildActivation


class LeNet5(nn.Module):
    def __init__(self, in_channels, num_classes, act_cfg):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc1 = nn.Linear(in_features=84, out_features=num_classes)
        self.relu = BuildActivation(act_cfg)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = self.fc3(x)
        return x



if __name__ == '__main__':
    model = LeNet5(1, 10, act_cfg={'type': 'ReLU', 'inplace': True})
    print(model)