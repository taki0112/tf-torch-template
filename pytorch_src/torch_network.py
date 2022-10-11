from torch import nn

class NetModel(nn.Module):
    def __init__(self, input_shape, feature_size):
        super(NetModel, self).__init__()

        self.feature_size = feature_size

        model = []
        model += [nn.Conv2d(in_channels=3, out_channels=self.feature_size, kernel_size=3, stride=2, padding=1, bias=True)]
        model += [nn.ReLU()]
        model += [nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size * 2, kernel_size=3, stride=2, padding=1, bias=True)]
        model += [nn.ReLU()]

        model += [nn.Flatten()]
        linear_dims = (input_shape // 4) * (input_shape // 4)
        model += [nn.Linear(in_features= linear_dims * self.feature_size * 2, out_features=10, bias=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x