from torch import nn


class Predictor(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.name = name


class FCPredictor(Predictor):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__(f'fc-h{hidden_size}')

        # self.linear = nn.Linear(input_size, output_size)
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True),
                                    nn.Linear(hidden_size, output_size))

    def forward(self, h):
        y = self.linear(h)
        return y


class NonePredictor(Predictor):
    def __init__(self, **kwargs):
        super().__init__('none')
