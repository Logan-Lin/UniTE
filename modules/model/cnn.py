from modules.model.base import *


class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size=2):
        super(DownLayer, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              1, padding='same')
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(pooling_size)

    def forward(self, x):
        """
        Args:
            x (C, H, W)
        """
        return self.pooling(self.relu(self.conv(x)))


class UpLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pooling_size=2):
        super(UpLayer, self).__init__()

        padding = kernel_size // 2
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         1, padding)
        self.relu = nn.ReLU()
        self.pooling = lambda x: F.interpolate(x, scale_factor=pooling_size, mode='nearest')

    def forward(self, x):
        return self.pooling(self.relu(self.conv_t(x)))


class CNNEncoder(Encoder):
    """
    Convolutional Neural Network-based Encoder.
    """
    def __init__(self, num_w, num_h, channels, kernel_sizes, output_size, sampler):
        super(CNNEncoder, self).__init__(sampler, f"CNN-{output_size}-" +
                                         ''.join([str(e) for e in channels]) + '-' +
                                         ''.join([str(e) for e in kernel_sizes]))

        self.output_size = output_size

        self.encoder = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder.append(DownLayer(channels[i], channels[i+1], kernel_sizes[i]))

        self.linear = nn.Linear(num_w // 2**len(kernel_sizes) * num_h // 2**len(kernel_sizes) * channels[-1],
                                output_size)

    def forward(self, x):
        for encoder in self.encoder:
            x = encoder(x)

        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.linear(x)

        return x


class CNNDecoder(Decoder):
    """
    Convolutional Neural Network-based Decoder.
    """
    def __init__(self, num_w, num_h, encode_size, channels, kernel_sizes):
        super().__init__(f"CNN-{encode_size}-" +
                         ''.join([str(e) for e in channels]) + '-' +
                         ''.join([str(e) for e in kernel_sizes]))

        self.linear = nn.Linear(encode_size,
                                num_w // 2**len(kernel_sizes) * num_h // 2**len(kernel_sizes) * channels[0])
        self.init_h = num_h // 2**len(kernel_sizes)
        self.init_w = num_w // 2**len(kernel_sizes)

        self.decoders = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.decoders.append(UpLayer(channels[i], channels[i+1], kernel_sizes[i]))

    def forward(self, x):
        x = self.linear(x)
        x = rearrange(x, 'b (c h w) -> b c h w', h=self.init_h, w=self.init_w)
        for decoder in self.decoders:
            x = decoder(x)

        return x
