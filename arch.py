import functools

from torch import nn

get_norm_layer = {'instance': functools.partial(nn.InstanceNorm2d, affine=False,
                                                track_running_stats=False),
                  'batch': functools.partial(nn.BatchNorm2d, affine=True)}


class convUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ups=False):
        super().__init__()
        block = []
        self.ups = ups
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1)
        )

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 2, 1, 1, bias=False)

    def forward(self, x):
        # a = self.block(x)
        # there i tried use different deconvolutional layers
        if self.ups:
            a = self.block(x)
        else:
            a = self.conv(x)
        return a


class ResidualBlock(nn.Module):
    def __init__(self, d, norm, use_dropout, use_bias, ups=False):
        norm_layer = get_norm_layer[norm]
        super().__init__()
        block = [nn.ReflectionPad2d(1),
                 layer(True, 'relu', d, d, kernel_size=3,
                       norm=norm, bias=use_bias, ups=ups)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [nn.ReflectionPad2d(1),
                  nn.Conv2d(d, d, kernel_size=3, padding=0, bias=use_bias),
                  norm_layer(d)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


def grad(net, state=False):
    for param in net.parameters():
        param.requires_grad = state


def layer(conv, act, in_ch, out_ch, kernel_size, stride=1, padding=0,
          norm='batch', bias=False, output_padding=0, ups=False):
    norm_layer = get_norm_layer[norm]
    layer = []
    if conv:
        layer += [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                            bias=bias)]
    else:
        layer += [convUpsample(in_ch, out_ch, kernel_size, stride, ups=ups)]
    layer += [norm_layer(out_ch)]
    if act == 'relu':
        layer += [nn.ReLU(True)]
    elif act == 'lrelu':
        layer += [nn.LeakyReLU(0.2, True)]
    return nn.Sequential(*layer)
