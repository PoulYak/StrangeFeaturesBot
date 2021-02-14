from torch import nn

from arch import get_norm_layer, layer, ResidualBlock
from utils import init_weights


class Generator(nn.Module):
    def __init__(self, input_ch=3, output_ch=3, ngf=64, norm='batch',
                 use_dropout=True, num_blocks=6, ups=False):
        super().__init__()
        use_bias = not (norm == 'batch')
        model = [
            nn.ReflectionPad2d(3),
            layer(True, 'relu', input_ch, ngf * 1, 7, norm=norm, bias=use_bias, ups=ups),
            layer(True, 'relu', ngf * 1, ngf * 2, 3, 2, 1, norm=norm, bias=use_bias, ups=ups),
            layer(True, 'relu', ngf * 2, ngf * 4, 3, 2, 1, norm=norm, bias=use_bias, ups=ups)
        ]

        for i in range(num_blocks):
            model += [ResidualBlock(ngf * 4, norm, use_dropout, use_bias, ups=ups)]

        model += [
            layer(False, 'relu', ngf * 4, ngf * 2, 3, 2, 1, norm=norm, bias=use_bias, output_padding=1, ups=ups),
            layer(False, 'relu', ngf * 2, ngf * 1, 3, 2, 1, norm=norm, bias=use_bias, output_padding=1, ups=ups),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_ch, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def Gen(input_nc, output_nc, ngf=64, norm='batch', use_dropout=False, is_cuda=True, ups=False):
    gen_net = Generator(input_nc, output_nc, ngf, norm=norm,
                        use_dropout=use_dropout, num_blocks=9, ups=ups)
    if is_cuda:
        gen_net.cuda()
    else:
        gen_net.cpu()
    gen_net = nn.DataParallel(gen_net)

    init_weights(gen_net)
    return gen_net


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm='batch', ups=False):
        super().__init__()
        use_bias = not (norm == 'batch')
        model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, True)]
        model += [layer(True, 'lrelu', ndf, ndf * 2, kernel_size=4, stride=2,
                        norm=norm, padding=1, bias=use_bias, ups=ups)]
        model += [layer(True, 'lrelu', ndf * 2, ndf * 4, kernel_size=4, stride=2,
                        norm=norm, padding=1, bias=use_bias, ups=ups)]
        model += [layer(True, 'lrelu', ndf * 4, ndf * 8, kernel_size=4, stride=1,
                        norm=norm, padding=1, bias=use_bias, ups=ups)]
        model += [nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


def Dis(input_nc, ndf=64, n_layers_D=3, norm='batch', ups=False):
    dis_net = Discriminator(input_nc, ndf, norm=norm, ups=ups)
    dis_net.cuda()
    dis_net = nn.DataParallel(dis_net)

    init_weights(dis_net)
    return dis_net
