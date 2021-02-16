import itertools
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from arch import grad
from models import Gen, Dis
from utils import LambdaLR, Dataset, Sample_from_Pool

'''
Class for CycleGAN with train() as a member function
'''


class CycleGAN(object):
    def __init__(self, dataset_name, ups=False):
        self.dataset_name = dataset_name
        self.checkpoint_dir = 'checkpoints/' + self.dataset_name
        lr = 0.0002
        use_dropout = True
        norm = 'instance'
        epochs = 1000
        self.epochs = epochs
        decay_epoch = 2
        self.gen_ab = Gen(input_nc=3, output_nc=3, norm=norm,
                          use_dropout=use_dropout, ups=ups)
        self.gen_ba = Gen(input_nc=3, output_nc=3, norm=norm,
                          use_dropout=use_dropout, ups=ups)
        self.dis_a = Dis(input_nc=3, norm=norm, ups=ups)
        self.dis_b = Dis(input_nc=3, norm=norm, ups=ups)

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        self.g_optimizer = torch.optim.Adam(itertools.chain(self.gen_ab.parameters(), self.gen_ba.parameters()), lr=lr,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.dis_a.parameters(), self.dis_b.parameters()), lr=lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=LambdaLR(epochs, 0, decay_epoch).step)

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        try:
            ckpt = torch.load('%s/last' % self.checkpoint_dir)
            self.start_epoch = ckpt['epoch']
            self.dis_a.load_state_dict(ckpt['dis_a'])
            self.dis_b.load_state_dict(ckpt['dis_b'])
            self.gen_ab.load_state_dict(ckpt['gen_ab'])
            self.gen_ba.load_state_dict(ckpt['gen_ba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print('No checkpoint!')
            self.start_epoch = 0

    def train(self):
        dataset_path = 'datasets/' + self.dataset_name
        train_a = Dataset(os.path.join(dataset_path, 'trainA'))
        train_b = Dataset(os.path.join(dataset_path, 'trainB'))
        lamda = 10
        idt_coef = 0.5
        batch_size = 1

        a_loader = DataLoader(train_a, batch_size=batch_size, shuffle=True, num_workers=4)
        b_loader = DataLoader(train_b, batch_size=batch_size, shuffle=True, num_workers=4)

        a_fake_sample = Sample_from_Pool()
        b_fake_sample = Sample_from_Pool()

        for epoch in range(self.start_epoch, self.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('lr = %.7f' % lr)

            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                grad(self.dis_a, False)
                grad(self.dis_b, False)

                self.g_optimizer.zero_grad()

                a_real = Variable(a_real)
                b_real = Variable(b_real)
                a_real, b_real = a_real.cuda(), b_real.cuda()

                a_fake = self.gen_ab(b_real)
                b_fake = self.gen_ba(a_real)

                a_recon = self.gen_ab(b_fake)
                b_recon = self.gen_ba(a_fake)

                a_idt = self.gen_ab(a_real)
                b_idt = self.gen_ba(b_real)

                a_idt_loss = self.L1(a_idt, a_real) * lamda * idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * lamda * idt_coef

                a_fake_dis = self.dis_a(a_fake)
                b_fake_dis = self.dis_b(b_fake)

                real_label = Variable(torch.ones(a_fake_dis.size())).cuda()

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                a_cycle_loss = self.L1(a_recon, a_real) * lamda
                b_cycle_loss = self.L1(b_recon, b_real) * lamda

                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                gen_loss.backward()
                self.g_optimizer.step()

                grad(self.dis_a, True)
                grad(self.dis_b, True)

                self.d_optimizer.zero_grad()

                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = a_fake.cuda(), b_fake.cuda()

                a_real_dis = self.dis_a(a_real)
                a_fake_dis = self.dis_a(a_fake)
                b_real_dis = self.dis_b(b_real)
                b_fake_dis = self.dis_b(b_fake)
                real_label = Variable(torch.ones(a_real_dis.size())).cuda()
                fake_label = Variable(torch.zeros(a_fake_dis.size())).cuda()

                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5

                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            torch.save(
                {'epoch': epoch + 1,
                 'dis_a': self.dis_a.state_dict(),
                 'dis_b': self.dis_b.state_dict(),
                 'gen_ab': self.gen_ab.state_dict(),
                 'gen_ba': self.gen_ba.state_dict(),
                 'd_optimizer': self.d_optimizer.state_dict(),
                 'g_optimizer': self.g_optimizer.state_dict()
                 },
                '%s/last' % self.checkpoint_dir
            )

            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()


def test(iters, dataname, results_dir='results', batch_size=1):
    dir_dataset, dir_checkpoints = 'datasets/' + dataname, 'checkpoints/' + dataname
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    testA = Dataset(os.path.join(dir_dataset, 'testA'))
    testB = Dataset(os.path.join(dir_dataset, 'testB'))

    use_dropout = True
    norm = 'instance'
    a_test_loader = DataLoader(testA, batch_size=batch_size, shuffle=True, num_workers=4)
    b_test_loader = DataLoader(testB, batch_size=batch_size, shuffle=True, num_workers=4)

    gen_ab = Gen(input_nc=3, output_nc=3, norm=norm,
                 use_dropout=use_dropout)
    gen_ba = Gen(input_nc=3, output_nc=3, norm=norm,
                 use_dropout=use_dropout)

    print('%s/last' % dir_checkpoints)
    checkpoint = torch.load('%s/last' % dir_checkpoints)

    print(checkpoint['gen_ab'].keys())

    gen_ab.load_state_dict(checkpoint['gen_ab'])
    gen_ba.load_state_dict(checkpoint['gen_ba'])

    gen_ab.eval()
    gen_ba.eval()

    for i in range(iters):
        a_real_test = Variable(iter(a_test_loader).next(), True)
        b_real_test = Variable(iter(b_test_loader).next(), True)
        a_real_test, b_real_test = a_real_test.cuda(), b_real_test.cuda()

        with torch.no_grad():
            a_fake_test = gen_ab(b_real_test)
            b_fake_test = gen_ba(a_real_test)
            a_recon_test = gen_ab(b_fake_test)
            b_recon_test = gen_ba(a_fake_test)

        pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test],
                         dim=0).data + 1) / 2.0

        torchvision.utils.save_image(pic, results_dir + '/image' + str(i) + '.jpg', nrow=3)


def get_image(path, dataname, id, orint='ba', results_dir='results', batch_size=1, ups=False):
    dir_dataset, dir_checkpoints = 'results/' + id, 'checkpoints/' + dataname
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    x = Image.open(path)
    x = x.convert('RGB')
    height, width = x.size

    x = transform(x)
    x = x[np.newaxis, :]
    use_dropout = True
    norm = 'instance'

    gen = Gen(input_nc=3, output_nc=3, norm=norm,
              use_dropout=use_dropout, ups=ups, is_cuda=False)

    checkpoint = torch.load('%s/last' % dir_checkpoints, map_location='cpu')
    gen.load_state_dict(checkpoint['gen_' + orint])

    gen.eval()

    x = Variable(x, True)
    x = x.cpu()
    with torch.no_grad():
        x = gen(x)
    pic = ((x).data + 1) / 2.0
    if height > width:
        height = int(256 * height / width)
        width = 256
    else:
        width = int(256 * width / height)
        height = 256

    pic = transforms.Resize([width, height])(pic)
    torchvision.utils.save_image(pic, 'results/' + id + '/imag.jpg')
