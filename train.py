import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from model import DecomNet
from dataset import ImageDataset
from loss import DecomLoss
import tqdm

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='RetinexNet args setting')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default='0', help='GPU idx')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--workers', dest='workers', type=int, default=8, help='num workers of dataloader')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--save_interval', dest='save_interval', type=int, default=20, help='save model every # epoch')

parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/low', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=0,
                    help='decom flag, 0 for enhanced results only and 1 for decomposition results')

parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--route', type=str, default='./data', help='root directory of the dataset')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

args = parser.parse_args()

if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

decom_net = DecomNet()


if args.use_gpu:
    decom_net = decom_net.cuda()

    cudnn.benchmark = True
    cudnn.enabled = True

lr = args.start_lr * np.ones([args.epoch])
lr[20:] = lr[0] / 10.0

decom_optim = torch.optim.Adam(decom_net.parameters(), lr=args.start_lr)




decom_criterion = DecomLoss()


transforms_ = [ transforms.Resize(int(args.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(args.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


def train():
    decom_net.train()


    for epoch in range(args.epoch):
        times_per_epoch, sum_loss = 0, 0.

        # dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
        #                                          num_workers=args.workers, pin_memory=True)
        dataloader = DataLoader(ImageDataset(args.route, transforms_=transforms_, unaligned=True), 
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers) 
        decom_optim.param_groups[0]['lr'] = lr[epoch]

        for data in tqdm.tqdm(dataloader):
            times_per_epoch += 1
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()
  
           
            _, r_low, l_low = decom_net(low_im)
            _, r_high, l_high = decom_net(high_im)
            loss = decom_criterion(r_low, l_low, r_high, l_high, low_im, high_im)
            decom_optim.zero_grad()
            loss.backward()
            decom_optim.step()

            sum_loss += loss
  
        print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch ))
        if (epoch+1) % args.save_interval == 0:
            torch.save(decom_net.state_dict(), args.ckpt_dir + '/decom_' + str(epoch) + '.pth')

    torch.save(decom_net.state_dict(), args.ckpt_dir + '/decom_final.pth')

    # for epoch in range(args.epoch):
    #     times_per_epoch, sum_loss = 0, 0.

    #     dataloader = torch.utils.data.Dataloader(train_set, batch_size=args.batch_size, shuffle=True,
    #                                              num_workers=args.workers, pin_memory=True)
    #     relight_optim.param_groups[0]['lr'] = lr[epoch]

    #     for data in tqdm.tqdm(dataloader):
    #         times_per_epoch += 1
    #         low_im, high_im = data
    #         low_im, high_im = low_im.cuda(), high_im.cuda()

    #         relight_optim.zero_grad()
    #         lr_low, r_low, _ = decom_net(low_im)
    #         l_delta = relight_net(lr_low.detach())
    #         loss = relight_criterion(l_delta, r_low.detach(), high_im)
    #         loss.backward()
    #         relight_optim.step()

    #         sum_loss += loss

    #     print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch))
    #     if (epoch+1) % args.save_interval == 0:
    #         torch.save(relight_net.state_dict(), args.ckpt_dir + '/relight_' + str(epoch) + '.pth')

    # torch.save(relight_net.state_dict(), args.ckpt_dir + '/relight_final.pth')


if __name__ == '__main__':
    train()
