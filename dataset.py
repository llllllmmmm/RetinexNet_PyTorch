#cycleGAN
import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch.utils.data as data
from utils import load_images, data_augmentation


class ImageDataset(Dataset):
    def __init__(self, route='./data/', phase='train', patch_size=48, transforms_=None, unaligned=False, mode='train'):
        self.route = route
        self.phase = phase
        self.patch_size = patch_size

        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_low = sorted(glob.glob(os.path.join(route, '%s/low' % phase) + '/*.*'))
        self.files_high = sorted(glob.glob(os.path.join(route, '%s/high' % phase) + '/*.*'))
        #(os.path.join(route, '%s/high' % phase) + '/*.*')
       
    def __getitem__(self, index):
        item_low = self.transform(Image.open(self.files_low[index % len(self.files_low)]))

        if self.unaligned:
            item_high = self.transform(Image.open(self.files_high[random.randint(0, len(self.files_high) - 1)]))
        else:
            item_high = self.transform(Image.open(self.files_high[index % len(self.files_high)]))
        # print(item_high.shape,item_low.shape)
        return  item_low, item_high

    def __len__(self):
        return max(len(self.files_low), len(self.files_high))






# import random
# import torch.utils.data as data
# from glob import glob
# from utils import load_images, data_augmentation

# from torch.utils.data import Dataset

# # folder structure
# # data
# #   |train
# #   |  |low
# #   |  |  |*.png
# #   |  |high
# #   |  |  |*.png
# #   |eval
# #   |  |low
# #   |  |  |*.png
# def get_dataset_len(route, phase):
#     if phase == 'train':
#         low_data_names = glob(route + phase + '/low/*.png')
#         high_data_names = glob(route + phase + '/high/*.png')
#         low_data_names.sort()
#         high_data_names.sort()
#         assert len(low_data_names) == len(high_data_names)
#         return len(low_data_names), [low_data_names, high_data_names]
#     elif phase == 'eval':
#         low_data_names = glob(route + phase + '/low/*.png')
#         return len(low_data_names), low_data_names
#     else:
#         return 0, []


# def getitem(phase, data_names, item, patch_size):
#     if phase == 'train':
#         low_im = load_images(data_names[0][item])
#         high_im = load_images(data_names[1][item])

#         h, w, _ = low_im.shape
#         x = random.randint(0, h - patch_size)
#         y = random.randint(0, w - patch_size)
#         rand_mode = random.randint(0, 7)

#         low_im = data_augmentation(low_im[x:x + patch_size, y:y + patch_size, :], rand_mode)
#         high_im = data_augmentation(high_im[x:x + patch_size, y:y + patch_size, :], rand_mode)

#         return low_im, high_im
#     elif phase == 'eval':
#         low_im = load_images(data_names[item])
#         return low_im


# class TheDataset(Dataset):

#     def __init__(self, route='./data/', phase='train', patch_size=48):
#         self.route = route
#         self.phase = phase
#         self.patch_size = patch_size
#         self.len, self.data_names = get_dataset_len(route, phase)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, item):
#         return getitem(self.phase, self.data_names, item, self.patch_size)


