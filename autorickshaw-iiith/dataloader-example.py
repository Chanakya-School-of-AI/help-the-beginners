#author Vipul Vaibhaw
#CSOAI

# reference - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import torch
from torch.utils import data
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import os
import json
import utils as file_paths_for_demo

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# CUDA for PyTorch
use_cuda = torch.cuda.is_available() #checking if gpu available
print(torch.cuda.get_device_name())
device = torch.device("cuda:0" if use_cuda else "cpu") #if no gpu then use cpu

class AutoRickShawDataset(Dataset):
    """Auto Rickshaw dataset."""

    def __init__(self, bboxes_file, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.bbs = json.loads(open(os.path.join(self.root_dir,bboxes_file)).read())
        self.transform = transform #ignore for now

    def __len__(self):
        # returns the size of the dataset
        return len(self.bbs) 

    def __getitem__(self, idx):
        # to support the indexing such that dataset[i] can be used to get ith sample
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                "images/"+str(idx)+".jpg")
        image = io.imread(img_name)
        bbs = self.bbs[idx]
        bbs = np.array([bbs])
        bbs = bbs.astype('float')
        sample = {'image': image, 'bbs': bbs}

        if self.transform:
            sample = self.transform(sample)

        return sample

auto_dataset = AutoRickShawDataset(bboxes_file=file_paths_for_demo.root_dir+file_paths_for_demo.labels,
                                root_dir=file_paths_for_demo.root_dir)

for i in range(len(auto_dataset)):
    sample = auto_dataset[i]
    print(i, sample['image'].shape, sample['bbs'])

    for j in range(len(sample['bbs'])):
        x, y = sample['bbs'][j][0]
        h = sample['bbs'][j][-1][1] - y  
        w = sample['bbs'][j][1][0] - x 

        print(x,y,w,h)

    # Ideally no break here, we put model training logic here
    if i==3:
        break