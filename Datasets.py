import os
import shutil
import random

import matplotlib.pyplot as plt
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class myDatasets(Dataset):
	# 需包括 __init__, __len__和__getitem__
	def __init__(self, img_dir, mode, transform=None, target_transform=None):
		super().__init__()
		import scipy.io as scio
		self.data = scio.loadmat(img_dir)
		self.mode = mode  # "train/test"
		self.labels = [int(key[-1]) for key, values in self.data.items() for _ in range(len(values)) if
					   key.startswith(self.mode)]
		self.imgs = [value for key, values in self.data.items() for value in values if key.startswith(self.mode)]
		self.transform = transform
		self.target_transform = target_transform
	
	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, item):
		# 返回PIL格式图片
		img = self.imgs[item].reshape((28, 28))
		label = self.labels[item]
		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			label = self.target_transform(label)
		return img, label


"""train_data = myDatasets(r"./mydata/mnist_all.mat", mode="train", transform=Compose([ToTensor(), Resize((224, 224))]))
val_data = myDatasets(r"./mydata/mnist_all.mat", mode="test", transform=Compose([ToTensor(), Resize((224, 224))]))
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=512, shuffle=False, num_workers=0)"""

