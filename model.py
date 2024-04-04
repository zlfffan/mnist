import math
import torch.nn as nn
import torch


class my_model(nn.Module):
	def __init__(self, out_channel):
		assert out_channel % 4 == 0, "output_channel can't be divided by 4"
		super().__init__()
		self.convblock1 = nn.Sequential(
			nn.Conv2d(1, int(out_channel / 2), kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(int(out_channel / 2)),
			nn.ReLU(True),
			nn.Conv2d(int(out_channel / 2), int(out_channel / 2), kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(int(out_channel / 2)),
			nn.ReLU(True)
		)
		self.short1 = nn.Sequential(
			nn.Conv2d(1, int(out_channel / 2), 1, bias=False),
			nn.BatchNorm2d(int(out_channel / 2))
		)
		self.maxpool = nn.MaxPool2d((3, 3), padding=1, stride=2)
		self.convblock2 = nn.Sequential(
			nn.Conv2d(int(out_channel / 2), out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(True),
			nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(True)
		)
		self.short2 = nn.Sequential(
			nn.Conv2d(int(out_channel / 2), out_channel, 1, bias=False),
			nn.BatchNorm2d(out_channel)
		)
		self.classifier = nn.Sequential(nn.Linear(7 * 7 * out_channel, 7 * 7 * int(out_channel / 2)),
		                                nn.ReLU(True),
		                                nn.Linear(7 * 7 * int(out_channel / 2), 7 * 7 * int(out_channel / 2)),
		                                nn.ReLU(True),
		                                nn.Linear(7 * 7 * int(out_channel / 2), 10)
		                                )
		self.classifier = nn.Sequential(
			nn.Conv2d(out_channel, int(out_channel / 2), kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(int(out_channel / 2)),
			nn.ReLU(True),
			nn.Conv2d(int(out_channel / 2), int(out_channel / 2), kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(int(out_channel / 2)),
			nn.ReLU(True),
			nn.Conv2d(int(out_channel / 2), int(out_channel / 4), kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(int(out_channel / 4)),
			nn.ReLU(True),
			nn.Conv2d(int(out_channel / 4), int(out_channel / 4), kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(int(out_channel / 4)),
			nn.ReLU(True),
			nn.Conv2d(int(out_channel / 4), 10, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(10),
			nn.ReLU(True),
			nn.Conv2d(10, 10, kernel_size=3, padding=1, stride=2)
		)
	
	def get_lr(self, iteration, epochs, decay_rate=0.01):
		init_lr = 1e-2
		lr = init_lr * (decay_rate ** (iteration / (epochs-1)))
		return lr
	
	def forward(self, x):
		x = self.convblock1(x) + self.short1(x)
		x = self.maxpool(x)
		x = self.convblock2(x) + self.short2(x)
		x = self.maxpool(x)
		x = self.classifier(x)
		x = torch.flatten(x, 1)  # [n, 10]
		return x
