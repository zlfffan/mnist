from torch.utils import tensorboard
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, RandomCrop
import torch.nn as nn
from model import my_model
import sys
import torch
from Datasets import myDatasets

transform = {
	"train": Compose([ToTensor(),
	                  RandomCrop((28, 28), 4),
	                  Normalize(0.5, 0.5)]),
	"val": Compose([ToTensor(),
	                Normalize(0.5, 0.5)])}

train_data = myDatasets(r"mnist_all.mat", mode="train", transform=transform["train"])
val_data = myDatasets(r"mnist_all.mat", mode="test", transform=transform["val"])

print("using {} images for training, {} images for validation.".format(len(train_data), len(val_data)))

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")

output_channel = 300
model = my_model(output_channel, mode="full_conn")
model = model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0, momentum=0.95)
epochs = 10

len_batch = len(val_loader)
val_size = len(val_data)
best_acc = 0

writer = tensorboard.SummaryWriter("run/my_model")
# tensorboard --logdir=run

for epoch in range(epochs):
	model.train()
	train_bar = tqdm(train_loader, file=sys.stdout, unit=" batches")
	train_correct = 0
	train_loss = 0
	train_acc = 0
	lr = model.get_lr(epoch, epochs)
	# 动态调整学习率和动量
	for i in optimizer.param_groups:
		i['lr'] = lr
		if epochs >= epochs // 2:
			i["monument"] = 0.9
	for batch, (imgs, labels) in enumerate(train_bar):
		imgs = imgs.to(device)
		labels = labels.to(device)
		predict = model(imgs)
		train_correct += (predict.argmax(1) == labels).sum().item()
		train_acc = train_correct / ((batch + 1) * batch_size)
		loss = loss_func(predict, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss = (batch * train_loss + loss.item()) / (batch + 1)
		train_bar.desc = "data epoch[{}/{}] train_acc:{:.3f} train_loss:{:.3f} lr:{:.6f}".format(epoch + 1, epochs,
		                                                                                         train_acc, train_loss, lr)
	
	model.eval()
	val_acc = 0
	correct = 0
	val_loss = 0
	with torch.no_grad():
		val_bar = tqdm(val_loader, file=sys.stdout, unit=" batches")
		for imgs, labels in val_bar:
			imgs = imgs.to(device)
			labels = labels.to(device)
			predict = model(imgs)
			correct += (predict.argmax(1) == labels).sum().item()
			val_loss += loss_func(predict, labels).item()
		val_loss /= len_batch
		val_acc = correct / val_size
	print('[epoch %d]  val_acc: %.3f  val_loss: %.3f  ' % (epoch + 1, val_acc, val_loss))
	
	if val_acc > best_acc:
		best_acc = val_acc
		torch.save(model.state_dict(), "minist.pth")
		print("-----save best model------")
	
	writer.add_scalars('loss', {"train": round(train_loss, 3), "val": round(val_loss, 3)}, epoch + 1)
	writer.add_scalars('acc', {"train": round(train_acc, 3), "val": round(val_acc, 3)}, epoch + 1)

writer.close()
print('Finished Training')
