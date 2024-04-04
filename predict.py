import torchvision.transforms as T
import matplotlib.pyplot as plt
from Datasets import myDatasets
from model import my_model
import torch
import random

data_transform = T.Compose([T.ToTensor(), T.Normalize(std=0.5, mean=0.5)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = my_model(300)

model.to(device)
model.load_state_dict(torch.load(r"minist.pth"))
val_data = myDatasets(r"mnist_all.mat", mode="test", transform=data_transform)

for i in random.sample(range(len(val_data)), len(val_data)):
	plt.xticks([])
	plt.yticks([])
	img = val_data[i][0]
	label = val_data[i][1]

	model.eval()
	with torch.no_grad():
		predict = model(torch.stack([img], dim=0).to(device))
		p = torch.softmax(predict, dim=1)
		cls = str(p.argmax(1).item())
		prob = p.max().item()

	plt.title("pre:{}/{:.3f},label:{}".format(cls, prob, label))
	plt.imshow(img.permute(1, 2, 0))
	plt.show()
