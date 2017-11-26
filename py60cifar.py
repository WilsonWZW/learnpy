from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# Prepare the data
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg,(1, 2, 0)))
	#plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Define the neural network
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5) #in_channels, out_channels, kernel_size, etc.
		self.pool = nn.MaxPool2d(2,2) #kernel_size, stride
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()
print(net)

# Define Loss and Optimisation
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

#Training
for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0): # enumerate(sequence, start=0)
		inputs, labels = data # get the inputs
		inputs, labels = Variable(inputs), Variable(labels) # wrap them in Variable
		optimizer.zero_grad() # reset gradient in each passing

		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward() # calculate the gradients
		optimizer.step() # optimization

		running_loss += loss.data[0]
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Traning')
PATH = 'save/cifar.pt'
# net.save_state_dict('cifar.pt')
torch.save(net.state_dict(), PATH)
print('Saved')

# net.load_state_dic(torch.load('cifar.pt'))
net = Net()
net.load_state_dict(torch.load(PATH))
print('Loaded')

# pred random pics
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GT:',' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1) # return  (max, index). input(input tensor, dim)
print('Pred:',' '.join('%5s' % classes[predicted[j]] for j in range(4)))

class_correct = list(0. for i in range(10)) # 0. means 0.0, 0 means int 0
class_total = list(0. for i in range(10))
for data in testloader:
	images, labels = data
	outputs = net(Variable(images))
	_, predicted = torch.max(outputs.data, 1)
	c = (predicted == labels).squeeze()
	for i in range(4):
		label = labels[i]
		class_correct[label] += c[i]
		class_total[label] += 1

for i in range(10):
	print('Accruracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))