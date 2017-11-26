from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print ("PyTorch Basics")

# x = Variable(torch.ones(2,2), requires_grad = True)
# print (x)
# y = x + 2
# print (y)
# z = y * y * 3
# out = z.mean()
# print (z, out)

# print (x.grad) #none. before prop backward
# out.backward()
# print (x.grad)

# x = torch.randn(3)
# x = Variable(x, requires_grad=True)

# y = x * 2
# while y.data.norm() < 1000:
#     y = y * 2

# print(y)

# gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
# print (gradients)
# y.backward(gradients)
# print (x.grad)
# print (x)

print ("Define the network")

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		#class torch.nn.Linear(in_features, out_features, bias=True)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# conv-relu-pool
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		# view is reshape. # the size -1 is inferred from other dimensions
		x = x.view(-1, self.num_flat_features(x))
		# forward to fc layers
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# output layer dont need relu
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):

		size = x.size()[1:] #CAUTION. the 1st dimension is batch index, we only need size from 2nd column
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
net = Net()
print(net)
params = list(net.parameters())
print("num of params:"+str(len(params))) #weights in all layers
for i in range (0,len(params)):
	print(params[i].size())

# inputs
input = Variable(torch.randn(1, 1, 32, 32))
# input 4D tensor: nSamples x nChannels x Height x Width.
# input.unsqueeze(0) to add fake number dimension
out = net(input)
print ("output:"+str(out))

# net.zero_grad()
# backward(variables) Variables of which the derivative will be computed
# grad can be implicitly created only for scalar outputs (1-by-1 matrix)
# out.backward(torch.randn(1, 10))

# training target
target = Variable(torch.arange(1,11)) # dummy target example
criterion = nn.MSELoss() # we can define our own loss function here
loss = criterion(out, target) 
print("Loss:" + str(loss))
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Optimisation
# create optimizer
print ("OPTIMISATION")
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# run
optimizer.zero_grad() # zero everything first :)
output = net(input)
loss = criterion(output, target)
print("START")
print("Grad")
print(net.conv1.bias.grad)

params = list(net.parameters())
print("Weights sample")
print(params[0][0]) 

loss.backward()
print("After backward")
print("Grad")
print(net.conv1.bias.grad)

optimizer.step()
print("After update")
print("Weights sample")
print(params[0][0]) 



