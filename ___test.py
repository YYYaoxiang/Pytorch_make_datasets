import torch
import PIL
from torchvision import transforms
from _main import Net
from torch.autograd import Variable
import torch.nn as nn
#
# mxp=nn.MaxPool2d(2,2)
# conv1=nn.Conv2d(1,3,3)
# conv2=nn.Conv2d(3,6,3)
# data_in=torch.randn(1,28,28)
#
# data_out1=conv1(data_in)
# data_out2=mxp(data_out1)
# data_out3=conv2(data_out2)

in_image=PIL.Image.open('C:\\99999\\class4\\MNIST_FC\\mnist_image_label\\mnist_test_jpg_10000'
                        '\\15_5.jpg')

tf=transforms.ToTensor()
img_as_tensor=tf(in_image)
model_=Net()
model_.load_state_dict(torch.load('finished_model.pt'))
output=model_(Variable(img_as_tensor))
_, predicted = torch.max(output.data, 1)
print(predicted)


