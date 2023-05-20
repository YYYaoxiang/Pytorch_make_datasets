import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import PIL
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CDFromImages(Dataset):
    def __init__(self, csv_path, csv_p2):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            transform: transform 操作
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # 读取 csv 文件
        self.data_info = pd.read_csv(csv_path, header=None)
        # 文件第一列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # 第二列是图像的 label
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # 计算 length
        self.data_len = len(self.data_info.index)
        self.p2 = csv_p2

    def __getitem__(self, index):
        # 从 pandas df 中得到文件名
        single_image_name = self.image_arr[index]
        # 读取图像文件
        img_as_img = PIL.Image.open(
            'C:\\99999\\class4\\MNIST_FC\\mnist_image_label\\' + self.p2 + '\\' + single_image_name)

        # 把图像转换成 tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # 得到图像的 label
        single_image_label = self.label_arr[index]

        return img_as_tensor, single_image_label

    def __len__(self):
        return self.data_len


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)  # 3*26*26
        self.pool = nn.MaxPool2d(2, 2)  # 3*13*13
        self.conv2 = nn.Conv2d(3, 6, 3)  # 6*11*11
        self.fc1 = nn.Linear(6 * 11 * 11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 6 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":

    custom_mnist_from_images = CDFromImages('file.csv', 'mnist_train_jpg_60000')
    test_file = CDFromImages('test_file.csv', 'mnist_test_jpg_10000')
    # transformations = transforms.Compose([transforms.ToTensor()])
    trainloader = torch.utils.data.DataLoader(custom_mnist_from_images, batch_size=8,
                                              shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_file, batch_size=8,
                                             shuffle=False, num_workers=2)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    torch.save(net.state_dict(),'finished_model.pt')
