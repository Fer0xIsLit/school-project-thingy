import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

s = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(s)
print(s)

fw = open('images', 'rb')
train_img = fw.read()
fw.close()

fw = open('labels', 'rb')
train_lab = fw.read()
fw.close()

fw = open('test images', 'rb')
test_img = fw.read()
fw.close()

fw = open('test labels', 'rb')
test_lab = fw.read()
fw.close()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(784, 18)
        self.fc1 = nn.Linear(18, 18)
        self.fc2 = nn.Linear(18, 10)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x


def bytes_to_int(byte):
    s = ''
    for b in byte:
        m = hex(b).split('x')[1]
        if len(m) == 1:
            m = '0'+m
        
        s += m
    
    return int(s, 16)

def get_image_test(i):
    img = np.array([j for j in test_img[16+784*i:16+784*(i+1)]]).reshape(28, 28) / 255.0
    lab = test_lab[8+i]
    
    return img, lab

def get_image(i):
    img = np.array([j for j in train_img[16+784*i:16+784*(i+1)]]).reshape(28, 28) / 255.0
    lab = train_lab[8+i]
    
    return img, lab

def make_training_data():
    X, y = [], []
    for i in tqdm(range(60000)):
        img = get_image(i)
        X.append(img[0])
        y.append(np.eye(10)[img[1]])
    
    return torch.tensor(X), torch.tensor(y)
#    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

def make_test_data():
    X, y = [], []
    for i in tqdm(range(10000)):
        img = get_image_test(i)
        X.append(torch.tensor(img[0]))
        y.append(torch.tensor(np.eye(10)[img[1]]))
    
    return X, y
    #return torch.Tensor(X), torch.Tensor(y)

def train(net, epochs=3, batch_size=10):
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    loss_function = nn.MSELoss()
    
    for epoch in range(epochs):
        for i in tqdm(range(0, len(train_x), batch_size)):
            x = train_x[i:i+batch_size].float().to(device)
            y = train_y[i:i+batch_size].float().to(device)
            
            net.zero_grad()
            
            outputs = net(x.view(-1, 784))
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
        
        print(loss)

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, img in tqdm(list(enumerate(test_x))):
            guess = torch.argmax(net(img.view(1, 784).float().to(device)))
            if guess == torch.argmax(test_y[i].to(device)):
                correct += 1

            total += 1
    
    return correct/total