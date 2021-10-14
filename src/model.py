import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

gpu = "cuda:1"
device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

def conv_block(in_channels, out_channels):
    '''Convolution Block of 3x3 kernels + batch norm + maxpool of 2x2'''

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def set_grad(model, grad):
    for param in model.parameters():
        param.requires_grad = grad


class multi_task_model(nn.Module):
    def __init__(self):
        super(multi_task_model, self).__init__()

        self.conv1 = conv_block(1, 20)
        self.conv2 = conv_block(20, 40)
        self.conv3 = conv_block(40, 60)
        self.fc = nn.Linear(9 * 4 * 60, 100)
        self.head1 = nn.Linear(100, 50)
        self.head = nn.Linear(50, 2)

    def forward(self, x):
        # Shared layers across all task
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        feat = x.reshape(x.size(0), -1)   # Features of shape (batch_size, 64)

        # Task specific layers
        fc1 = self.fc(feat)
        out1 = self.head1(fc1)
        out1 = self.head(out1)

        fc2 = self.fc(feat)
        out2 = self.head1(fc2)
        out2 = self.head(out2)

        fc3 = self.fc(feat)
        out3 = self.head1(fc3)
        out3 = self.head(out3)

        fc4 = self.fc(feat)
        out4 = self.head1(fc4)
        out4 = self.head(out4)

        fc5 = self.fc(feat)
        out5 = self.head1(fc5)
        out5 = self.head(out5)

        fc6 = self.fc(feat)
        out6 = self.head1(fc6)
        out6 = self.head(out6)

        fc7 = self.fc(feat)
        out7 = self.head1(fc7)
        out7 = self.head(out7)

        fc8 = self.fc(feat)
        out8 = self.head1(fc8)
        out8 = self.head(out8)

        fc9 = self.fc(feat)
        out9 = self.head1(fc9)
        out9 = self.head(out9)

        return out1, out2, out3, out4, out5, out6, out7, out8, out9

class multi_task_model_2(nn.Module):
    def __init__(self):
        super(multi_task_model_2, self).__init__()

        self.conv1 = conv_block(1, 20)
        self.conv2 = conv_block(20, 40)
        self.conv3 = conv_block(40, 60)
        self.fc = nn.Linear(9 * 4 * 60, 100)
        self.head = nn.Linear(100, 2)

    def forward(self, x):
        # Shared layers across all task
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        feat = x.reshape(x.size(0), -1)   # Features of shape (batch_size, 64)

        # Task specific layers
        fc1 = self.fc(feat)
        out1 = self.head(fc1)

        fc2 = self.fc(feat)
        out2 = self.head(fc2)

        fc3 = self.fc(feat)
        out3 = self.head(fc3)

        fc4 = self.fc(feat)
        out4 = self.head(fc4)

        fc5 = self.fc(feat)
        out5 = self.head(fc5)

        fc6 = self.fc(feat)
        out6 = self.head(fc6)

        fc7 = self.fc(feat)
        out7 = self.head(fc7)

        fc8 = self.fc(feat)
        out8 = self.head(fc8)

        fc9 = self.fc(feat)
        out9 = self.head(fc9)

        return out1, out2, out3, out4, out5, out6, out7, out8, out9

class single_task_model(nn.Module):
    def __init__(self):
        super(single_task_model, self).__init__()

        self.conv1 = conv_block(1, 20)
        self.conv2 = conv_block(20, 40)
        self.conv3 = conv_block(40, 60)
        self.fc = nn.Linear(9 * 4 * 60, 100)
        self.head1 = nn.Linear(100, 50)
        self.head = nn.Linear(50, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Features of shape (batch_size, 64)
        feat = x.reshape(x.size(0), -1)

        # Output
        out = self.fc(feat)
        out = self.head1(out)
        out = self.head(out)

        return out

class Shared_Model(nn.Module):
    def __init__(self,):
        super(Shared_Model, self).__init__()
        
        self.conv1 = conv_block(1, 20)
        self.conv2 = conv_block(20, 40)
        self.conv3 = conv_block(40, 60)
        
        self.conv11 = conv_block(1, 20)
        self.conv22 = conv_block(20, 40)
        self.conv33 = conv_block(40, 60)
        
        self.fc = nn.Linear(9 * 4 * 60 * 2, 100)

    
       #heads
        self.y1o = nn.Linear(100,2)
      
    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        
        x2 = self.conv11(x)
        x2 = self.conv22(x2)
        x2 = self.conv33(x2)

        # Features of shape (batch_size, 64)
        feat1 = x1.reshape(x1.size(0), -1)
        feat2 = x2.reshape(x2.size(0), -1)
        feat  = torch.cat((feat1, feat2),1)
        
        head = self.fc(feat)
  
        # heads
        out = self.y1o(head)
    
        return out

class Private_Model(nn.Module):
    def __init__(self, private_layers):
        super(Private_Model, self).__init__()
        
        self.x = private_layers
        
        # Loading shared layers
        s_model = Shared_Model
        s_model = s_model().to(device)
        try:
            checkpoint = torch.load(f'models/shared_private/model_{0}.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
            s_model.load_state_dict(checkpoint['weights'])
            
#             print(f'loaded shared layer for model {self.x}')
        except:
#             pdb.set_trace()
            print(f"------------Create a brand new shared layer from scrath -------------------------")
        self.conv11 = s_model.conv11
        self.conv22 = s_model.conv22
        self.conv33 = s_model.conv33
        

        # Loading private layers
        p_model = Shared_Model
        p_model = p_model().to(device)
        try:
            checkpoint = torch.load(f'models/shared_private/model_{self.x}.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
            p_model.load_state_dict(checkpoint['weights'])
            
#             print(f'loaded private layer for model {x}')
        except:
#             pdb.set_trace()
#             print(f"------------Create a brand new private layer for subject {self.x} from scrath -------------------------")
            pass
    
        self.conv1 = p_model.conv1
        self.conv2 = p_model.conv2
        self.conv3 = p_model.conv3
        self.fc = p_model.fc
        self.y1o = p_model.y1o       
      
    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        
        x2 = self.conv11(x)
        x2 = self.conv22(x2)
        x2 = self.conv33(x2)

        # Features of shape (batch_size, 64)
        feat1 = x1.reshape(x1.size(0), -1)
        feat2 = x2.reshape(x2.size(0), -1)
        feat  = torch.cat((feat1, feat2),1)
        
        head = self.fc(feat)
  
        # heads
        out = self.y1o(head)
    
        return out

class Adversarial_Net(nn.Module):
    def __init__(self):
        super(Adversarial_Net, self).__init__()

        self.conv1 = conv_block(1, 20)
        self.conv2 = conv_block(20, 40)
        self.conv3 = conv_block(40, 60)

        self.head = nn.Linear(9 * 4 * 60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Features of shape (batch_size, 64)
        feat = x.reshape(x.size(0), -1)

        # Output
        out = self.head(feat)

        return out


class Multitask_Net(nn.Module):
    def __init__(self):
        super(Multitask_Net, self).__init__()

        self.conv1 = conv_block(1, 20)
        self.conv2 = conv_block(20, 40)
        self.conv3 = conv_block(40, 60)
        self.fc = nn.Linear(9 * 4 * 60 * 2, 100)
        
        task_model = Adversarial_Net().to(device)
        try:
            checkpoint = torch.load('models/adversarial_multi_task_train_batch/tasknet.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
            task_model.load_state_dict(checkpoint['weights'])
        except:
            print("------------Create a brand new model from scrath -------------------------")
        self.conv11 = task_model.conv1
        self.conv22 = task_model.conv2
        self.conv33 = task_model.conv3
    
       #heads
        self.y1o = nn.Linear(100,2)
        self.y2o = nn.Linear(100,2)
        self.y3o = nn.Linear(100,2)
        self.y4o = nn.Linear(100, 2)
        self.y5o = nn.Linear(100, 2)
        self.y6o = nn.Linear(100,2)
        self.y7o = nn.Linear(100, 2)
        self.y8o = nn.Linear(100, 2)
        self.y9o = nn.Linear(100, 2)
        
    def freeze(self, subject):
        set_grad(self.y1o, False)
        set_grad(self.y2o, False)
        set_grad(self.y3o, False)
        set_grad(self.y4o, False)
        set_grad(self.y5o, False)
        set_grad(self.y6o, False)
        set_grad(self.y7o, False)
        set_grad(self.y8o, False)
        set_grad(self.y9o, False)
        
        if subject==1:
            set_grad(self.y1o, True)
        if subject==2:
            set_grad(self.y2o, True)
        if subject==3:
            set_grad(self.y3o, True)
        if subject==4:
            set_grad(self.y4o, True)
        if subject==5:
            set_grad(self.y5o, True)
        if subject==6:
            set_grad(self.y6o, True)
        if subject==7:
            set_grad(self.y7o, True)
        if subject==8:
            set_grad(self.y8o, True)
        if subject==9:
            set_grad(self.y9o, True)
        
    

    def forward(self, x):
        
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        
        x2 = self.conv11(x)
        x2 = self.conv22(x2)
        x2 = self.conv33(x2)
        
        

        # Features of shape (batch_size, 64)
        feat1 = x1.reshape(x1.size(0), -1)
        feat2 = x2.reshape(x2.size(0), -1)
        feat  = torch.cat((feat1, feat2),1)
        
        head = self.fc(feat)
  
        # heads
        out1 = self.y1o(head)
        out2 = self.y2o(head)
        out3 = self.y3o(head)
        out4 = self.y4o(head)
        out5 = self.y5o(head)
        out6 = self.y6o(head)
        out7 = self.y7o(head)
        out8 = self.y8o(head)
        out9 = self.y9o(head)
        return out1, out2, out3, out4, out5, out6, out7, out8, out9