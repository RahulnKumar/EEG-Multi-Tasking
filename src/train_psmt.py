import pdb
import torch 
import random
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import multi_task_dataset
from src.model import Shared_Model
from src.model import Private_Model
import torchvision.transforms as transforms

# Device configuration
gpu = "cuda:1"
device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

# Hyper parameters
episodes = 2
num_epochs = 2
num_classes = 2
batch_size = 50
lr = 0.000001

Model_map = {0:Shared_Model, 1:Private_Model}
df = pd.read_csv('input/data.csv')
df = df.sample(frac=1).reset_index(drop=True)

def print_training_accuracy():
    df = pd.read_csv('input/data.csv')
    df_train = df.loc[(df.Type=='train') & (df.Subject!=10)].reset_index(drop = True)
    train_dataset = multi_task_dataset(df_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, 
                                               shuffle=True)
#     model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for image, label, subject in train_loader:
            n = subject[0]
            model = Model_map[1]
            model = model(n).to(device)
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
    #         print('predicted and label = ', predicted.item(), label.item())     

            total += label.size(0)
            correct += (predicted == label).sum().item()
#             print(predicted.item(), label.item())
    # print(total, correct)
    print('Train Accuracy : {} %'.format(100 * correct / total))

# Test the model on test data
def print_testing_accuracy():
    df = pd.read_csv('input/data.csv')
    df_test = df.loc[(df.Type=='test') & (df.Subject!=10)].reset_index(drop = True)
    test_dataset = multi_task_dataset(df_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1, 
                                               shuffle=True)
#     model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for image, label, subject in test_loader:
            n = int(subject[0])
            model = Model_map[1]
            model = model(n).to(device)
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
#             print('predicted and label = ', predicted.item(), label.item())     

            total += label.size(0)
            correct += (predicted == label).sum().item()
#             print(predicted.item(), label.item())
#     print(total, correct)
    print('Test Accuracy : {} %'.format(100 * correct / total))


# Train the model

def private_shared_multi_task_train():

    for episode in range(episodes):
        
    #     pdb.set_trace()
        n = random.randint(1,9)
    #     n = 1
        df_train = df.loc[(df.Type=='train') & (df.Subject==n)].reset_index(drop = True)
        train_dataset = multi_task_dataset(df_train)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)

        model = Model_map[1]
        model = model(n).to(device)


        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr)


        total_step = len(train_loader)
        for epoch in range(num_epochs):


            for i, (images, labels, subjects) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                subjects = subjects.to(device)

                # Forward pass 1
                outputs = model(images)
                loss = criterion(outputs, labels)
    #             model1.freeze(subjects[0].item())
    #             model1.to(device)




                # Backward and optimize 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                torch.save({'weights': model.state_dict()},f'models/shared_private/model_{n}.pth')
                torch.save({'weights': model.state_dict()},f'models/shared_private/model_{0}.pth')
                if (i+1) % 500 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        print(f"-------------  EPISODE : {episode}  ------------")
        print("------------------------------------------------")
        print_training_accuracy()
        print_testing_accuracy()
        print("------------------------------------------------", '\n')
    
