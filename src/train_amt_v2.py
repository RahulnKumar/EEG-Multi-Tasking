import torch 
import random
import traceback
import torchvision
import json
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import single_task_dataset
from src.dataset import multi_task_dataset
from src.model import Adversarial_Net
from src.model import Multitask_Net
import torchvision.transforms as transforms

# Device configuration
gpu = "cuda:2"
device = torch.device(gpu if torch.cuda.is_available() else 'cpu')

# Hyper parameters
episodes = 10
num_epochs = 2
num_classes = 2
batch_size = 20
# lr1 = 0.0001
# lr2 = 0.000001

df = pd.read_csv('input/data.csv')
df = df.sample(frac=1).reset_index(drop=True)

df_train = df.loc[(df.Type=='train') & (df.Subject!=14)].reset_index(drop = True)
df_test = df.loc[(df.Type=='test' )& (df.Subject!=14)].reset_index(drop = True)
train_dataset = multi_task_dataset(df_train)
test_dataset = multi_task_dataset(df_test)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


# Test the model

def test_accuracy_multitask_net():
    
    df_test = df.loc[(df.Type=='test') & (df.Subject!=10)].reset_index(drop = True)
    test_dataset = multi_task_dataset(df_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    
    model1 = Multitask_Net().to(device)
    try:
        checkpoint = torch.load('models/adversarial_multi_task_train_batch/convnet2.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
        model1.load_state_dict(checkpoint['weights'])
    #         print("--------- Successfully loaded previous model weights --------------------")
    except:
        print("-------Creating convnet from scratch -----------")
    

    model1.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, subjects in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model1(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
    a = 100 * correct / total
    return a


# Test the model

def train_accuracy_multitask_net():
    
    df_train = df.loc[(df.Type=='train') & (df.Subject!=10)].reset_index(drop = True)
    train_dataset = multi_task_dataset(df_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    
    model1 = Multitask_Net().to(device)
    try:
        checkpoint = torch.load('models/adversarial_multi_task_train_batch/convnet2.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
        model1.load_state_dict(checkpoint['weights'])
    #         print("--------- Successfully loaded previous model weights --------------------")
    except:
        print("-------Creating convnet from scratch -----------")

    model1.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, subjects in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model1(images)
            _, predicted = torch.max(outputs[0].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Train Accuracy: {} %'.format(100 * correct / total))
    a = 100 * correct / total
    return a


# Test the model
def test_accuracy_adversarial_net():
    
    df_test = df.loc[(df.Type=='test') & (df.Subject!=10)].reset_index(drop = True)
    test_dataset = multi_task_dataset(df_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    
    model2 = Adversarial_Net().to(device)
    try:
        checkpoint = torch.load('models/adversarial_multi_task_train_batch/tasknet2.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
        model2.load_state_dict(checkpoint['weights'])
    #         print("--------- Successfully loaded previous model weights --------------------")
    except:
        print("-------Creating task net from scratch -----------")
    
    
    model2.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, subjects in test_loader:
            images = images.to(device)
            labels = subjects.to(device)
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
#             print(predicted.item(), labels.item())

    print('Test Accuracy for tasknet: {} %'.format(100 * correct / total))
    a = 100 * correct / total
    return a

# Test the model
def train_accuracy_adversarial_net():
    
    df_train = df.loc[(df.Type=='train') & (df.Subject!=10)].reset_index(drop = True)
    train_dataset = multi_task_dataset(df_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    
    model2 = Adversarial_Net().to(device)
    try:
        checkpoint = torch.load('models/adversarial_multi_task_train_batch/tasknet2.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
        model2.load_state_dict(checkpoint['weights'])
    #         print("--------- Successfully loaded previous model weights --------------------")
    except:
        print("-------Creating task net from scratch -----------")
    
    
    model2.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels, subjects in train_loader:
            images = images.to(device)
            labels = subjects.to(device)
            outputs = model2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
#             print(predicted.item(), labels.item())

    print('Train Accuracy for tasknet: {} %'.format(100 * correct / total))
    a = 100 * correct / total
    return a





# Train the model

def adversarial_multi_task_train_v2():

    for episode in range(episodes):

        

        if episode < 10:
            lr1 = 0.000001
            lr2 = 0.00001   
        else:
            lr1 = 0.0000001
            lr2 = 0.000001

        n = random.randint(1,9)

        df_train = df.loc[(df.Type=='train') & (df.Subject!=10)].reset_index(drop = True)
        train_dataset = multi_task_dataset(df_train)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=True)

        try:
            checkpoint = torch.load('models/adversarial_multi_task_train_batch/convnet2.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
            model1 = Multitask_Net().to(device)
            model1.load_state_dict(checkpoint['weights'])
            model1 = model1.to(device)
        #         print('-------Trained model loaded-----------')
        except :
            model1 = Multitask_Net()
            model1 = model1.to(device)
            print('-------New convnet model created --------------')

        try:
            checkpoint = torch.load('models/adversarial_multi_task_train_batch/tasknet2.pth', map_location=torch.device(gpu if torch.cuda.is_available() else 'cpu'))
            model2 = Adversarial_Net().to(device)
            model2.load_state_dict(checkpoint['weights'])
            model2 = model2.to(device)
        #         print('-------Trained model loaded-----------')
        except :
    #         traceback.print_exc()
            model2 = Adversarial_Net()
            model2 = model2.to(device)
            print('-------New tasknet created --------------')



        # Loss and optimizer
        criterion1 = nn.CrossEntropyLoss()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr1)

        criterion2 = nn.CrossEntropyLoss()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr2)


        total_step = len(train_loader)
        for epoch in range(num_epochs):


            for i, (images, labels, subjects) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                subjects = subjects.to(device)

                # Forward pass 1
                outputs1 = model1(images)
                loss1 = criterion1(outputs1[0], labels)
    #             model1.freeze(subjects[0].item())
    #             model1.to(device)



                # Forward pass 2
                outputs2 = model2(images)
                loss2 = criterion2(outputs2, subjects)

                # Backward and optimize 1
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()

                if episode < 5:
                    # Backward and optimize 2
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()

                torch.save({'weights': model2.state_dict()},'models/adversarial_multi_task_train_batch/tasknet2.pth')
                torch.save({'weights': model1.state_dict()},'models/adversarial_multi_task_train_batch/convnet2.pth')

                if (i+1) % 10000 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, num_epochs, i+1, total_step, loss1.item()))

        print(f"-------------  EPISODE : {episode}  ------------")
        print("------------------------------------------------")
        a = train_accuracy_multitask_net()
        b = test_accuracy_multitask_net()
        c = train_accuracy_adversarial_net()
        d = test_accuracy_adversarial_net()
        acc = [a,b,c,d]
        store_results(acc)
        print("------------------------------------------------", '\n')

def store_results(accuracy):
    try:
        with open('results/amt2.json') as file:
            accuracy_list = json.load(file)
    #         print(accuracy_list)
        with open('results/amt2.json', 'w') as file:
            accuracy_list.append(accuracy)
            json.dump(accuracy_list,file)
    #     print('writing on existing file')
    except:
    #     print('creating a new file')
        with open('results/amt2.json', 'w') as file:
            json.dump([accuracy],file)