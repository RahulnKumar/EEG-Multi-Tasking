import json
import time
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import src.config as config
from src.model import multi_task_model_2
from src.model import single_task_model
from src.dataset import multi_task_dataset
from src.dataset import single_task_dataset

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = config.num_epochs
num_classes = config.num_classes
batch_size = config.batch_size
learning_rate = config.learning_rate

# Training subject specific model
def single_task_train():
    all_acc = []
    t = 0

    for subject in range(1, 10):

        tic = time.time()
        print(f"---------------------------TRAINING SUBJECT {subject}--------------------------------")
        df = pd.read_csv('input/data.csv')
        df_train = df.loc[(df.Type == 'train') & (df.Subject == subject)].reset_index(drop=True)
        df_test = df.loc[(df.Type == 'test') & (df.Subject == subject)].reset_index(drop=True)
        df_train.shape, df_test.shape

        train_dataset = single_task_dataset(df_train)
        test_dataset = single_task_dataset(df_test)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        model = single_task_model().to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                # print('shapes = ', outputs.shape, labels.shape)
                # print(outputs, labels)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (i + 1) % 100 == 0:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


        # Saving the model

        torch.save({'weights': model.state_dict()}, f'{config.st_model_path}_{subject}.pth')

        toc = time.time()
        t = t + toc - tic

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        all_acc.append(acc)
        print(f'Test Accuracy for subject {subject}= {acc}')
    print(f'Average accuracy = {sum(all_acc)/len(all_acc)}')
    print('total time for training =', t)
    store_results(all_acc, round(t,2))

def store_results(all_acc:list, time:float):
    stats = {j:i for j, i in enumerate(all_acc, 1)}        # accuracy dictionary for all the subjects
    variables = vars(config)             # dictionary all the private and public variable in config.py file
    config_dic = {key: value for key, value in variables.items()
                  if str(key)[0] != '_'}   # dictionary of only public variable in config.py file
    stats['avg'] = sum(all_acc) / len(all_acc)                 # avg accuracy of all the subjects
    stats['time'] = time
    stats['config'] = config_dic
    with open(config.st_result_path, 'w') as file:
        json.dump(stats, file)