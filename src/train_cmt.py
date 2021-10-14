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

# Training the model for all the subjects
def conventional_multi_task_train():

    df = pd.read_csv('input/data.csv')
    df_train = df.loc[(df.Type == 'train') & (df.Subject != 10)].reset_index(drop=True)

    train_dataset = multi_task_dataset(df_train)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = multi_task_model_2().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    t = 0
    for epoch in range(num_epochs):
        tic = time.time()
        pbar = tqdm(total=total_step, desc='Epoch {}'.format(epoch+1))
        for i, (images, labels,subjects) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            # print('shapes = ', outputs.shape, labels.shape)
            # print(outputs, labels)
            loss = criterion(outputs[subjects-1], labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
            if (i+1) % 100 == 0:
                pbar.set_postfix({'Loss':  loss.item(),
                                 'Accuracy': 'Wait'})
        pbar.close()
        toc = time.time()
        t = t + toc - tic

        # Caluculate training accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels, subjects in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs[subjects-1].data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Training Accuracy at {epoch+1}th epoch = {100 * correct / total} \n')
        evaluate()


        # Saving the model
        torch.save({'weights': model.state_dict()}, config.cmt_model_path)

    print('total time for training =', t)
    with open(config.mt_result_path, 'w') as file:
        stats = {}
        variables = vars(config)  # dictionary all the private and public variable in config.py file
        config_dic = {key: value for key, value in variables.items()
                      if str(key)[0] != '_'}  # dictionary of only public variable in config.py file
        stats['config'] = config_dic
        stats['time'] = round(t,2)
        json.dump(stats, file)


def evaluate():

    all_acc = []
    checkpoint = torch.load(config.cmt_model_path)

    for subject in range(1, 10):
        # print(f"---------------------------TESTING SUBJECT {subject}--------------------------------")

        df = pd.read_csv('input/data.csv')
        df_test = df.loc[(df.Type == 'test') & (df.Subject == subject)].reset_index(drop=True)
        test_dataset = multi_task_dataset(df_test)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        test_dataset = multi_task_dataset(df_test)

        model = multi_task_model_2().to(device)
        model.load_state_dict(checkpoint['weights'])

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels, subjects in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs[subjects-1].data, 1)
        #         print('predicted and label = ', predicted.item(), labels.item())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        all_acc.append(acc)
        # print(f'Test Accuracy for subject {subject}= {acc}')
    print(f'Mean Testing accuracy = {sum(all_acc) / len(all_acc)}')
    store_results(all_acc)

def store_results(all_acc:list):
    stats = {j:i for j, i in enumerate(all_acc, 1)}
    stats['avg'] = sum(all_acc) / len(all_acc)
    with open(config.mt_result_path) as file:
        train_data = json.load(file)
        time = train_data['time']
        conf_dic = train_data['config']
        stats['time'] = time
        stats['config'] = conf_dic
    with open(config.mt_result_path, 'w') as file:
        json.dump(stats, file)