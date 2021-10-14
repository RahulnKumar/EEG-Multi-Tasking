# Test the model
import json
import torch
import pandas as pd
import src.config as config
from src.model import multi_task_model_2
from src.dataset import multi_task_dataset



batch_size = config.batch_size

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def evaluate():

    all_acc = []
    checkpoint = torch.load(config.mt_model_path)

    for subject in range(1, 10):
        print(f"---------------------------TESTING SUBJECT {subject}--------------------------------")

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
        print(f'Test Accuracy for subject {subject}= {acc}')
    print(f'Average accuracy = {sum(all_acc) / len(all_acc)}')
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