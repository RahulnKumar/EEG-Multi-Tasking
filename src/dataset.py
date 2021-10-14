from PIL import Image
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class single_task_dataset(Dataset):
    def __init__(self,df, transform=None):
        """
        Args:
            df (pandas dataframe): dataframe with path and labels of all EEG-images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        # import pdb; pdb.set_trace()
        label = self.df.Label.loc[index]
        label = torch.tensor(label).long()
        path = self.df.EEG_Path.iloc[index]
        path = f"input/{path}"
        # print('reading ', path)
        image = cv2.imread(path, 0)
        # print('reading successfully and size = ', image.shape)

        try:
            image = cv2.resize(image, (37, 75)) / 255
        except:
            pass
        image = np.reshape(image, (75, 37, 1))
       
        if self.transform is not None:
            image = self.transform(image)

        image = np.transpose(image, (2,0,1)).astype(np.float32)

        image = torch.tensor(image).float()

        
        #torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        #print(list_of_labels)
        return image, label
    

class multi_task_dataset(Dataset):
    def __init__(self,df, transform=None):
        """
        Args:
            df (pandas dataframe): dataframe with path and labels of all EEG-images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        # import pdb; pdb.set_trace()
        subject = self.df.Subject.loc[index]
        subject = torch.tensor(subject).long()
        label = self.df.Label.loc[index]
        label = torch.tensor(label).long()
        path = self.df.EEG_Path.iloc[index]
        path = f"input/{path}"
        # print('reading ', path)
        image = cv2.imread(path, 0)
        # print('reading successfully and size = ', image.shape)

        try:
            image = cv2.resize(image, (37, 75)) / 255
        except:
            pass
        image = np.reshape(image, (75, 37, 1))
       
        if self.transform is not None:
            image = self.transform(image)

        image = np.transpose(image, (2,0,1)).astype(np.float32)

        image = torch.tensor(image).float()

        
        #torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        #print(list_of_labels)
        return image, label, subject

    