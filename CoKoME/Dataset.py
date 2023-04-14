from torch.utils.data import Dataset, DataLoader
import random

class CustomDataset(Dataset):
    def __init__(self, data):
        self.emoList = ['neutral', 'surprise', 'angry', 'disgust', 'sad', 'happy', 'fear']
        self.session_dataset = data

    def __len__(self): 
        return len(self.session_dataset)
    
    def __getitem__(self, idx): 
        return self.session_dataset[idx]