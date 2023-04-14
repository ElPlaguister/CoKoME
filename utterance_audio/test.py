import glob
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoProcessor, Wav2Vec2Model
import gc

from Dataset import *
from Model import *

CFG = { "epochs" : 10,
       "learning_rate" : 1e-6,
       "batch_size" : 1,
       "seed" : 41
       }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
   
def validation(model, val_loader, class_s, device):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for audio_inputs, labels in iter(val_loader):
            audio_inputs = audio_inputs["input_values"][0].to(device)
            labels = labels.to(device)
            
            logit = model(audio_inputs)
             
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
    
    return print(classification_report(trues, preds, target_names=class_s))
    


@dataclass
class Config():
    mask_time_length: int = 3

def main():
    seed_everything(CFG['seed'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    train = pd.read_csv('./ETRI/train.csv')

    test = pd.read_csv('./ETRI/test.csv')

    le = LabelEncoder()
    le = le.fit(train['Label']) 
    test['Label'] = le.transform(test['Label'])


    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")


    test_dataset = CustomDataset(test['Wav_Path'].values, test['Label'].values, processor)
    test_loader = DataLoader(test_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=0)

    init_config = Config()
    model = Model("facebook/wav2vec2-base-960h", 7 , False, init_config)
    model.load_state_dict(torch.load('/home/yuntaeyang_0629/ETRI/utterence_audio/__pycache__/ETRI_wav.bin'))
    model = model.cuda()
    model.eval()
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
    validation(model, test_loader, le.classes_, device)
    print("---------------Done--------------")
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()