import glob
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass

from sklearn.metrics import f1_score
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

def model_train(model, epochs, optimizer, train_loader, val_loader, test_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        train_loss = []
        for audio_inputs, labels in iter(train_loader):
            audio_inputs = audio_inputs["input_values"][0].to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(audio_inputs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _train_loss = np.mean(train_loss)
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _test_loss, _test_score = validation(model, criterion, test_loader, device)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}] Test Loss : [{_test_loss:.5f}] Test F1 : [{_test_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
            torch.save(best_model.state_dict(), 'ETRI_wav.bin')
            print('***** Best Model *****')
            
   

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []
    
    with torch.no_grad():
        for audio_inputs, labels in iter(val_loader):
            audio_inputs = audio_inputs["input_values"][0].to(device)
            labels = labels.to(device)
            
            logit = model(audio_inputs)
            
            loss = criterion(logit, labels)
            
            val_loss.append(loss.item())
            
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
        
        _val_loss = np.mean(val_loss)
    
    _val_score = f1_score(trues, preds, average='weighted')
    return _val_loss, _val_score


@dataclass
class Config():
    mask_time_length: int = 3

def main():
    seed_everything(CFG['seed'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    train = pd.read_csv('./ETRI/train.csv')
    dev = pd.read_csv('./ETRI/val.csv')
    test = pd.read_csv('./ETRI/test.csv')

    le = LabelEncoder()
    le = le.fit(train['Label']) 
    train['Label'] = le.transform(train['Label'])  
    dev['Label'] = le.transform(dev['Label']) 
    test['Label'] = le.transform(test['Label'])


    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

    train_dataset = CustomDataset(train['Wav_Path'].values, train['Label'].values, processor)
    train_loader = DataLoader(train_dataset, batch_size = CFG['batch_size'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(dev['Wav_Path'].values, dev['Label'].values, processor)
    val_loader = DataLoader(val_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=0)

    test_dataset = CustomDataset(test['Wav_Path'].values, test['Label'].values, processor)
    test_loader = DataLoader(test_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=0)

    init_config = Config()
    model = Model("facebook/wav2vec2-base-960h", 7 , False, init_config)
    model.eval()
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = CFG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
    infer_model, train_loss_epoch_, valid_loss_epoch_, valid_score_epoch_ = model_train(model, CFG['epochs'], optimizer, train_loader, val_loader, test_loader, scheduler, device)
    print("---------------Done--------------")
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()