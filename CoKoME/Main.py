import glob
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass

from sklearn.metrics import precision_recall_fscore_support

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoProcessor, Wav2Vec2Model
import gc

from Data_preprocessing1 import *
from Data_preprocessing2 import *
from Dataset import *
from Model import *

CFG = { "epochs" : 15,
       "learning_rate" : 1e-6,
       "batch_size" : 1,
       "seed" : 44
       }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def model_train(training_epochs, model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, save_path):
    best_dev_fscore, best_test_fscore = 0, 0   
    best_epoch = 0
    for epoch in tqdm(range(training_epochs)):
        model.train() 
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            """Prediction"""
            batch_input_tokens, batch_audio, batch_labels = data
            batch_input_tokens, batch_audio, batch_labels = batch_input_tokens.cuda(), batch_audio.cuda(), batch_labels.cuda()
            pred_logits = model(batch_input_tokens, batch_audio)

            """Loss calculation & training"""
            loss_val = CELoss(pred_logits, batch_labels)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
   
            
        model.eval()    
        dev_pred_list, dev_label_list = evaluation(model, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')                
        print(f"dev_score : {dev_fbeta}")

        model.eval()    
        test_pred_list, test_label_list = evaluation(model, test_dataloader)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                
        print(f"test_score : {test_fbeta}")

        if test_fbeta > best_dev_fscore:
            best_dev_fscore = test_fbeta
            best_epoch = epoch
            _SaveModel(model, save_path)

def evaluation(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []

    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, batch_audio, batch_labels = data
            batch_input_tokens, batch_audio, batch_labels = batch_input_tokens.cuda(), batch_audio.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_tokens, batch_audio)
            
            """Calculation"""    
            pred_logits_sort = pred_logits.sort(descending=True)
            indices = pred_logits_sort.indices.tolist()[0]
            
            pred_label = indices[0] # pred_logits.argmax(1).item()
            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)

    return pred_list, label_list


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))

def main():
    seed_everything(CFG['seed'])
    @dataclass
    class Config():
        mask_time_length: int = 3


    sample = 1
    text_model = "klue/roberta-base"
    audio_model = "facebook/wav2vec2-base-960h"

    data_path = './ETRI/' 

    make_batch = make_batchs

        
    train_path = data_path + 'train.csv' 

    val_path = data_path + 'val.csv'

    test_path = data_path + 'test.csv'


    train_dataset = CustomDataset(preprocessing(train_path))
    train_loader = DataLoader(train_dataset, batch_size = CFG['batch_size'], shuffle=True, num_workers=4, collate_fn=make_batch)

    dev_dataset = CustomDataset(preprocessing(val_path))
    dev_loader = DataLoader(dev_dataset, batch_size = CFG['batch_size'], shuffle=True, num_workers=4, collate_fn=make_batch)

    test_dataset = CustomDataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=4, collate_fn=make_batch)

    save_path = os.path.join('./ETRI')
    
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    last = False
        
    clsNum = len(train_dataset.emoList)
    init_config = Config()
    model = Model(text_model, audio_model, clsNum, last, init_config)
    model = model.cuda()
    model.train()

    """Training Setting"""        
    training_epochs = CFG['epochs']
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = CFG['learning_rate']
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    model_train(training_epochs, model, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, save_path)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()