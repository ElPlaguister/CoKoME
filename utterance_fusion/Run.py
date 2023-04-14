import glob
import os
import pandas as pd
import numpy as np
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
from transformers import AutoProcessor, Wav2Vec2Model
import gc


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
    torch.save(model.state_dict(), os.path.join(path, 'utterence.bin'))