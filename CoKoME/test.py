import glob
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
from sklearn.metrics import classification_report
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


def evaluation(model, dataloader):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    disqust_list = []
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
            if true_label == 3:
                disqust_list.append(pred_label)
            pred_list.append(pred_label)
            label_list.append(true_label)

    return pred_list, label_list, disqust_list



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


    test_path = data_path + 'test.csv'


    test_dataset = CustomDataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=4, collate_fn=make_batch)


    last = False
        
    clsNum = 7
    init_config = Config()
    model = Model(text_model, audio_model, clsNum, last, init_config)
    model.load_state_dict(torch.load('/home/yuntaeyang_0629/ETRI/model.bin'))
    model = model.cuda()
    model.eval()

    test_pred_list, test_label_list, disgust_list = evaluation(model, test_loader)
    print(disgust_list)
    print(classification_report(test_label_list, test_pred_list, target_names=['neutral', 'surprise', 'angry', 'disqust', 'sad', 'happy', 'fear']))
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()