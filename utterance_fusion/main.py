from Dataset import *
from Model import *
from utils import *
from sklearn.preprocessing import LabelEncoder
import warnings
import gc
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoProcessor, Wav2Vec2Model
from sklearn.metrics import classification_report

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
warnings.filterwarnings('ignore')


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
    
def main():

    CFG = { "epochs" : 10,
       "learning_rate" : 1e-6,
       "batch_size" : 1,
       "seed" : 44
       }
       
    seed_everything(CFG['seed'])
    @dataclass
    class Config():
        mask_time_length: int = 3


    text_model = "klue/roberta-base"
    audio_model = "facebook/wav2vec2-base-960h"

    data_path = './ETRI/' 
     
    train_path = data_path + 'train.csv' 


    last = False

    test_path = data_path + 'test.csv'

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    le = LabelEncoder()
    le=le.fit(train['Label'])
    train['Label']=le.transform(train['Label'])
 
    test['Label']=le.transform(test['Label'])

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    roberta_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")


    test_dataset = CustomDataset(test, roberta_tokenizer, processor)
    test_loader = DataLoader(test_dataset, batch_size = CFG['batch_size'], shuffle=False, num_workers=4)

        
    clsNum = 7
    init_config = Config()
    model = Model(text_model, audio_model, clsNum, last, init_config)
    model.load_state_dict(torch.load('/home/yuntaeyang_0629/ETRI/utterence.bin'))
    model = model.cuda()
    model.eval()

   
    dev_pred_list, dev_label_list = evaluation(model, test_loader)
    print(classification_report(dev_label_list, dev_pred_list, target_names=le.classes_))  
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    main()