import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
import pdb

from transformers import AutoModel, AutoTokenizer
from transformers import Wav2Vec2Model

class Model(nn.Module):
    def __init__(self, text_model, audio_model, clsNum, last, init_config):
        super(Model, self).__init__()
        self.gpu = True
        self.last = last
        
        """Text Model"""
        tmodel_path = text_model
        if text_model == "klue/roberta-base":
            self.text_model = AutoModel.from_pretrained(tmodel_path)
            tokenizer = AutoTokenizer.from_pretrained(tmodel_path)
            self.speaker_list = ["[SEP1]", "[SEP2]"]
            self.speaker_tokens_dict = {'additional_special_tokens': self.speaker_list}
            tokenizer.add_special_tokens(self.speaker_tokens_dict)
            
        self.text_model.resize_token_embeddings(len(tokenizer))
        self.text_hiddenDim = self.text_model.config.hidden_size
        
        """Audio Model"""
        amodel_path = audio_model
        if audio_model == "facebook/wav2vec2-base-960h":

            self.audio_model = Wav2Vec2Model.from_pretrained(amodel_path)
            self.audio_model.config.update(init_config.__dict__) 

        self.audio_hiddenDim = self.audio_model.config.hidden_size

        
        #self.multihead_attn = nn.MultiheadAttention(self.audio_hiddenDim, 6)
        #self.norm = nn.LayerNorm([1,self.audio_hiddenDim])
        """score"""
        #self.W = nn.Linear(self.text_hiddenDim, clsNum)
        self.W = nn.Linear(self.text_hiddenDim + self.audio_hiddenDim, clsNum)

    def forward(self, batch_input_tokens, batch_audio):

        batch_audio_output = self.audio_model(batch_audio).last_hidden_state[:,0,:] # (batch, 768)
        batch_context_output = self.text_model(batch_input_tokens).last_hidden_state[:,0,:] # (batch, 768)
        concat_hidden_feature=torch.cat([batch_context_output, batch_audio_output], axis=1) # (batch , 768 + 768)
        #attn_output, _ = self.multihead_attn(batch_context_output, batch_audio_output, batch_context_output)

        #add_output = self.norm( attn_output + batch_context_output)

        #context_logit = self.W(add_output)
        context_logit = self.W(concat_hidden_feature)

        return context_logit