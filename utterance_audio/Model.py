import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Model(nn.Module):
    def __init__(self, model_type, clsNum, last, init_config):
        super(Model, self).__init__()
        self.gpu = True
        self.last = last
        
        """Model Setting"""
        # model_path = '/data/project/rw/rung/model/'+model_type
        model_path = model_type
        if model_type == "facebook/wav2vec2-base-960h":

            self.model = Wav2Vec2Model.from_pretrained(model_path)
            self.model.config.update(init_config.__dict__)

        self.hiddenDim = self.model.config.hidden_size
            
        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_input):
        """
            batch_input_tokens: (batch, len)
        """
        if self.last:
            batch_audio_output = self.model(batch_input).last_hidden_state[:,-1,:] # (batch, 768)
        else:
            batch_audio_output = self.model(batch_input).last_hidden_state[:,0,:] # (batch, 768)
        audio_logit = self.W(batch_audio_output) # (batch, clsNum)
        
        return audio_logit