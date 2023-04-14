import librosa
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  
    def __init__(self, data, tokenizer, processor):
        self.dataset = data
        self.tokenizer = tokenizer
        self.processor = processor 

        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''text'''
        text = self.dataset['Utterance'][idx]
        inputs = self.tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]
        
        '''audio'''
        audio_input = self.get_audio(self.dataset['Wav_Path'][idx])
        
        y = self.dataset['Label'][idx]
        
        return input_ids, audio_input["input_values"][0], y

    def get_audio(self, path):
      audio, _ = librosa.load(path)
      inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
      return inputs