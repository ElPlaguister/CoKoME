import librosa

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, audio_path_list, label_list, feature_extractor):
        self.audio_path_list = audio_path_list
        self.label_list = label_list
        self.feature_extractor = feature_extractor

        
    def __getitem__(self, index):
        audio_input = self.get_audio(self.audio_path_list[index])
        label = self.label_list[index]

        return audio_input, label
        
    def __len__(self):
        return len(self.audio_path_list)
    
    def get_audio(self, path):
        audio, _ = librosa.load(path)
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        return inputs