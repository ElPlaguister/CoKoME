import torch
from transformers import AutoModel, AutoTokenizer
import librosa

from transformers import AutoProcessor
from moviepy.editor import VideoFileClip, concatenate_videoclips

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
roberta_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base") 
speaker_list = ["[SEP1]", "[SEP2]"]
speaker_tokens_dict = {'additional_special_tokens': speaker_list}
roberta_tokenizer.add_special_tokens(speaker_tokens_dict)

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.mask_token_id]

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids)
    
    return torch.tensor(pad_ids)

def padding_audio(batch_audio):
    max_len = 0
    for ids in batch_audio:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in batch_audio:
        pad_len = max_len-len(ids)
        add_ids = [ 0 for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids.tolist())
    
    return torch.tensor(pad_ids)

def get_audio(processor, path):
    audio, _ = librosa.load(path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    return inputs["input_values"][0]


def make_batchs(sessions):
    label_list = ['neutral', 'surprise', 'angry', 'disqust', 'sad', 'happy', 'fear'] # neutral, happy, surprise, angry, sad, disqust, fear
    batch_input, batch_audio, batch_labels = [], [], []
    for session in sessions:
        inputString = ""
        now_speaker = None
        audio_path = session[-1][2]
        for turn, line in enumerate(session):
            speaker, utt, _, emotion = line

            inputString += '[SEP' + str(speaker+1) + '] ' # [SEP1], [SEP2], [SEP3], ....
            inputString += utt + " "
            now_speaker = speaker

        # audio_input 
        audio_input = get_audio(processor, audio_path)  
        batch_audio.append(audio_input) 

        # text_input
        concat_string = inputString.strip()
        concat_string += " " + "[SEP]"
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

        # label
        label_ind = label_list.index(emotion)
        batch_labels.append(label_ind)        

    # trunc, padding
    batch_input_tokens = padding(batch_input, roberta_tokenizer)
    batch_audio = padding_audio(batch_audio) 
    batch_labels = torch.tensor(batch_labels)    
    
    return batch_input_tokens, batch_audio, batch_labels
