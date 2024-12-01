import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
from voice.helper_voice import *
from voice.config import *


voi_txt = pd.read_csv("voice_text.csv")

new_dataset = Dataset.from_pandas(voi_txt)

processor = Wav2Vec2Processor.from_pretrained("Nhut/wav2vec2-large-xlsr-vietnamese")
model = Wav2Vec2ForCTC.from_pretrained("Nhut/wav2vec2-large-xlsr-vietnamese")


for i in range(len(new_dataset)):
    combined_dataset = combined_dataset.add_item(new_dataset[i])

inputs = processor(combined_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)
with torch.no_grad():
  logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
predicted_ids = torch.argmax(logits, dim=-1)

# wer = load_metric("wer")
device = torch.device("cpu")
model.to(device)

chars_to_ignore_regex = '[\\\+\@\ǀ\,\?\.\!\-\;\:\"\“\%\‘\”\�]'


def voice2text(link_voice):
    waveform, sampling_rate = torchaudio.load(link_voice)
    if waveform.shape[0] == 2:  
        waveform = torch.mean(waveform, dim=0, keepdim=True) 
    target_sampling_rate = 16000
    if sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=target_sampling_rate)
        waveform = resampler(waveform)
        sampling_rate = target_sampling_rate
    waveform = waveform.squeeze(0)  
    waveform = waveform.unsqueeze(0)
    input_values = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values
    input_v = input_values[0]
    with torch.no_grad():
        logits = model(input_v).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return decode_string(transcription[0])




def get_result(path, qa):
    text = voice2text(path)  
    print("text",text)
    response = qa.invoke({"query": text})  
    result = response["result"]
    result = result.lower()
    print("result",result)
    if "tắt" in result:
        if "đèn" in result:
            if "hai" in result:
                return "tắt đèn hai"
            else: 
                return "tắt đèn một"
        elif "quạt" in result:
            return "tắt quạt"
    elif "bật" in result:
        if "đèn" in result:
            if "hai" in result:
                return "bật đèn hai"
            else:
                return "bật đèn một"
        elif "quạt" in result:
            return "bật quạt"
    elif "đóng" in result:
        return "đóng cửa"
    elif "mở" in result:
        return "mở cửa"
    else:
        return "unknown"







