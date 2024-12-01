from voice.config import *
from datasets import load_dataset, load_metric, Dataset
import re
import torchaudio

resampler = torchaudio.transforms.Resample(48_000, 16_000)

def decode_string(x):
  for k, v in list(reversed(list(ENCODER.items()))):
    x = x.replace(v, k)
  return x

test_dataset = load_dataset("mozilla-foundation/common_voice_13_0", "vi", split="test[:2%]", token=access_token, trust_remote_code=True)

selected_columns = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in ['sentence',"path"]])

def read_audio_file(link_voice):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

def speech_file_to_array_fn(batch):
  speech_array, sampling_rate = torchaudio.load(batch["path"])
  if speech_array.shape[0] == 2:
    speech_array = speech_array.mean(dim=0, keepdim=True)
  batch["speech"] = resampler(speech_array).squeeze().numpy()
  return batch

def speech_file_to_array_fn_valid(batch):
  batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
  speech_array, sampling_rate = torchaudio.load(batch["path"])
  batch["speech"] = resampler(speech_array).squeeze().numpy()
  return batch


selected_columns = selected_columns.map(speech_file_to_array_fn)

combined_dataset = Dataset.from_dict(selected_columns.to_dict())


