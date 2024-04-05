
import pysrt
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizers import ByteLevelBPETokenizer

from audio_processor import load_audio, get_audio_length


def collate_fn(batch):
    batch = [s for s in batch if s != None]
    if len(batch) == 0:
        print('Empty batch because of None')
        return None
    if len([(a,b,p) for (a,b,p) in batch if len(b) < len(a)/16000*12.5]) == 0:
        print('Empty batch because of something to do with < 12.5')
        return None
    audio_tensors, bbpe_tensors, audio_paths = list(zip(*batch))
    audio_lens = [len(audio) for audio in audio_tensors]
    bbpe_lens = [len(bbpe) for bbpe in bbpe_tensors]
    audios_tensor = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True)
    audio_lens = torch.LongTensor(audio_lens)
    bbpes_tensor = torch.nn.utils.rnn.pad_sequence(bbpe_tensors, batch_first=True)
    bbpe_lens = torch.LongTensor(bbpe_lens)
    return audios_tensor, audio_lens, bbpes_tensor, bbpe_lens, audio_paths



def get_subs_in_range(subs, start, end):
    subs_in_range = []
    for sub in subs:
        if sub.start.ordinal/1000 > start and sub.end.ordinal/1000 < end:
            subs_in_range.append(sub)
    return subs_in_range

def split_text_by_time(subs, end_time: float, T: float = 60.000):
    random_start = np.random.uniform(0, max(0, end_time - T))
    random_end = random_start + T
    # Find the subs that are in the time range
    subs_in_range = get_subs_in_range(subs, random_start, random_end)
    return subs_in_range


class CourseraDataset(Dataset):
    def __init__(self, ds_path: pathlib.Path, episodes_list: list, tokenizer_path: pathlib.Path, T: float = 60.000):
        self.ds_path = ds_path
        self.episodes_list = episodes_list
        self.bbpe_tokenizer = ByteLevelBPETokenizer.from_file(f'{tokenizer_path}/vocab.json', f'{tokenizer_path}/merges.txt')

        self.T = T

    def __len__(self):
        return len(self.episodes_list)

    def __getitem__(self, idx):
        episode = self.episodes_list[idx]
        if isinstance(episode, str):
            episode_path = self.ds_path / episode
            audio_path = episode_path / 'audio.wav'
            srt_path = episode_path / 'srts/English.srt'
        else:
            audio_path = episode[0]
            srt_path = episode[1]

        # Text
        try:
            subs = pysrt.open(srt_path, encoding='iso-8859-1')
        except Exception as e:
            print(f'Error: {e}')
            return None

        audio_len = get_audio_length(audio_path)
        subs = split_text_by_time(subs, audio_len, self.T)

        text = ' '.join([sub.text for sub in subs])
        text = ' '.join(text.split('\n'))
        bbpes_list = [1] + self.bbpe_tokenizer.encode(text).ids + [2]
        # BBPEs to tensor
        bbpes_tensor = torch.tensor(bbpes_list, dtype=torch.long)

        # Audio
        try:
            start_time = subs[0].start.ordinal/1000
            end_time = subs[-1].end.ordinal/1000
            duration = end_time - start_time
            audio_tensor = load_audio(audio_path, offset=start_time, duration=duration)
            return audio_tensor, bbpes_tensor, audio_path
        except Exception as e:
            print(f'Error: {e}')
            return None