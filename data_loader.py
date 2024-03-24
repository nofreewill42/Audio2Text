
import pysrt
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from xformers.ops import fmha

from tokenizers import ByteLevelBPETokenizer

from audio_processor import load_audio, get_audio_length


def collate_fn(batch):
    batch = [s for s in batch if s != None]
    if len(batch) == 0:
        print('Empty batch because of None')
        return None
    if len([(a,b,m) for (a,b,m) in batch if len(b) < len(a)/16000*12.5]) == 0:
        print('Empty batch because of something to do with < 12.5')
        return None
    cross_mask = batch[0][2]  # TODO: do the mask here for all the batch (one big mask)
    batch = [(a,b) for (a,b,m) in batch]
    audio_tensors, bbpe_tensors = list(zip(*batch))
    audio_lens = [len(audio) for audio in audio_tensors]
    bbpe_lens = [len(bbpe) for bbpe in bbpe_tensors]
    audios_tensor = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True)
    audio_lens = torch.LongTensor(audio_lens)
    bbpes_tensor = torch.nn.utils.rnn.pad_sequence(bbpe_tensors, batch_first=True)
    bbpe_lens = torch.LongTensor(bbpe_lens)
    return audios_tensor, audio_lens, bbpes_tensor, bbpe_lens, cross_mask



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
        episode_path = self.ds_path / self.episodes_list[idx]
        audio_path = episode_path / 'audio.wav'
        srt_path = episode_path / 'srts/English.srt'

        # Text
        try:
            subs = pysrt.open(srt_path, encoding='iso-8859-1')
        except Exception as e:
            print(f'Error: {e}')
            return None

        audio_len = get_audio_length(audio_path)
        subs = split_text_by_time(subs, audio_len, self.T)

        # text = ' '.join([sub.text for sub in subs])
        # text = ' '.join(text.split('\n'))
        texts = [sub.text.replace('\n', ' ') for sub in subs]
        bbpeses = [self.bbpe_tokenizer.encode(text).ids + [2] for text in texts]  # 2 is the end token, for each audio chunk
        starts = [sub.start.ordinal for sub in subs]
        ends = [sub.end.ordinal for sub in subs]
        try:
            start_time = starts[0]
        except Exception as e:
            print(f'Error: {e}')
            return None
        end_time = ends[-1]
        starts = [s - start_time for s in starts]
        ends = [e - start_time for e in ends]
        start_time /= 1000
        end_time /= 1000
        # Fill in gaps between ends[i] and starts[i+1] - START
        for i in range(len(starts)-1):
            s_i, e_i = starts[i], ends[i]
            s_i1, e_i1 = starts[i+1], ends[i+1]
            if s_i1 - e_i > 0:
                if True:
                    starts.insert(i+1, e_i)
                    ends.insert(i+1, s_i1)
                    bbpeses.insert(i+1, [2])  # 2 is the end token (solely that one as there is no text in a gap (shouldn't be))
                else:
                    pass # TODO: expand the next text's audio's context's start to the beginning of the gap with the silence (required for being able to detect when the next text starts...)
        bbpes_list = [1] + [bbpe for bbpes in bbpeses for bbpe in bbpes]  # 1 is the start token (only one time at the very beginning)
        # BBPEs to tensor
        bbpes_tensor = torch.tensor(bbpes_list, dtype=torch.long)

        # Mask for the creation of the BlockDiagonalGappyKeysMask - START
        # q_lens - START
        q_lens = [len(bbpes) for bbpes in bbpeses]
        q_lens[0] += 1  # for the added start token
        q_lens[-1] -= 1  # for the removed end token (the very last one)
        # q_lens - END
        # kv_lens - START          # kv_lens = [(e-s)//80 for s,e in zip(starts, ends)]; wrong implementation, doesn't work !!!
        kv_lens = [int(e//80-s//80) for s,e in zip(starts, ends)]  # 80ms is the hop_length on the cnn output
        # kv_lens - END
        # kv_starts - START
        kv_starts = [int(s//80) for s in starts] + [int(ends[-1]//80)]
        # kv_starts - END

        #cross_mask = fmha.BlockDiagonalGappyKeysMask.from_seqlens(q_seqlen=q_lens, kv_seqstarts=kv_starts, kv_seqlen=kv_lens)
        cross_mask = fmha.BlockDiagonalMask.from_seqlens(q_seqlen=q_lens, kv_seqlen=kv_lens)
        # Mask for the creation of the BlockDiagonalGappyKeysMask - END

        # import matplotlib.pyplot as plt
        # plt.imshow(cross_mask.materialize([sum(q_lens), sum(kv_lens)]))
        # plt.show()

        # Audio
        try:
            duration = end_time - start_time
            audio_tensor = load_audio(audio_path, offset=start_time, duration=duration)
            return audio_tensor, bbpes_tensor, cross_mask
        except Exception as e:
            print(f'Error: {e}')
            return None