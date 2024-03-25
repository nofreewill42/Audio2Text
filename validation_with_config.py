
from pathlib import Path

import torch

from audio_processor import load_audio, AudioProcessor
from tokenizers import ByteLevelBPETokenizer

import edlib

from pathlib import Path
import json
import torch
from audio_processor import AudioProcessor, load_audio
from tokenizers import ByteLevelBPETokenizer
from model_architecture.model import XModel

from utils import load_model


# Usage example
config_name = "encoder_only_causal"#offline"#
dtype = torch.bfloat16
step_number = 200#1000 #45000
audio_proc, model, bbpe_tokenizer, device = load_model(config_name, step_number, dtype=dtype)
#model.eval()
print(f'Using device: {device}')
audio_proc.print_info()

audio_path = Path('example_audios/validation.wav')
text_path = Path('example_audios/validation.txt')

audio_tensor = load_audio(audio_path)
audio_tensor = audio_tensor.to(device)
audio_len = torch.tensor([len(audio_tensor)]).to(device)
audio_tensor = audio_tensor.unsqueeze(0)
# Load audio - END

# Forward - START
with torch.no_grad():
    import time
    start = time.time()
    mels_tensor, mel_lens = audio_proc.process(audio_tensor, audio_len)
    mels_tensor = mels_tensor.to(dtype=dtype)
    enc_out, enc_lens, kv_caches = model(mels_tensor, mel_lens)
    print(enc_out, enc_lens)
    print(time.time()-start)

    ctcs = enc_out.argmax(-1).squeeze(0).tolist()
# Forward - END

# CTC to BBPEs - START
def ctc_to_bbpes(ctcs):
    '''[0 0 878 878 0 835 0 0 0] -> [0 878 0 835 0] -> [878 835]'''
    bbpes = [ctcs[0]]
    # remove consecutive duplicates
    for i in range(1, len(ctcs)):
        if ctcs[i] != ctcs[i-1]:
            bbpes.append(ctcs[i])
    # remove zeros
    bbpes = [bbpe for bbpe in bbpes if bbpe != 0]
    return bbpes

bbpes = ctc_to_bbpes(ctcs)
print(bbpe_tokenizer.decode(bbpes))
# CTC to BBPEs - END

# Compare with ground truth - START
with open(text_path, 'r') as f:
    text = f.read().strip()
    predicted_text = bbpe_tokenizer.decode(bbpes)
    
    result = edlib.align(text, predicted_text, task='path')
    nice = edlib.getNiceAlignment(result, text, predicted_text)
    print(result['editDistance']/len(text))
    print(None)
# Compare with ground truth - END