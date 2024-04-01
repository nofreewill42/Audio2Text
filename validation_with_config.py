
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
step_number = 46473 #45000
audio_proc, model, bbpe_tokenizer, device = load_model(config_name, step_number, dtype=dtype)
#model.eval()
print(f'Using device: {device}')
audio_proc.print_info()


validation_name = 'ctcdecoder_documentation_read'
audio_path = Path(f'example_audios/{validation_name}.wav')
text_path = Path(f'example_audios/{validation_name}.txt')

# Load audio - START
audio_tensor = load_audio(audio_path)
audio_tensor = audio_tensor.to(device)
audio_len = torch.tensor([len(audio_tensor)]).to(device)
audio_tensor = audio_tensor.unsqueeze(0)
# Load audio - END

# Load text - START
text = text_path.read_text().strip()
bbpes_list = [1] + bbpe_tokenizer.encode(text).ids
bbpes_tensor = torch.tensor(bbpes_list).long().unsqueeze(0).to(device)

# Forward - START
with torch.no_grad():
    import time
    start = time.time()
    mels_tensor, mel_lens = audio_proc.process(audio_tensor, audio_len)
    mels_tensor = mels_tensor.to(dtype=dtype)
    dec_out, enc_out, enc_lens, kv_caches = model(bbpes_tensor, mels_tensor, mel_lens)
    #print(enc_out, enc_lens)
    print(time.time()-start)

    print('dec_out:', dec_out.shape)
    print((dec_out.argmax(-1)[:,:-1] == bbpes_tensor[:,1:]).sum()/bbpes_tensor.shape[1])

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
predicted_text = bbpe_tokenizer.decode(bbpes)
print(predicted_text)
# CTC to BBPEs - END
    
# Decode step by step - START
initial_text = ' The online decoder'
bbpes = bbpe_tokenizer.encode(initial_text).ids

enc_out, enc_lens, kv_caches = model.encoder_forward(mels_tensor, mel_lens)
tgt = torch.ones(1, 1).long().to(mels_tensor.device)
#tgt = torch.cat((tgt, torch.tensor(bbpes).long().unsqueeze(0).to(mels_tensor.device)), dim=1)
dec_kv_caches = None
for j in range(1000):
    if j == 0:
        last_bbpe = tgt
    else:
        last_bbpe = tgt[:, -1:]
    dec_out, dec_kv_caches = model.decoder_forward(last_bbpe, kv_caches, dec_kv_caches)
    dec_out = dec_out[:,-1:].argmax(-1)
    print(bbpe_tokenizer.decode(dec_out.squeeze(0).tolist()), end='')
    tgt = torch.cat((tgt, dec_out), dim=1)

    if dec_out[-1] == 2:
        break

decoded_text = bbpe_tokenizer.decode(tgt.squeeze(0).tolist())




# Compare with ground truth - START
with open(text_path, 'r') as f:
    text = f.read()
    
    result = edlib.align(text, predicted_text, task='path')
    nice = edlib.getNiceAlignment(result, text, predicted_text)
    print()
    print('CTC')
    print(result['editDistance']/len(text))

    result = edlib.align(text, decoded_text, task='path')
    nice = edlib.getNiceAlignment(result, text, decoded_text)
    print('Dec')
    print(result['editDistance']/len(text))
    print(None)
# Compare with ground truth - END