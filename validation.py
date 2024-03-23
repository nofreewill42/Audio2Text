
from pathlib import Path

import torch

from audio_processor import load_audio, AudioProcessor
from tokenizers import ByteLevelBPETokenizer

import edlib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Model - START
from model_architecture.model import XModel

n_bbpe = 4096
n_layers = 16
d_model = 1024
d_ff = 2048
n_heads = 16
window_size = 48
dropout = 0.1
model = XModel(n_bbpe, n_layers, d_model, d_ff, n_heads, window_size, dropout)
model.load_state_dict(torch.load('model_weights/model_45000.pt', map_location=device))
model = model.to(device=device, dtype=torch.bfloat16)
model.eval()
# Model - END

# Tokenizer - START
tokenizer_path = Path('/home/nofreewill/Documents/Projects/Audio2Text_archived/tokenizer/')
bbpe_tokenizer = ByteLevelBPETokenizer.from_file(f'{tokenizer_path}/vocab.json', f'{tokenizer_path}/merges.txt')
# Tokenizer - END

# Load audio - START
audio_proc = AudioProcessor()
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
    mels_tensor = mels_tensor.to(dtype=torch.bfloat16)
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
    print(None)
# Compare with ground truth - END