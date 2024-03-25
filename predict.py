
from pathlib import Path
import torch
from model_architecture.model import XModel
from tokenizers import ByteLevelBPETokenizer

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

def ctc_to_timedbbpes(ctcs):
    bbpes = [ctcs[0]]
    starts = [0]
    ends = [1]
    for i, ctc in enumerate(ctcs[1:]):
        if ctc == bbpes[-1]:
            ends[-1] = i+1
        else:
            bbpes.append(ctc)
            starts.append(i)
            ends.append(i+1)
    return [(bbpe, start, end) for bbpe, start, end in zip(bbpes, starts, ends) if bbpe != 0]

def bbpes_to_timedwords(bbpes):
    '''
    bbpes: [(bbpe, start, end), ...]
    bbpe, start and end are ints
    start and end are in 80ms units
    '''
    chunk, start, end = bbpes[0]
    words = [[bbpe_tokenizer.decode([chunk]), start, end]]
    for bbpe, start, end in bbpes[1:]:
        chunk = bbpe_tokenizer.decode([bbpe])
        if chunk.startswith(' '):
            words.append([chunk, start, end])
        else:
            words[-1][0] += chunk
            words[-1][2] = end
    # to seconds
    for word in words:
        word[1] = round(word[1]*0.08, 2)
        word[2] = round(word[2]*0.08, 2)        
    return words



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer_path = Path('/home/nofreewill/Documents/Projects/Audio2Text_archived/tokenizer/')
bbpe_tokenizer = ByteLevelBPETokenizer.from_file(f'{tokenizer_path}/vocab.json', f'{tokenizer_path}/merges.txt')

n_bbpe = bbpe_tokenizer.get_vocab_size()
n_layers = 16
d_model = 1024
d_ff = 2048
n_heads = 16
window_size = 48
dropout = 0.0
model = XModel(n_bbpe, n_layers, d_model, d_ff, n_heads, window_size, dropout)
model.load_state_dict(torch.load('model_weights/model_45000.pt', map_location=device))
model = model.to(device=device, dtype=torch.bfloat16)
model.eval()

# load audio
from audio_processor import load_audio, AudioProcessor

audio_proc = AudioProcessor()
#audio_proc.print_info()

audio_path = Path('example_audios/ctcdecoder_documentation_read.wav')

audio_tensor = load_audio(audio_path)
audio_tensor = audio_tensor.to(device)
audio_len = torch.tensor([len(audio_tensor)]).to(device)
audio_tensor = audio_tensor.unsqueeze(0)

# forward
with torch.no_grad():
    import time
    start = time.time()
    mels_tensor, mel_lens = audio_proc.process(audio_tensor, audio_len)
    mels_tensor = mels_tensor.to(dtype=torch.bfloat16)
    enc_out, enc_lens, kv_caches = model(mels_tensor, mel_lens)
    #print(enc_out, enc_lens)
    print(time.time()-start)

    ctcs = enc_out.argmax(-1).squeeze(0).tolist()

bbpes = ctc_to_timedbbpes(ctcs)
timed_words = bbpes_to_timedwords(bbpes)


