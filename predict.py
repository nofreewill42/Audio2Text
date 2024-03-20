
from pathlib import Path
import torch
from model_architecture.model import XModel
from tokenizers import ByteLevelBPETokenizer

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
model.load_state_dict(torch.load('model_weights/model_0.pt', map_location=device))
model = model.to(device=device, dtype=torch.bfloat16)
#model.eval()

# load audio
from audio_processor import load_audio, AudioProcessor

audio_proc = AudioProcessor()
audio_proc.print_info()

audio_path = Path('microphone.wav')

audio_tensor = load_audio(audio_path)
audio_tensor = audio_tensor.to(device)
audio_len = torch.tensor([len(audio_tensor)]).to(device)
audio_tensor = audio_tensor.unsqueeze(0)

# forward
with torch.no_grad():
    mels_tensor, mel_lens = audio_proc.process(audio_tensor, audio_len)
    mels_tensor = mels_tensor.to(dtype=torch.bfloat16)
    enc_out, enc_lens = model(mels_tensor, mel_lens)
    print(enc_out, enc_lens)
