
from pathlib import Path
import torch
from model_architecture.model import XModel
from tokenizers import ByteLevelBPETokenizer

def ctc_to_bbpes(ctcs):
    '''0 0 878 878 0 835 0 0 0 -> 0 878 0 835 0 -> 878 835'''
    bbpes = [ctcs[0]]
    # remove consecutive duplicates
    for i in range(1, len(ctcs)):
        if ctcs[i] != ctcs[i-1]:
            bbpes.append(ctcs[i])
    # remove zeros
    bbpes = [bbpe for bbpe in bbpes if bbpe != 0]
    return bbpes


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
audio_proc.print_info()


# get audio bytes from client through socket
import socket

HOST = '127.0.0.1'
PORT = 65432

# create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# bind the socket to the address and port
server_socket.bind((HOST, PORT))
# listen for incoming connections
server_socket.listen()

print(f"Server listening on {HOST}:{PORT}")


import numpy as np
audio_data = torch.tensor([], dtype=torch.float32, device=device).unsqueeze(0)
n_cnn_processed = 0
all_ctcs = []
text = ''
kv_caches = None
# accept a connection
client_socket, addr = server_socket.accept()
print(f"Connection from {addr}")
with client_socket:
    while True:
        # receive data from client
        audio_bytes = client_socket.recv(4096)  # 8192 bytes is 4096 audio samples is 256ms
        #print(f"Received {len(audio_bytes)} bytes")
        if not audio_bytes:
            break
        # convert bytes to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_tensor = torch.from_numpy(audio_np).float() / 32768.0
        audio_tensor = audio_tensor.to(device).unsqueeze(0)
        # append to audio_data
        audio_data = torch.cat((audio_data, audio_tensor), dim=1)
        audio_len = torch.tensor([len(audio_data)]).long().to(device)
        mels_tensor, mel_lens = audio_proc.process(audio_data, audio_len)
        mels_tensor = mels_tensor.to(dtype=torch.bfloat16)

        # mels_tensor.shape[-1] to be divisible by 8 (cnn stride)
        if mels_tensor.shape[-1] % 8 != 0:
            mels_tensor = mels_tensor[..., :-(mels_tensor.shape[-1] % 8)]
            mel_lens = mels_tensor.shape[-1]
        
        if mels_tensor.shape[-1] - n_cnn_processed*8 < 16: continue

        # forward
        with torch.no_grad():
            enc_out, enc_lens, kv_caches = model(mels_tensor, mel_lens, kv_caches, n_cnn_processed)
            ctcs = enc_out.argmax(-1).squeeze(0).tolist()

        all_ctcs += ctcs
        n_cnn_processed += enc_out.shape[1]

        # decode bbpes
        bbpes = ctc_to_bbpes(all_ctcs)
        current_text = bbpe_tokenizer.decode(bbpes)
        if len(current_text) > len(text):
            print(current_text[len(text):], end='', flush=True)
            text = current_text