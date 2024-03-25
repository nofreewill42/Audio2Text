
from pathlib import Path

import torch

from audio_processor import load_audio, AudioProcessor
from tokenizers import ByteLevelBPETokenizer

import edlib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

import json

config_path = Path('configs/causal.json')
config = json.loads(config_path.read_text())

# Model - START
from model_architecture.model import XModel

model = XModel(config)

mapping = {
    "cnnemb.conv1.0.weight": "cnnemb.convs.0.conv.weight",
    "cnnemb.conv1.0.bias": "cnnemb.convs.0.conv.bias",
    "cnnemb.conv1.1.weight": "cnnemb.convs.0.bn.weight",
    "cnnemb.conv1.1.bias": "cnnemb.convs.0.bn.bias",
    "cnnemb.conv1.1.running_mean": "cnnemb.convs.0.bn.running_mean",
    "cnnemb.conv1.1.running_var": "cnnemb.convs.0.bn.running_var",
    "cnnemb.conv1.1.num_batches_tracked": "cnnemb.convs.0.bn.num_batches_tracked",
    "cnnemb.conv2.0.weight": "cnnemb.convs.1.conv.weight",
    "cnnemb.conv2.0.bias": "cnnemb.convs.1.conv.bias",
    "cnnemb.conv2.1.weight": "cnnemb.convs.1.bn.weight",
    "cnnemb.conv2.1.bias": "cnnemb.convs.1.bn.bias",
    "cnnemb.conv2.1.running_mean": "cnnemb.convs.1.bn.running_mean",
    "cnnemb.conv2.1.running_var": "cnnemb.convs.1.bn.running_var",
    "cnnemb.conv2.1.num_batches_tracked": "cnnemb.convs.1.bn.num_batches_tracked",
    "cnnemb.conv3.0.weight": "cnnemb.convs.2.conv.weight",
    "cnnemb.conv3.0.bias": "cnnemb.convs.2.conv.bias",
    "cnnemb.conv3.1.weight": "cnnemb.convs.2.bn.weight",
    "cnnemb.conv3.1.bias": "cnnemb.convs.2.bn.bias",
    "cnnemb.conv3.1.running_mean": "cnnemb.convs.2.bn.running_mean",
    "cnnemb.conv3.1.running_var": "cnnemb.convs.2.bn.running_var",
    "cnnemb.conv3.1.num_batches_tracked": "cnnemb.convs.2.bn.num_batches_tracked",
}

step_numbers = [1000,2000,10000,25000,45000]
for step_number in step_numbers:
    from collections import OrderedDict
    old_state_dict = torch.load(f'model_weights/{config_path.stem}/model_{step_number}.pt', map_location=device)
    new_state_dict = OrderedDict()
    for old_key, value in old_state_dict.items():
        if old_key in mapping:
            new_key = mapping[old_key]
        else:
            new_key = old_key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    # save the new state dict
    torch.save(model.state_dict(), f'model_weights/{config_path.stem}/model_{step_number}_new.pt')
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
    print(result['editDistance']/len(text))
    print(None)
# Compare with ground truth - END