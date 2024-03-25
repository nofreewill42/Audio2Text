import json
from pathlib import Path

import torch

from tokenizers import ByteLevelBPETokenizer

from model_architecture.model import XModel
from audio_processor import AudioProcessor

def load_model(config_name, step_number=None, device=None, dtype=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    config_path = Path(f'configs/{config_name}.json')
    config = json.loads(config_path.read_text())

    # Initialize model
    model = XModel(config)
    if step_number is not None:
        state_dict = torch.load(f'model_weights/{config_name}/model_{step_number}.pt', map_location=device)
        model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype)  # None is okay

    # Initialize tokenizer
    tokenizer_path = Path(f'tokenizer/bbpe_{config["n_bbpe"]}')
    bbpe_tokenizer = ByteLevelBPETokenizer.from_file(f'{tokenizer_path}/vocab.json', f'{tokenizer_path}/merges.txt')

    # Initialize audio processor
    audio_proc = AudioProcessor(config)

    return audio_proc, model, bbpe_tokenizer, device