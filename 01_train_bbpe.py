
# ByteLevelBPETokenizer
# https://huggingface.co/blog/how-to-train

import pandas as pd
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer


bbpe_num = 4096
text_filenames = ['commonvoice_en.txt'] * 1
text_filenames += ['commonvoice_es.txt'] * 3
text_filenames += ['commonvoice_hu.txt'] * 10
text_filenames += ['commonvoice_de.txt'] * 2

ds_path = Path('tokenizer/texts')
text_paths = [str(ds_path/text_filename) for text_filename in text_filenames]
tokenizer_path = Path(f'tokenizer/bbpe_{bbpe_num}')
vocab_path = tokenizer_path/'vocab.json'


if not vocab_path.exists():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=text_paths, vocab_size=bbpe_num, min_frequency=2, special_tokens=[
        "<pad>",
        "<s>",
        "</s>",
    ])

    tokenizer.save_model(str(tokenizer_path))