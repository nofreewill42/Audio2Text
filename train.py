'''
{
    "the-talmud/video-16b-mah-hatzad-applying-a-rabbinic-logical-tool-NvAdj": [
        "Javanese.srt",
        "ptPt.srt",
        "Spanish.srt",
        "English.srt",
        "Russian.srt",
        "Romanian.srt",
        "French.srt"
    ],
    "the-talmud/video-23-parallel-narratives-pngb7": [
        "Javanese.srt",
        "ptPt.srt",
        "Spanish.srt",
        "English.srt",
        "Russian.srt",
        "Romanian.srt",
        "French.srt"
    ],'''

import json
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data_loader import CourseraDataset, collate_fn
from audio_processor import AudioProcessor

from utils import load_model


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_dtype = None
    config_name = 'encoder_only_causal'#offline'
    audio_proc, model, bbpe_tokenizer, device = load_model(config_name, device=device, dtype=torch_dtype)
    print(f"Device: {device}")

    # load model statedict
    #state_dict_path = Path('model_weights/encoder_only_causal/model_45000.pt')
    state_dict_path = Path('model_weights/encoder_only_offline/model_1000.pt')
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    print('Dataset - START')
    tokenizer_path = Path(f'tokenizer/bbpe_{model.n_bbpe}')
    ds_path = Path('/media/nofreewill/8TB-SSD/Audio/Coursera/courses/')
    langs_dict_path = Path('/home/nofreewill/Documents/Projects/PrepareCoursera/langs_dict.json')
    langs_dict = json.loads(langs_dict_path.read_text())
    episodes_list = [k for k,v in langs_dict.items() if v == 'en']
    
    
    ds = CourseraDataset(ds_path, episodes_list, tokenizer_path, T=240.000)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    print('Dataset - END')

    print("Training - START")
    lr = 1e-4
    n_grad_acc = 32
    epochs_num = 1#0
    total_steps = len(dl)*epochs_num
    weight_decay = 1e-4
    pct_start = 0.1/epochs_num
    ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay)
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                    total_steps=total_steps + 2,
                                                    div_factor=25, final_div_factor=1e4, pct_start=pct_start)
    scaler = torch.cuda.amp.GradScaler()

    # load last saved model and start from there
    save_every_step = 1000
    model_save_path = Path(f'model_weights/{config_name}')
    model_save_path.mkdir(parents=True, exist_ok=True)

    model_laststep_number = max([int(f.stem.split('_')[-1]) for f in model_save_path.iterdir() if f.is_file()])
    if model_laststep_number > 0:
        print(f'Loading model from {model_save_path}/model_{model_laststep_number}.pt')
        model.load_state_dict(torch.load(f'{model_save_path}/model_{model_laststep_number}.pt'))
        # set lr_sched to last step
        model_laststep_number = 4000  # delete this line RIGHT NOW!!!
        print(f'Resuming training from last step {model_laststep_number * n_grad_acc}')
        for _ in tqdm(range(model_laststep_number * n_grad_acc)):
            lr_sched.step()
        start_epoch = model_laststep_number // len(dl)
    # model_laststep_number = 0
    # start_epoch = 0

    # Stats
    import time
    stats_path = Path(f'training_stats/{time.strftime("%Y%m%d_%H%M%S")}')
    stats_path.mkdir(parents=True, exist_ok=True)
    train_csv_path = Path(f'{stats_path}/train.csv')
    w_trn = open(str(train_csv_path), 'w', buffering=1)

    # write hyperparameters in first line
    w_trn.write(f'''TRAINING:\n
                    epochs_num,{epochs_num}\n
                    n_grad_acc,{n_grad_acc}\n
                    model_laststep_number,{model_laststep_number}\n
                    start_epoch,{start_epoch}\n
                    lr,{lr}\n
                    weight_decay,{weight_decay}\n
                    MODEL:\n
                    pct_start,{pct_start}\n
                    config_name,{config_name}\n
                    ''')

    
    step_counter = 0

    losses = []
    n_acc_losses = 100

    try:
        for epoch_num in range(epochs_num):
            if epoch_num < start_epoch:
                continue
            pbar = tqdm(dl)
            for i, batch in enumerate(pbar):
                if batch == None:
                    continue
                audios_tensor, audio_lens, bbpes_tensor, bbpe_lens = batch
                # Move to device
                audios_tensor, audio_lens = audios_tensor.to(device), audio_lens.to(device)
                bbpes_tensor, bbpe_lens = bbpes_tensor.to(device), bbpe_lens.to(device)
                input_tensor = bbpes_tensor[:, :-1]
                target_tensor = bbpes_tensor[:, 1:]
                with torch.no_grad():
                    log_mels_tensor, log_mel_lens = audio_proc.process(audios_tensor, audio_lens)
                    log_mels_tensor = log_mels_tensor.to(dtype=torch_dtype)
                
                # Forward
                with torch.cuda.amp.autocast(torch_dtype == None):
                    enc_out, enc_lens, kv_caches = model(log_mels_tensor, log_mel_lens)
                    enc_log_probs = torch.log_softmax(enc_out, dim=-1)#.float()  # "ctc_loss_cuda" not implemented for 'BFloat16'
                    ctc_tgt = bbpes_tensor.clone()
                    ctc_tgt[(ctc_tgt==1) | (ctc_tgt==2)] = 0
                    enc_loss = ctc_loss_fn(enc_log_probs.transpose(0,1), ctc_tgt, enc_lens, bbpe_lens)  # ctclossfn is wrong so correct it here
                    loss = enc_loss

                    losses.append(loss.item())
                    losses = losses[-n_acc_losses:]
                    avg_loss = sum(losses)/len(losses)
                    if loss.item() > 2*avg_loss:
                        print(f'Loss too high: {loss.item()}, avg_loss: {avg_loss}')
                        # remove last
                        #losses.pop()
                        continue
                    
                    loss = loss / n_grad_acc
                # Backward
                scaler.scale(loss).backward()
                if (i+1) % n_grad_acc == n_grad_acc-1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if step_counter % save_every_step == 0:
                        torch.save(model.state_dict(), f'{model_save_path}/model_{step_counter}.pt')
                    step_counter += 1
                lr_sched.step()

                # Record
                pbar.set_description(f'epoch: {epoch_num}, enc: {enc_loss.item():.2f}, lr: {lr_sched.get_last_lr()[0]:.4e}')
                w_trn.write(f'{epoch_num},{i},{lr_sched.get_last_lr()[0]:.4e},{enc_loss.item():.4f},{audios_tensor.shape[1]},{bbpes_tensor.shape[1]}\n')
                w_trn.flush()
            
            pbar.close()

        w_trn.close()
    # keyboard interrupt
    except KeyboardInterrupt:
        # save model
        torch.save(model.state_dict(), f'{model_save_path}/model_{step_counter}.pt')
        w_trn.close()
        print(f'Saved model at {model_save_path}/model_{step_counter}.pt')
        print(f'Saved training stats at {train_csv_path}')


        
    print("Training - END")

    
    print(None)