
from typing import Union, Optional
from pathlib import Path
import wave
import librosa

import numpy as np
import torch


def get_audio_length(path: Union[str, Path]):
    '''
    Returns the length of the audio in seconds.
    '''
    wr = wave.open(str(path), 'r')
    n_frames = wr.getnframes()
    sr = wr.getframerate()
    return n_frames / sr

def load_audio(path: Union[str, Path], offset: Optional[float] = None, duration: Optional[float] = None):
    '''
    Loads audio from offset seconds to offset + duration seconds from wav file.
    '''
    wr = wave.open(str(path), 'r')
    n_frames = wr.getnframes()
    sr = wr.getframerate()
    if offset:
        offset = int(offset * sr)
        wr.setpos(offset)
    if duration:
        duration = int(duration * sr)
        n_frames = min(n_frames, duration)
    audio_bytes = wr.readframes(n_frames)
    audio_numpy = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_tensor = torch.from_numpy(audio_numpy).float() / 32768.0
    return audio_tensor


class AudioProcessor():
    #def __init__(self, sr=16000, n_mels=128, n_fft=512, hop_length=160, f_min=0.0, f_max=None):
    def __init__(self, config):

        # CONFIG - START
        audio_config = config['audio_processing']
        sr = audio_config['sr']
        n_mels = audio_config['n_mels']
        n_fft = audio_config['n_fft']
        hop_length = audio_config['hop_length']
        f_min = audio_config['f_min']
        f_max = audio_config.get('f_max', None)  # Use .get for optional fields

        self.window = torch.hann_window(n_fft)
        self.mel_basis = torch.from_numpy(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max)).float()

        # Normalize
        self.mean = torch.tensor(audio_config['mean'])
        self.std = torch.tensor(audio_config['std'])
        # self
        self.sr, self.n_mels, self.n_fft, self.hop_length, self.f_min, self.f_max = sr, n_mels, n_fft, hop_length, f_min, f_max
        # CONFIG - END
    
    def print_info(self):
        print(f'sr: {self.sr}')
        print(f'n_mels: {self.n_mels}')
        print(f'n_fft: {self.n_fft}')
        print(f'hop_length: {self.hop_length}')
        print(f'f_min: {self.f_min}')
        print(f'f_max: {self.f_max}')
        print(f'window: torch.hann_window(self.n_fft: {self.n_fft})')
        print(f'mel_basis: librosa.filters.mel(sr: {self.sr}, n_fft: {self.n_fft}, n_mels: {self.n_mels}, fmin: {self.f_min}, fmax: {self.f_max})')
        print(f'mean: {self.mean}')
        print(f'std: {self.std}')
        

    def process(self, audio_tensor: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
        log_mel_tensor = self.get_log_mel(audio_tensor)
        log_mel_lens = (audio_lens - self.n_fft) // self.hop_length + 1
        return log_mel_tensor, log_mel_lens


    def get_log_mel(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        device = audio_tensor.device
        if self.window.device != device:
            self.window, self.mel_basis = self.window.to(device), self.mel_basis.to(device)
            self.mean, self.std = self.mean.to(device), self.std.to(device)
        stft_tensor = torch.stft(audio_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True, center=False)
        #stft_tensor = torch.view_as_real(stft_tensor)  # (batch, n_frames, n_freqs[complex]) -> (batch, n_frames, n_freqs, 2)
        #power_tensor = torch.norm(stft_tensor, dim=-1)
        #power_tensor = stft_tensor.pow(2).sum(-1)
        power_tensor = stft_tensor.abs().pow(2.)
        mel_tensor = torch.matmul(self.mel_basis, power_tensor)
        log_mel_tensor = torch.log10(torch.clamp(mel_tensor, min=1e-10))
        # Normalize
        log_mel_tensor = (log_mel_tensor - self.mean) / self.n_mels / self.std
        return log_mel_tensor
    
    