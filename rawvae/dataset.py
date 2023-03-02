import torch
import librosa

from pathlib import Path
import numpy as np

class AudioDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, audio_np, segment_length, sampling_rate, hop_size, transform=None):
        
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.hop_size = hop_size
        
        if segment_length % hop_size != 0:
            raise ValueError("segment_length {} is not a multiple of hop_size {}".format(segment_length, hop_size))

        if len(audio_np) % hop_size != 0:
            num_zeros = hop_size - (len(audio_np) % hop_size)
            audio_np = np.pad(audio_np, (0, num_zeros), 'constant', constant_values=(0,0))

        self.audio_np = audio_np
        
    def __getitem__(self, index):
        
        # Take segment
        seg_start = index * self.hop_size
        seg_end = (index * self.hop_size) + self.segment_length
        sample = self.audio_np[ seg_start : seg_end ]
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return (len(self.audio_np) // self.hop_size) - (self.segment_length // self.hop_size) + 1

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)

class TestDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, audio_np, segment_length, sampling_rate, transform=None):
        
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        
        if len(audio_np) % segment_length != 0:
            num_zeros = segment_length - (len(audio_np) % segment_length)
            audio_np = np.pad(audio_np, (0, num_zeros), 'constant', constant_values=(0,0))

        self.audio_np = audio_np
        
    def __getitem__(self, index):
        
        # Take segment
        seg_start = index * self.segment_length
        seg_end = (index * self.segment_length) + self.segment_length
        sample = self.audio_np[ seg_start : seg_end ]
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.audio_np) // self.segment_length