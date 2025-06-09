import torch
import librosa
from torch.utils.data import IterableDataset

from pathlib import Path
import numpy as np
from itertools import chain, cycle

class IterableAudioDataset(IterableDataset):
    """
    This is the main class that calculates the streams CQT spectrogram frames

    # Iterable Dataset class structure source:
    Source: https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

    The current loading mechanism shuffles the audio file list, but not the audio windows. 
    
    """

    def __init__(self, audio_folder, sampling_rate, hop_size, dtype, device, shuffle = True):
        
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.dtype = dtype
        self.device = device
        self.shuffle = shuffle

        if isinstance(audio_folder, pathlib.PurePath):
            self.audio_folder = audio_folder
        else: 
            self.audio_folder = Path(audio_folder)

        self.audio_file_list = [f for f in audio_folder.glob('*.wav')]
        self.num_files = len(self.audio_file_list)

    @property
    def shuffled_data_list(self):
        # This is a workaround for shuffling. We only shuffle the audio file list
        # MAKE SURE THE DATALOADER IN THE TRAINING SCRIPT IS SHUFFLE FALSE.
        return random.sample(self.audio_file_list, len(self.audio_file_list))

    def process_data(self, audio_file):
        
        # torchaudio.load with multichannel. Let's use all channels as content for training
        audio_np, audio_sr = torchaudio.load(audio_file)

        # Check if the file sampling rate is different than the config sampling_rate. This is done because librosa loads slower if sr != None above.
        if audio_sr != self.sampling_rate:
            audio_np = torchaudio.functional.resample(audio_np, audio_sr, self.sampling_rate)

        # Pad if the length is not a multiplier of hop_size
        if audio_np.shape[1] % self.hop_size != 0:
            num_zeros = self.hop_size - (audio_np.shape[1] % self.hop_size)
            audio_np = torch.nn.functional.pad(audio_np, (0, num_zeros), 'constant')            
        
        # Check if we are using cuda then move the audio to cuda
        if self.device.type == "cuda":
            audio_np = audio_np.to(self.device)

        for frame in audio_np:
            yield frame

    def get_stream(self, audio_file_list):
        return chain.from_iterable(map(self.process_data, cycle(audio_file_list)))
        
    def __iter__(self):
        if self.shuffle:
            return self.get_stream(self.shuffled_data_list)
        else: 
            return self.get_stream(self.audio_file_list)

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
