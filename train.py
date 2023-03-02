# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from rawvae.model import VAE, loss_function
from rawvae.tests import init_test_audio
from rawvae.dataset import AudioDataset, ToTensor

import random
import numpy as np

import os, sys, argparse, time
from pathlib import Path

import librosa
import soundfile as sf
import configparser
import random
import json
import matplotlib.pyplot as plt
import pdb

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
args = parser.parse_args()

# Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
try: 
  config.read(config_path)
except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()

# Import audio configs 
sampling_rate = config['audio'].getint('sampling_rate')
hop_length = config['audio'].getint('hop_length')
segment_length = config['audio'].getint('segment_length')

# Dataset
dataset = Path(config['dataset'].get('datapath'))
if not dataset.exists():
  raise FileNotFoundError(dataset.resolve())

run_number = config['dataset'].getint('run_number')

my_audio = dataset / 'audio'

test_audio = config['dataset'].get('test_dataset')
dataset_test_audio = dataset / test_audio

if not dataset_test_audio.exists():
  raise FileNotFoundError(dataset_test_audio.resolve())

generate_test = config['dataset'].get('generate_test')    

# Training configs
epochs = config['training'].getint('epochs')
learning_rate = config['training'].getfloat('learning_rate')
batch_size = config['training'].getint('batch_size')
checkpoint_interval = config['training'].getint('checkpoint_interval')
save_best_model_after = config['training'].getint('save_best_model_after')

# Model configs
latent_dim = config['VAE'].getint('latent_dim')
n_units = config['VAE'].getint('n_units')
kl_beta = config['VAE'].getfloat('kl_beta')
device = config['VAE'].get('device')

# etc
example_length = config['extra'].getint('example_length')
normalize_examples = config['extra'].getboolean('normalize_examples')
plot_model = config['extra'].getboolean('plot_model')

desc = config['extra'].get('description')
start_time = time.time()
config['extra']['start'] = time.asctime( time.localtime(start_time) )

device = torch.device(device)
device_name = torch.cuda.get_device_name()
print('Device: {}'.format(device_name))
config['VAE']['device_name'] = device_name

# Create workspace
run_id = run_number
while True:
    try:
        my_runs = dataset / desc
        run_name = 'run-{:03d}'.format(run_id)
        workdir = my_runs / run_name 
        os.makedirs(workdir)

        break
    except OSError:
        if workdir.is_dir():
            run_id = run_id + 1
            continue
        raise

config['dataset']['workspace'] = str(workdir.resolve())

print("Workspace: {}".format(workdir))

# Create the dataset
print('creating the dataset...')
training_array = []
new_loop = True

for f in my_audio.glob('*.wav'): 
  print('adding-> %s' % f.stem)
  new_array, _ = librosa.load(f, sr=sampling_rate)

  if new_loop:
      training_array = new_array
      new_loop = False
  else:
      training_array = np.concatenate((training_array, new_array), axis=0)

total_frames = len(training_array) // segment_length
print('Total number of audio frames: {}'.format(total_frames))
config['dataset']['total_frames'] = str(total_frames)

# Create the dataset
training_dataset = AudioDataset(training_array, segment_length = segment_length, sampling_rate = sampling_rate, hop_size = hop_length, transform=ToTensor())
training_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle=True)

print("saving initial configs...")
config_path = workdir / 'config.ini'
with open(config_path, 'w') as configfile:
  config.write(configfile)

# Train
model_dir = workdir / "model"
checkpoint_dir = model_dir / 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

log_dir = workdir / 'logs'
os.makedirs(log_dir, exist_ok=True)

if generate_test:

  test_dataset, audio_log_dir = init_test_audio(workdir, test_audio, dataset_test_audio, sampling_rate, segment_length)
  test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

# Neural Network

model = VAE(segment_length, n_units, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Some dummy variables to keep track of loss situation

train_loss_prev = 1000000
best_loss = 1000000
final_loss = 1000000

for epoch in range(epochs):
  
  print('Epoch {}/{}'.format(epoch, epochs - 1))
  print('-' * 10)

  model.train()
  train_loss = 0
  
  for i, data in enumerate(training_dataloader):
    
    # data, = data
    data = data.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
  
  print('====> Epoch: {} - Total loss: {} - Average loss: {:.9f}'.format(
          epoch, train_loss, train_loss / len(training_dataset)))
  
  if epoch % checkpoint_interval == 0 and epoch != 0: 
    print('Checkpoint - Epoch {}'.format(epoch))
    state = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
    }
    
    if generate_test:
      
      init_test = True
      
      for iterno, test_sample in enumerate(test_dataloader):
        with torch.no_grad():
          test_sample = test_sample.to(device)
          test_pred = model(test_sample)[0]
        
        if init_test:
          test_predictions = test_pred
          init_test = False
        
        else:
          test_predictions = torch.cat(test_predictions, test_pred, 0)
        
      audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format( epoch))
      test_predictions_np = test_predictions.view(-1).cpu().numpy()
      sf.write( audio_out, test_predictions_np, sampling_rate)
      print('Audio examples generated: {}'.format(audio_out))
    
    torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epoch)))
  
    if (train_loss < train_loss_prev) and (epoch > save_best_model_after):
      
      save_path = workdir.joinpath('model').joinpath('best_model.pt')
      torch.save(model, save_path)
      print('Epoch {:05d}: Saved {}'.format(epoch, save_path))
      config['training']['best_epoch'] = str(epoch)
      best_loss = train_loss

    elif (train_loss > train_loss_prev):
      print("Average loss did not improve.")
  
  final_loss = train_loss

print('Last Checkpoint - Epoch {}'.format(epoch))
state = {
  'epoch': epoch,
  'state_dict': model.state_dict(),
  'optimizer': optimizer.state_dict()
}

if generate_test:
      
  init_test = True
  
  for iterno, test_sample in enumerate(test_dataloader):
    with torch.no_grad():
      test_sample = test_sample.to(device)
      test_pred = model(test_sample)[0]
  
    if init_test:
      test_predictions = test_pred
      init_test = False
    
    else:
      test_predictions = torch.cat(test_predictions, test_pred, 0)
    
  audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format(epochs))
  test_predictions_np = test_predictions.view(-1).cpu().numpy()
  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('Audio examples generated: {}'.format(audio_out))

  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('Last Audio examples generated: {}'.format(audio_out))

# Save the last model as a checkpoint dict
torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epochs)))

if train_loss > train_loss_prev:
  print("Final loss was not better than the last best model.")
  print("Final Loss: {}".format(final_loss))
  print("Best Loss: {}".format(best_loss))
  
  # Save the last model using torch.save 
  save_path = workdir.joinpath('model').joinpath('last_model.pt')
  torch.save(model, save_path)
  print('Training Finished: Saved the last model')

else:
  print("The last model is the best model.")

with open(config_path, 'w') as configfile:
  config.write(configfile)