# rawaudiovae

Tested with torch 2.0.1.

Check out tutorial.ipynb for testing a trained model. You can download a dataset with trained models here: 
    https://drive.google.com/file/d/1e_X2Ir26iypSdSa6pRCJBy2q5t9zXFBb

Please download this folder and unzip it under the ./content folder

train.py script is for training a rawaudiovae, using a dataset folder. The dataset folder is a simple folder including a folder titled "audio", where the wav files should be. 

default.ini is a config file, and hyperparameter can be adjusted using the config file. 

The tutorial.ipynb uses a trained self-organizing map as well. The training of SOM can be found in another repo: https://github.com/ktatar/mlaudiosalad

This code is built within the following research residency:
https://kivanctatar.com/Coding-Latent-No-1

We have a paper on this work published at the Sound and Music Conference 2023: https://arxiv.org/pdf/2305.15571

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program – Humanities and Society (WASP-HS) funded by the Marianne and Marcus Wallenberg Foundation and the Marcus and Amalia Wallenberg Foundation.
