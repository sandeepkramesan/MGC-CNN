"""
Run instructions:
python wav_extract_mfcc.py

"""

import librosa
#import scipy.io.wavfile
import os
import sys
import glob
import pathlib
import numpy as np

import warnings
warnings.filterwarnings('ignore')

genres = os.listdir('./converted/training_data')
for g in genres:
    pathlib.Path(f'mfcc_extracted/training_data/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./converted/training_data/{g}'):
        songname = f'./converted/training_data/{g}/{filename}'
        #sample_rate, song_array = scipy.io.wavfile.read(songname)
        songarray, sr = librosa.load(songname, mono=True, duration=30)
        ceps = librosa.feature.mfcc(y = songarray, sr = sr )
        np.save(f'mfcc_extracted/training_data/{g}/{filename[:-3].replace(".", "")}.ceps', ceps)
    
    
for g in genres:
    pathlib.Path(f'mfcc_extracted/test_data/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./converted/test_data/{g}'):
        songname = f'./converted/test_data/{g}/{filename}'
        #sample_rate, song_array = scipy.io.wavfile.read(songname)
        songarray, sr = librosa.load(songname, mono=True, duration=30)
        ceps = librosa.feature.mfcc(y = songarray, sr = sr )
        np.save(f'mfcc_extracted/test_data/{g}/{filename[:-3].replace(".", "")}.ceps', ceps)
    
    
