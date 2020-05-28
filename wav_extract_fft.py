"""
Run Instruction:
python3 wav_extract_fft.py
"""

import scipy
import scipy.io.wavfile
import os
import sys
import glob
import pathlib
import numpy as np

import warnings
warnings.filterwarnings('ignore')

genres = os.listdir('./converted')
for g in genres:
    pathlib.Path(f'fft_extracted/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./converted/{g}'):
        songname = f'./converted/{g}/{filename}'
        sample_rate, song_array = scipy.io.wavfile.read(songname)
        fft_features = abs(scipy.fft.fft(song_array[:10000]))
        #base_fn, ext = os.path.splitext(songname)
        #data_fn = base_fn + ".fft"
        np.save(f'fft_extracted/{g}/{filename[:-3].replace(".", "")}.fft', fft_features)
    
    



#if __name__ == "__main__":
#	main()
	
