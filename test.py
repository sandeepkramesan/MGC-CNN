"""
Run Instruction:
python3 test.py ~/path/to/your/song/file
"""


import os	
import sys 
import joblib
import glob
import pathlib
import scipy
import scipy.io.wavfile
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import librosa


probs = []
X = []
y = []

genres = os.listdir('./fft_extracted')

test_file = sys.argv[1]

pathlib.Path(f'./testing{test_file[:-4]}').mkdir(parents=True, exist_ok=True)

os.system("ffmpeg -t 30 -i " + test_file + f' ./testing{test_file[:-4]}/song.wav')
 
sample_rate, song_array = scipy.io.wavfile.read(f'./testing{test_file[:-4]}/song.wav')
fft_features = abs(scipy.fft.fft(song_array[:10000]))

#np.save("testing.fft",fft_features)
#np.save(f'./test{test_file[:-4]}/song.fft', fft_features)
#fft_feature = np.load("testing.fft.npy")

#X,y = read_fft(genres,'./test')
for label,genre in enumerate(genres):
	y.append(label)
	
X.append(fft_features)
X = np.array(X)

X = X.reshape((-1,100,100,1))

clf = joblib.load('./model.pkl')

probs = clf.predict_proba(X)

print("\t".join(str(x) for x in genres))
print("\t".join(str("%.3f" % x) for x in probs[0]))

probs=probs[0]
max_prob = max(probs)


for i,j in enumerate(probs):
    if probs[i] == max_prob:
        max_prob_index=i
   
print(max_prob_index)
predicted_genre = genres[max_prob_index]
print("\n\npredicted genre = ",predicted_genre)
os.system("rm -r "+ f'testing*')



