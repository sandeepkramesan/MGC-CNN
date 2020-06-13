
import os
import pathlib
import warnings
import librosa
import numpy as np
warnings.filterwarnings('ignore')


genres = os.listdir('./gtzan')


def change_pitch_and_speed(data):
    y_pitch_speed = data.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac), np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed


def change_pitch(data, sr):
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), sr, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)
    return y_pitch

def value_aug(data):
    y_aug = data.copy()
    dyn_change = np.random.uniform(low=1.5, high=3)
    y_aug = y_aug * dyn_change
    return y_aug


def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise


def hpss(data):
    y_harmonic, y_percussive = librosa.effects.hpss(data.astype('float64'))
    return y_harmonic, y_percussive


def shift(data):
    return np.roll(data, 1600)


def stretch(data, rate=1):
    input_length = len(data)
    streching = librosa.effects.time_stretch(data, rate)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching

def change_speed(data):
    y_speed = data.copy()
    speed_change = np.random.uniform(low=0.9, high=1.1)
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed
    
    
def main():
    print('Augmentation')
    for g in genres:
        pathlib.Path(f'augmented/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'./gtzan/{g}'):
            songname = f'./gtz/an{g}/{filename}'
            #print(filename,songname)
            y, sr = librosa.load(songname)
            data_noise = add_noise(y)
            data_roll = shift(y)
            data_stretch = stretch(y)
            pitch_speed = change_pitch_and_speed(y)
            pitch = change_pitch(y,sr)
            speed = change_speed(y)
            value = value_aug(y)
            y_harmonic, y_percussive = hpss(y)
            y_shift = shift(y)
            
            save_path = "./augmented/" + g #os.path.join(songname.split(g + '.')[0])
            save_name =  g + '.'+songname.split(g + '.')[1]
            print(save_path)
            
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'a.wav')), data_noise,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'b.wav')), data_roll,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'c.wav')), data_stretch,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'd.wav')), pitch_speed,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'e.wav')), pitch,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'f.wav')), speed,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'g.wav')), value,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'h.wav')), y_percussive,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'i.wav')), y_shift,
                                     sr)
            librosa.output.write_wav(os.path.join(save_path, save_name.replace('.au', 'j.wav')),y,
            			     sr)
            
            #os.system("rm "+ f'songname')
                
            print('finished')
            
if __name__ == '__main__':
    main()
