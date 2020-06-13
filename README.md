## Music Genre Classification using CNN

#### Dataset Heirarchy
```
gtzan_dataset
	\blues
	\classical
	\country
	\disco
	\hiphop
	\jazz
	\metal
	\pop
	\reggae
	\rock
```

#### Run order of files:

```
python3 data_augmentation
```
Increases the size of dataset from 1000 to 10000.  
```
python3 wav_convert.py
```
Converts music files to the needed .wav format and store into ./converted.  

```
python3 wav_extract_fft.py
```
Extracts fft values and saves numpy files to ./fft_extracted.  

```
python3 cnn_fft.py fft_extracted/
```
Applies Logistic Regression to train and classify; gives prediction accuracy, saves confusion matrix image and model.  

```
python3 test.py /home/path/to/your/song/file
```
Loads saved model to predict genre of new file.  
