"""

Run instructions:
python3 cnn_mfcc.py /path/to/fft_extracted 


"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sklearn 
from sklearn import linear_model
#from sklearn.externals 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import scipy
import sys
import glob
import numpy as np
from keras.utils import to_categorical



"""reads FFT-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required FFT-files
base_dir must contain genre_list of directories
"""
def read_fft(genre_list, base_dir):
	X = []
	y = []
	for label, genre in enumerate(genre_list):
		# create UNIX pathnames to id FFT-files.
		genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		# get path names that math genre-dir
		file_list = glob.glob(genre_dir)
		#print(file_list)
		for file in file_list:
			fft_features = np.load(file)
			X.append(fft_features)
			y.append(label)
	
	#print(X[0])
	#print(len(X))
	#print(len(y))

	return np.asarray(X), np.asarray(y)


"""reads MFCC-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required MFCC-files
base_dir must contain genre_list of directories
"""
def read_ceps(genre_list, base_dir):
	X, y = [], []
	for label, genre in enumerate(genre_list):
		for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
			ceps = np.load(fn)
			num_ceps = len(ceps)
			#print(num_ceps)			
			X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
			#X.append(ceps)
			y.append(label)
	
	#print(ceps)
	#print(X[0])
	#print(y)
	return np.array(X), np.array(y)


def cnn(X_train, y_train, X_test, y_test, genre_list):

    print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
    model = Sequential()
    # convolutional layer
    print(X_train.shape)
    #print(y_train.shape)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #print(y_train.shape)
    #print("1")
    #model.add(Conv1D(256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(8, kernel_size=(3,3), strides=2, activation='sigmoid',padding='same',input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, kernel_size=(3,3), strides=2, activation='sigmoid',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Conv2D(32, kernel_size=(3,3), strides=2, activation='relu',padding='same'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    # flatten output of conv
    #print("3")
    model.add(Flatten())
    #model.add(Dropout(.1))    
    # hidden layer
    #print("4")
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    # output layer
    #print("5")
    model.add(Dense(10, activation='softmax'))
    # compiling the sequential model
    #loss = categorical_crossentropy, hinge, mean_squared_error
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    # training the model for 10 epochs
    #print("6")
    #X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1]), X_train.shape[2])
    #X_train = X_train.transpose()

    print(X_train.shape)
    model.fit(X_train, y_train,batch_size=50,epochs=500)
    test_loss, test_acc = model.evaluate(X_test,y_test,batch_size=1)
    print('test_acc: ',test_acc)	

    joblib.dump(model, 'model.pkl')
    print("Model Saved\n")
    print(model.summary())    


def main():
	# first command line argument is the base folder that consists of the fft files for each genre
	base_dir_fft  = sys.argv[1]
	# second command line argument is the base folder that consists of the mfcc files for each genre
	#base_dir_mfcc = sys.argv[1]
	
	"""list of genres (these must be folder names consisting .wav of respective genre in the base_dir)
	Change list if needed.
	"""
	genre_list = os.listdir('./fft_extracted')
  #print(genre_list)
	
	#genre_list = ["classical", "jazz"] IF YOU WANT TO CLASSIFY ONLY CLASSICAL AND JAZZ

	# use FFT
	X, y = read_fft(genre_list, base_dir_fft)
	print(X.shape)
	X = X.reshape(1000,100,100,1)
	#print(X.shape)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

	# print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
	
	print('\n******USING FFT******')
	cnn(X_train, y_train, X_test, y_test, genre_list)
	print('*********************\n')

	# use MFCC
	#X, y = read_ceps(genre_list, base_dir_mfcc)
	#print(X.shape)
	#X = X.reshape(800,1,1)
	#print(X.shape)
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	
	
	#print('******USING MFCC******')
	#cnn(X_train, y_train, X_test, y_test, genre_list)
	#print('*********************')
	

if __name__ == "__main__":
	main()
