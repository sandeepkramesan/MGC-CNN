"""

Run instructions:
python3 cnn_fft.py /path/to/fft_extracted 


"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sklearn 
from sklearn import linear_model
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
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
=

	return np.asarray(X), np.asarray(y)


def cnn(X_train, y_train, X_test, y_test, genre_list):

    print("X_train = " + str(len(X_train)), "y_train = " + str(len(y_train)), "X_test = " + str(len(X_test)), "y_test = " + str(len(y_test)))
    model = Sequential()
    # convolutional layer
    print(X_train.shape)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
  
    model.add(Conv2D(64, kernel_size=(3,3), strides = 2, activation='relu',kernel_initializer='he_normal',padding='same',input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(.1))    

    model.add(Conv2D(256, kernel_size=(2,2),strides = 2, activation='relu',kernel_initializer='he_normal',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(.1))    


    # flatten output of conv
    model.add(Flatten())

    model.add(BatchNormalization())


    model.add(Dropout(.1))    

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(.5))    
    model.add(Dense(10, activation='softmax'))
    # compiling the sequential model
    #loss = categorical_crossentropy, hinge, mean_squared_error
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    model.fit(X_train, y_train,batch_size=50,epochs=25,verbose=1)#,validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test,batch_size=25)
    print('loss: ', loss, '\naccuracy: ', accuracy)


    joblib.dump(model, 'model.pkl')
    print("Model Saved\n")
    print(model.summary())    


def main():
    
    dir_fft = sys.argv[1]
    
    genre_list = os.listdir('./fft_extracted_L')
    
    X,y = read_fft(genre_list,dir_fft)

    X = X.reshape(10000,100,100,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

    cnn(X_train, y_train, X_test, y_test, genre_list)
    

if __name__ == "__main__":
	main()
