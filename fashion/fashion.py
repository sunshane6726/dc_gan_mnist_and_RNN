# import packages

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers import Dense, MaxPool2D, Conv2D, InputLayer

####################
# DATABASE SETTING #
####################
# load DB and split into train vs. test
# fashion MNIST data is 28x28 gray-scale image
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
print('\n== DB SHAPING INFO ==')
print("X train shape:", X_train.shape)
print("Y train shape:", Y_train.shape)
print("X test shape:", X_test.shape)
print("Y test shape:", Y_test.shape)

# global constants and hyper-parameters
MY_EPOCH = 5
MY_BATCH = 50

# define labels

labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
'sandal', 'shirt', 'sneaker', 'bag', 'ankle-boot']

# display a sample image and label
print('\n== SAMPLE DATA (RAW) ==')
sample = Y_train[122]
print("This is a", labels[sample], sample)
#print(X_train[122])
#plt.imshow(X_train[122]) # dataset -> 흑백으로 해서 이진수를 그림으로 구현
#plt.show()

# data scaling into [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0
# reshaping before entering CNN
# one-hot encoding is used for the output
# we use channel-last ordering (keras default)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

#print('\n== DB SHAPING INFO ==')
#print("X train shape:", X_train.shape)
#print("Y train shape:", Y_train.shape)
#print("X test shape:", X_test.shape)
#print("Y test shape:", Y_test.shape)

###############################
# MODEL BUILDING AND TRAINING #
###############################
# build a keras sequential model of our CNN

# total parameter count formula
# = (filter_height * filter_width * input_channels + 1) * #_filters

model = Sequential()
model.add(InputLayer(input_shape = (28, 28, 1)))

# no. parameters = (2 x 2 x 1 + 1) x 32
model.add(Conv2D(32, kernel_size = (2, 2), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(padding = 'same', pool_size = (2, 2)))

# no. parameters = (2 x 2 x 32 + 1) x 64
model.add(Conv2D(64, kernel_size = (2, 2), padding = 'same', activation = 'relu'))
model.add(MaxPool2D(padding = 'same', pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()

# model training and saving
model.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = MY_EPOCH, batch_size = MY_BATCH, verbose = 1)

####################
# MODEL EVALUATION #
####################

# model evaluation
score = model.evaluate(X_test, Y_test, verbose = 1)
print('\nKeras CNN model loss = ', score[0])
print('Keras CNN model accuracy = ', score[1])

# testing the model after learning with a sample
sample = X_train[123]
sample = sample.reshape(1, 28, 28, 1)
pred = model.predict(sample)
print("\nPrediction after learning:", labels[np.argmax(pred)], "\n")