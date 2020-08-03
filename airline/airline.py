# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import InputLayer, Dense
from keras.layers import LSTM, SimpleRNN
from keras.models import Sequential

# global constants and hyper-parameters
MY_PAST = 12 # 개월 수 여행객만 측정 년도 측정x 년도 측정하려면 144가 있어야함 또는 축이 2개
MY_SPLIT = 0.8
MY_NEURON = 3000
MY_SHAPE = (MY_PAST, 1)
MY_EPOCH = 100
MY_BATCH = 64

####################
# DATABASE SETTING #
####################

# passenger data stored in a file  파일에 저장된 승객 데이터
# we process the data and build training/test sets ourselves 데이터를 처리하고 교육 / 테스트 세트를 직접 작성합니다.
# raw data will be normalized for better learning 더 나은 학습을 위해 원시 데이터가 정규화됩니다
df = pd.read_csv('airline.csv', usecols = [1])

# convert panda data format to numpy array
raw_DB = np.array(df).astype(float)
print('== DB INFO (RAW) ==')
print('DB shape: ', raw_DB.shape)
print('All data before transformation:')
print(raw_DB.flatten())

# [0, 1] data normalization
# "scaler" will be used again later for inverse transformation
scaler = MinMaxScaler()
raw_DB = scaler.fit_transform(raw_DB)
print('\nAll data after transformation:')
print(raw_DB.flatten())

# slice data into groups of 13
# ex: 0-13, 1-14, 2-15, ... 130-143
data = []
for i in range(len(raw_DB) - MY_PAST):
    data.append(raw_DB[i: i + MY_PAST + 1])

reshaped_data = np.array(data)
np.random.shuffle(reshaped_data)

#print('\n== DB INFO (SPLIT) ==')
#print('DB shape:', reshaped_data.shape)
#print('\nGroup 0:\n', data[0])
#print('\nGroup 1:\n', data[1])

# slicing between input and output
# use index 0 - 11 (12 months) for input and index 12 for output
X_data = reshaped_data[:, 0:MY_PAST]
Y_data = reshaped_data[:, -1]
# split between training and test sets
split_boundary = int(reshaped_data.shape[0] * MY_SPLIT)
X_train = X_data[: split_boundary]
X_test = X_data[split_boundary:]
Y_train = Y_data[: split_boundary]
Y_test = Y_data[split_boundary:]

# print shape information
print('\n== DB SHAPE INFO ==')
print('X_train shape = ', X_train.shape)
print('X_test shape = ', X_test.shape)
print('Y_train shape = ', Y_train.shape)
print('Y_test shape = ', Y_test.shape)

###############################
# MODEL BUILDING AND TRAINING #
###############################

# keras sequential model for RNN
# RNN is unrolled 12 times to accept 12 passenger data
model = Sequential()

# input_shape needs 2D: (time_steps, seq_len)
# size of hidden layer in LSTM cell is MY_NEURON
# lastly, we form a fully connected layer for output
model.add(InputLayer(input_shape = MY_SHAPE))
model.add(LSTM(MY_NEURON))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

# model training and saving
model.compile(loss= 'mape', optimizer='rmsprop', metrics = ['acc'])
model.fit(X_train, Y_train, epochs = MY_EPOCH, batch_size = MY_BATCH, verbose = 1)
model.save('chap_airplane.h5')

####################
# MODEL EVALUATION #
####################
# model evaluation

score = model.evaluate(X_test, Y_test, verbose = 1)
print('\nKeras RNN model loss = ', score[0])
print('Keras RNN model accuracy = ', score[1])

# transform [0, 1] values back to the original range
# we need to keep the "scaler" used for initial transformation
# because "scaler" remembers mean and dev
predict = model.predict(X_test)
predict = scaler.inverse_transform(predict)
Y_test = scaler.inverse_transform(Y_test)

# plot predicted vs real output
plt.plot(predict, 'r:')
plt.plot(Y_test, 'g-')
plt.legend(['predict', 'true'])
plt.show()