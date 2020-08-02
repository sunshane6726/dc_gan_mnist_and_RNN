from keras.utils import np_utils

import numpy as np

my_array = [0, 2, 1, 2, 0]
# one-hot encoding
one_hot = np_utils.to_categorical(my_array, 3)
print(one_hot)