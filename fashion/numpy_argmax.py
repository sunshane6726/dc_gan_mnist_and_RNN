import numpy as np

my_array = [[10, 14, 12], [13, 11, 15]]

# flatten the array
print(np.argmax(my_array))
# axis = 0 means along the column
print(np.argmax(my_array, axis = 0))
# axis = 1 means along the row
print(np.argmax(my_array, axis = 1))