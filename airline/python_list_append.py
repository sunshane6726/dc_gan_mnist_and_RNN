# attach a new element at the end of python list
import numpy as np
my_data = np.array([3, 5, 1, 7, 2])
time_step = 2
new = []
# extract data
for i in range(len(my_data) - time_step):
    new.append(my_data[i: i + time_step + 1])

print(np.array(new))