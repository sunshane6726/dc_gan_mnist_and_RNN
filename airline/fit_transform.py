# conducts data centering
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# we must use 2D array
my_array = [[1], [2], [3], [6]]
x = scaler.fit_transform(my_array)
print(x)
# converts scaled data back to original
y = scaler.inverse_transform(x)
print(y)