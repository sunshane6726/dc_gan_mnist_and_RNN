# Boston Housing Price Prediction with DNN
# June 27, 2019
# Sung Kyu Lim
# Georgia Institute of Technology
# limsk@ece.gatech.edu
# 교수님 자료 인용

# 설치 패키지 : pandas, matplotlib, keras, scikit-learn

import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# hyper-parameter

MY_EPOCH = 100
MY_BATCH = 32 # 한번에 가지고 오는 데이터 수

                ####################
                # DATABASE SETTING #
                ####################

heading = ['crim', 'zn', 'indus', 'char', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'block', 'lstat', 'medv']

# 보스턴 집값 Dataset 세부 사항 14가지 요건
# crim : 범죄율
# zn : 25,000 평방피트를 (약 700평) 초과하는 거주지역 비율
# indu : 비소매 상업지역 면적 비율
# chas : 찰스강의 경계에 위치한 경우는 1, 아니면 0
# nox : 일산화질소 농도
# rm : 주택당 방 수
# age : 1940년 이전에 건축된 주택의 비율

# dis : 직업센터의 거리
# rad : 방사형 고속도로까지의 거리
# tax : 재산세율
# Ptratio : 학생/교사 비율
# Block : 인구 중 흑인 비율
# Lstat : 인구 중 하위 계층 비율
# medv : 평균 주택 가격(1,000달러)

df = pd.read_csv('housing.csv', delim_whitespace=True, names=heading)

#print(df.head(5))
#print(df.describe())

# z- 점수 정규화 (종류가 다른 점수의 표준 분포를 비교(상관관계를 가능하게)할수 있는 식)
scaler = StandardScaler()
# z는 dataframe이 아니라 numpy!!!!! 잊지말자
z = scaler.fit_transform((df)) # fit_transform()함수는 사용된 평균과 표준편차를 기억 ,
                                  # 나중에 z-점수를 원래 점수로 역전환 할때사용
#print(type(z))

# numpy를 pandas의 dataframe으로
z = pd.DataFrame(z, columns=heading)

#print(z.head(5))


#print(z.describe())

# z - 점수 정규화 -> pandas 프레임으로 잘 그림, 차트, SQL 처럼으로 구현해주는 것이 pandas?
#print('\n== BOX PLOT OF SCALED DATA ==')
#boxplot = z.boxplot(column=heading)
#plt.show()

# X/Y로 나눔
X = z.drop('medv', axis =1) # 14가지요소 다양하게 예측 age, rm, medv
print(X.shape)

Y= z['age'] # ""
print(Y.shape)

# 학급용/평가용

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.3, random_state= 5) # random_state = 5는 왜 사용하는 건지 모르겠다.

###############################
# DNN 구현
###############################

model = Sequential()
model.add(Dense(200, input_dim= 13 , activation='relu'))# bias 안넣음 # Dense 4개 13가지 특징= headings, # 시냅스 200개
model.add(Dense(1000, activation='relu')) # Dense 시냅스 1000개 활성화 함수 relu
#model.add(Dense(500, activation= 'relu'))
model.add(Dense(1, activation='linear')) # output 1개
model.summary()

#DNN 학습

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])# 최적화 Adam, sge, loss = mse
model.save('chap2.h5')
# mertrics 인자 뭐였지 ?

# 학습용 데이터 사용

from time import time
begin = time()
model.fit(X_train, Y_train, epochs= MY_EPOCH, batch_size=MY_BATCH, verbose=1) #verbose 보여주는것 얼마큼 trainning 되는 지를
total = time() - begin
print('time = ', total)
# trainning rate 시간 보여줌 learnning rate는 원래 model.summary에서 보여준다.

# 평가 ! 평가용 데이터 사용


loss, acc = model.evaluate(X_test, Y_test, verbose= 1)
print('\nMSE of DNN model', loss)
print('Model accuracy', acc)

# comparison with linear regression 선형회귀 시스템과 비교 -> 훨씬 Keras DNN이 잘나왔다는 것을 figure 통해 확인

from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(X_train, Y_train)
Y_model2 = model2.predict(X_test)

Y_model = model.predict(X_test)

# plot keras DNN modeling result
plt.figure(1)
plt.subplot(121)
plt.scatter(Y_test, Y_model)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Keras DNN Model")
#plt.scatter(Y_test, Y_model)

# plot linear regression modeling result
plt.subplot(122)
plt.scatter(Y_test, Y_model2)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Linear Regression Model")

#plt.show()

# calculate mean square error of linear regression model

mse = mean_squared_error(Y_test, Y_model2)
print('\nMSE of linear regression model', mse) # 0.22정도 차이가 난다. DNN보다



