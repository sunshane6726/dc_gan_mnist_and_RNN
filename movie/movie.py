print("hello world")

# import packages
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, SimpleRNN, LSTM
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix, f1_score

#global constants and hyper-parameters

MY_WORDS = 10000 # dictionary array ex) hello = 1 yellow = 2
MY_LENGTH = 80 # if a > 80 , cutting apablet
MY_DIM = 32  # 10000개 -> 32개로 WORD EMBEDDING
MY_SAMPLE = 9
MY_EPOCH = 15
MY_BATCH = 200

####################
# DATABASE SETTING #
####################


# load the IMDB dataset from keras
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=MY_WORDS)

# display DB info
# input is 1-dimensional array of lists
# output is 1-dimensional array of binary numbers

# 리뷰를 4등분하여서 테스트와 훈련을 구분한다. 훈련 2만개 테스트 5천개

print('\n== DB SHAPE INFO ==')
print('X_train shape = ', X_train.shape)
print('X_test shape = ', X_test.shape)
print('Y_train shape = ', Y_train.shape)
print('Y_test shape = ', Y_test.shape)

#print(X_train[0])
#print((Y_train))

def show_length():
    for i in range(10):
        print('리뷰', i, '길이', len(X_train[i]))

show_length()
#print(X_train[MY_SAMPLE])

# 사전 제작
word_to_id = imdb.get_word_index()
#print(word_to_id['started'])

id_to_word = {}
for key, val in word_to_id.items():
    id_to_word[val] = key

print("\n")
#print(id_to_word[2])



# padding 자리  # pre 앞이 0 'post'는 뒤에 0 변환
X_train = pad_sequences(X_train, truncating = 'post', padding = 'post', maxlen = MY_LENGTH)

X_test = pad_sequences(X_test, truncating = 'post', padding = 'post', maxlen = MY_LENGTH)

show_length() #80단어수 삭제하는 것


print(X_train[MY_SAMPLE])


# 1234번째 리뷰를 숫자에서 단어로 전환
review = []
for i in X_train[MY_SAMPLE]:
    word = id_to_word.get(i-3, "???")
    review.append(word)


#print(review)
# i-3 해야 한다. 3가지 특수문자가 존재하기 때문이다. 누락된문자, 첫리뷰의 첫자리, 패딩
# 단어 임베딩 원샷에 비해서 속도는 조금 느리지만 메모리 개선에 탁월하다.
# illustration of pad_sequences
# RNN을 시작 할 것이다.]

# 임베딩과 RNN구현

# word embedding is key in natural language process (NLP)
# it simplifies vector representation of words
# each word is reduced from MY_WORDS (10,000) down to MY_DIM (32)

model = Sequential() # 만개가 -> 32개 임베딩
model.add(Embedding(MY_WORDS, MY_DIM, input_length=MY_LENGTH))
model.add(LSTM(MY_DIM, input_shape=(MY_LENGTH, MY_DIM)))

model.add(Dense(1)) # 0.01 시냅스 - 32 bias = 1 dense = 33개
model.add(Activation('sigmoid'))
model.summary()

# RNN을 학습할때 Compile , Fit이 사용되고 있습니다.(수능 모의고사)

model.compile(optimizer='rmsprop', loss= 'binary_crossentropy', metrics=['acc'])

model.fit(X_train, Y_train, epochs=MY_EPOCH, batch_size=MY_BATCH, verbose=0)

score = model.evaluate(X_test, Y_test, verbose=1)
print(score)


# RNN 활용

predict =model.predict(X_test, verbose=1)
print(predict[0])
print(Y_test[0])

# boolen
fi_score = 0
print(predict)
predict = predict > 0.5
print(predict)
print(confusion_matrix(Y_test, predict))
print('최종값은 :', f1_score(Y_test, predict,average = 'micro' ))

# [ TP FP   ] 두번째가 기계가 내린 정답 첫번째가 사실(맞는것)
# [ FN TN   ]

# [TP  FP ] => PRECISION( TP/(TP+FP))
# [TP FN] => RECALL (TP/(TP + FN))
# [TP TN ] => ACCURACAY (TP + TN)/ TOTAL
# F1 SCORE = (2 * PRECISION * RECALL)/(PRECISION + CALL))



