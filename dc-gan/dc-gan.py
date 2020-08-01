print("hello world")

## 1. IMPORT PACKAGES ###

import matplotlib.pyplot as plt
import numpy as np
import os

from time import time
from keras.datasets import fashion_mnist, cifar10, fashion_mnist
from keras.layers import Activation, BatchNormalization
from keras.layers import Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU, ReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.models import Sequential

# 하이퍼 파라미터
MY_EPOCH = 100
MY_BATCH = 10
MY_NOISE = 100  # 생성자가 사용하는 입력값
MY_SHAPE = (28, 28, 1)

# 출력 이미지를 저장하는 폴더 생성
MY_OUTPUT = 'output'
if not os.path.exists(MY_OUTPUT): # 전역변수로 파이썬이 설정되어있어서 os.path 경로 되어있다.
    os.makedirs(MY_OUTPUT)
### 3. LOAD AND MANIPULATE DATASET (MNIST 데이터 가져오기
def read_dataset():
    (X_train, _), (_, _) = fashion_mnist.load_data()

    print(X_train.shape)
    #print(X_train[0])
    X_train = X_train / 127.5 - 1.0 # tanh 활성화 함수랑 잘 사용이 돕니다. 그래서 -1 ~ 1로 구현 -> 표준편차와 정규화를 위해서 사용하는 z 정수 정규화 x
    #print(X_train[0])

    X_train = np.expand_dims(X_train, axis =3) # axis n차원을 들려라하는 방법이다. (60000, 28, 28, 1) 3차원 시작해서
    #print(X_train.shape)

        # X_train o
        # Y_train x 비지도 학습이라서 필요없다.
        # X_test  x 생성자에게 직접 볼꺼야 testset이 필요없다는 말이다.
        # Y_test  x

    return X_train

# 호출
X_train = read_dataset()

# 생성자 구현

def build_generator():
    model = Sequential()

    # 입력층 + 은익층 z-score ->  BN 사용 다른 두 영역을 비교하기 위해서
    model.add(Dense(7*7*256, input_dim=MY_NOISE))
    model.add(Reshape((7, 7, 256)))

    # 1st convolutional layer, from 28 x 28 x 1 into 14 x 14 x 32 tensor
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')) # conv2D, BN, LeakyReLU 3가지는 짝궁들이다.
    model.add(BatchNormalization()) # BN -> 속도및 기울기 소실 방지
    model.add(LeakyReLU(alpha=0.01)) # LeakuReLU 의 필요성을 확인해보는 것

    # 2nd convolutional layer, from 14 x 14 x 32 into 7 x 7 x 64 tensor
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))  # conv2D, BN, LeakyReLU 3가지는 짝궁들이다.
    model.add(BatchNormalization())  # BN -> 속도및 기울기 소실 방지
    model.add(LeakyReLU(alpha=0.01))  # LeakuReLU 의 필요성을 확인해보는 것

    # 3rd convolutional layer, transpose 블럭
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same',activation='tanh'))  # conv2D, BN, LeakyReLU 3가지는 짝궁들이다.
    # output layer with "tanh" activation
    # print model summary
    print("===== generator ====")
    #model.summary()
    return model
    #model.add(BatchNormalization())  # BN -> 속도및 기울기 소실 방지
    #model.add(LeakyReLU(alpha=0.01))  # LeakuReLU 의 필요성을 확인해보는 것

generator = build_generator()

def build_discriminator():
    model = Sequential()

    # 첫번째 convulution 시작
    model.add(Conv2D(32,kernel_size=3, strides=2, input_shape=MY_SHAPE, padding='same')) # MY_SHAPE = (28, 28, 1)
    model.add(LeakyReLU(alpha=0.01))

    # 두번째 convolution 시작
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())  # BN -> 속도및 기울기 소실 방지
    model.add(LeakyReLU(alpha=0.01))  # LeakuReLU 의 필요성을 확인해보는 것

    # 3rd convolutional layer, from 7 x 7 x 64 tensor into 4 x 4 x 128 tensor
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))


    # output layer with sigmoid activation
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # print model summary
    print('\n=== Discriminator summary')
    #model.summary()
    return model

discriminator = build_discriminator()
build_discriminator()

# GAN 실제로 만들기 구현
def build_GAN():

    # ------- 감별자 구축 ----------------------------------------------------
    # build discriminator first
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer='Adam',
    metrics=['accuracy'])
    discriminator.trainable = False  # 생성자 학습시 감별자 가중치는 고정 되어 있음음
    # ----------------------------------------------------------------------



    # 생성자 구축
    generator = build_generator()

    # GAN 구축
    gan = Sequential()
    gan.add(generator) # 반대로 해서는 안된다.
    gan.add(discriminator)
    gan.compile(optimizer='Adam', loss='binary_crossentropy') # GAN 훈련이 GENERATOR 랑 동일하게 만들었다. 엄청 가독성이 떨어진다.
    print('\n=== GAN summary')

    #gan.summary()

    return generator, discriminator, gan

#generator, discriminator, gan = build_GAN()

#build_GAN()

def train_discriminator():

    # label for real images: all ones
    # double parenthesis are needed to create 2-dim array
    all_1 = np.ones((MY_BATCH, 1))

    # label for fake images: all zeros
    all_0 = np.zeros((MY_BATCH, 1))

    # print(all_0) # 배치갯수
    # 진짜 ramdom 하게 배치 수 만큼 가져옴

    # get a random batch of real images
    pick = np.random.randint(0, X_train.shape[0], MY_BATCH) # 6만 5천개증에 10개뽑음

    #print(pick)
    real = X_train[pick]

    # 드디어 경찰 학습 : 진짜 10개로 !!! # 손실함수는 crossentropy
    d_loss_real = discriminator.train_on_batch(real, all_1) # all_1 10개아닌 한개인 이유는 평균이라서 # keras train_on_batch가 들어 있는 이유 배치만큼 수행을 해서 배치가 많다고 꼭 좋은 것은 아니다.
    #print(d_loss_real)

    # 드디어 경찰 학습 : 가짜 10개로 !!! # 손실함수는 crossentropy

    noise = np.random.normal(0, 1, (MY_BATCH, MY_NOISE))
    #print(noise)
    fake = generator.predict(noise)

    #plt.imshow(fake[0].squeeze(), cmap='gray')
    #plt.show()
    d_loss_fake = discriminator.train_on_batch(fake, all_0)
    #print(d_loss_fake)
    d_loss, acc = 0.5 * np.add(d_loss_real, d_loss_fake)

    #print(d_loss)
    return d_loss, acc

#train_discriminator()

def train_generator():
    noise = np.random.normal(0, 1, (MY_BATCH, MY_NOISE))
    # print(noise)

    #fake = generator.predict(noise) - GAN안에서 FAKE가 존재하기 때문에 상관이없다.
    all_1 = np.ones((MY_BATCH, 1))

    loss = gan.train_on_batch(noise, all_1) # 진짜같은 이미지 생성
    #print(loss)

    return loss
#train_generator()
def sample_images(itr):
    row = col = 4

    noise = np.random.normal(0, 1, (16, MY_NOISE))
    fake = generator.predict(noise)

    fake = 0.5 * fake + 0.5 # -1 ~ 1 -> 0 ~ 1이어야 그림이 나온다.
    _, axs = plt.subplots(row, col, figsize=(row, col))
    cnt = 0
    for i in range(row):
        for j in range(col):
            axs[i, j].imshow(fake[cnt, :, :, 0], cmap = 'gray')
            axs[i, j].axis('off')
            cnt += 1

    path = os.path.join(MY_OUTPUT, 'img-{}'.format(itr+1))
    plt.savefig(path)
    plt.close()

def train_GAN():
    begin = time()
    for itr in range(MY_EPOCH):
        d_loss, d_acc = train_discriminator()
        g_loss = train_generator()
        print('epoch: {}, gen loss: {:.3f},' 'dis loss:{:.3f},' 'dis acc:{:.3f}'.format(itr+1, g_loss, d_loss, d_acc)) # i = 19 -> i+1 = 20
        sample_images(itr)

    # print training time
    total = time() - begin
    print('총 학습시간:', total)

# ================메인 함수===============================
X_train = read_dataset()
generator, discriminator, gan = build_GAN()
train_GAN()

#gan.save_weights('save.h5')
#generator.load_weights('save.h5')






