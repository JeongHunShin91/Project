# 1. 제품적절성이 제품만족도에 미치는 영향 주제로 R을 이용한 단순 선형
# 회귀분석을실시하고, 딥러닝 교재 3장에서 사용된 인공신경망을 이용한 선형
# 회귀분석을python으로 실행하여 결과를 비교하시오.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib import font_manager, rc

#데이터 불러오기
product = pd.read_csv("product.csv", encoding='cp949')
product.head()

#데이터 전처리
x = product["제품_적절성"]
y = product["제품_만족도"]

#Neuron 생성
class Neuron:

    def __init__(self):
        self.w = 1.0     # 가중치를 초기화합니다
        self.b = 1.0     # 절편을 초기화합니다

    def forpass(self, x):
        y_hat = x * self.w + self.b       # 직선 방정식을 계산합니다
        return y_hat

    def backprop2(self, x, err2):
        w_grad = x * err2    # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err2    # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad


    def fit(self, x, y, lr = 0.3, epochs=100):
        for i in range(epochs):           # 에포크만큼 반복합니다
            for x_i, y_i in zip(x, y):    # 모든 샘플에 대해 반복합니다
                n = len(x)
                y_hat = self.forpass(x_i) # 정방향 계산
                # err = -(y_i - y_hat)      # 오차 계산
                err2 = -(2/n) *(y_i - y_hat)
                w_grad, b_grad = self.backprop2(x_i, err2)
                self.w -= w_grad *lr          # 가중치 업데이트
                self.b -= b_grad *lr          # 절편 업데이트

neuron = Neuron( )

# 가중치와 절편 구하기
neuron.fit(x,y)
neuron.w # 0.7462346285634439
neuron.b # 0.7594095886771006

# 회귀식에 대한 상관계수 구하기
est_y = np.array(x) * neuron.w + neuron.b # x의 실제 값들을 회귀식에 대입한 y 추정치
r2 = r2_score(y, est_y) #0.5880131758128271

# 선형 회귀분석 모델 시각화
# 산점도 그래프 그리기
plt.scatter(x, y, color = 'r', s = 20)

# 회귀선 그래프 그리기
pt1 =(1,1*neuron.w + neuron.b)
pt2 =(5,5*neuron.w + neuron.b)
plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]], color = 'orange')

# 텍스트 삽입
plt.text(1, 4.8, '$R^2$ = %.4f'%r2, size = 12) # (1, 4.8)의 위치에 크기 12로 R값 새김
plt.text(1, 4.5, 'y = %.4fx + %.4f'%(neuron.w, neuron.b), size = 12)
# (1, 4.5)위치에 추세선 식 표현

# 한글 깨짐 문제 해결
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 라벨 입력
plt.xlabel('제품_적절성')
plt.ylabel('제품_만족도')
plt.show()


# 2. 비 유무 예측 주제로 R을 이용한 로지스틱 회귀분석을 실시하고,
# 딥러닝 교재4장에서 사용된 인공신경망을 이용한 로지스틱 회귀분석을
# python으로 실행하여 결과를 비교하시오.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SyncRNG import SyncRNG
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve

date = pd.read_csv("weather.csv", encoding='cp949')
date['RainTomorrow'] = date['RainTomorrow'].apply(lambda x : 1 if x== "Yes" else 0)

# 데이터 셋  7:3 으로 분할
v=list(range(1,len(date)+1))
s=SyncRNG(seed=42)
ord=s.shuffle(v)
idx=ord[:round(len(date)*0.7)]

for i in range(0,len(idx)):
    idx[i]=idx[i]-1

train=date.loc[idx] # 70%
train = train.dropna()
test=date.drop(idx) # 30%
test =test.dropna()
x_train=train.drop(["Date","WindGustDir","WindDir","RainToday","RainTomorrow"],axis = 1)
y_train=train["RainTomorrow"]

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test=test.drop(["Date","WindGustDir","WindDir","RainToday","RainTomorrow"],axis = 1)
y_test=test["RainTomorrow"]
x_test = np.array(x_test)
y_test = np.array(y_test)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_test = scaler.fit_transform(x_test)
x_train = scaler.fit_transform(x_train)

class SingleLayer:

    def __init__(self, learning_rate=0.01, l1=0, l2=0):
        self.w = None
        self.b = None
        self.losses = []
        self.w_history = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산합니다
        return z

    def backprop(self, x, err):
        w_grad = x * err  # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err  # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)  # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a

    def fit(self, x, y, epochs=300, x_val=None, y_val=None):
        self.w = np.ones(x.shape[1])  # 가중치를 초기화합니다.
        self.b = 0  # 절편을 초기화합니다.
        self.w_history.append(self.w.copy())  # 가중치를 기록합니다.
        np.random.seed(42)  # 랜덤 시드를 지정합니다.
        for i in range(epochs):  # epochs만큼 반복합니다.
            loss = 0
            # 인덱스를 섞습니다
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:  # 모든 샘플에 대해 반복합니다
                z = self.forpass(x[i])  # 정방향 계산
                a = self.activation(z)  # 활성화 함수 적용
                err = -(y[i] - a)  # 오차 계산
                w_grad, b_grad = self.backprop(x[i], err)  # 역방향 계산
                # 그래디언트에서 페널티 항의 미분 값을 더합니다
                w_grad += self.l1 * np.sign(self.w) + self.l2 * self.w
                self.w -= self.lr * w_grad  # 가중치 업데이트
                self.b -= self.lr * b_grad  # 절편 업데이트
                # 가중치를 기록합니다.
                self.w_history.append(self.w.copy())
                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적합니다
                a = np.clip(a, 1e-10, 1 - 1e-10)
                loss += -(y[i] * np.log(a) + (1 - y[i]) * np.log(1 - a))
            # 에포크마다 평균 손실을 저장합니다
            self.losses.append(loss / len(y) + self.reg_loss()/len(y))

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]  # 정방향 계산
        return np.array(z) >= 0  # 스텝 함수 적용

    def score(self, x, y):
        return np.mean(self.predict(x) == y)

    def proba(self, x):
        z = [self.forpass(x_i) for x_i in x]  # 정방향 계산
        return np.array(z)  # 스텝 함수 적용

    def reg_loss(self):
        return self.l1 * np.sum(np.abs(self.w)) + self.l2 / 2 * np.sum(self.w ** 2)

neuron = SingleLayer()
neuron.fit(x_train, y_train)
neuron.score(x_test,y_test)

pred_positive_label = neuron.proba(x_test)
fprs, tprs, thresholds = roc_curve(y_test, pred_positive_label,pos_label=1)
precisions, recalls, thresholds = roc_curve(y_test, pred_positive_label,pos_label=1)
plt.figure(figsize=(15,5))
plt.plot(fprs,tprs,label='ROC')
plt.ylable("True positive rate")
plt.xlable("False positive rate")



# 3. 다중분류를 위한 머신러닝 기법을 선택하고 R을 이용하여 다중분류를 실시하고,
# 딥러닝 교재 7장에서 사용된 텐서플로와 케라스을 이용한 인공신경망을 python으로
# 실행하여 결과를 비교하시오.
# 데이터: iris
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터 로드
iris2 = load_iris()

X = iris2['data']
y = iris2['target']

# 원-핫 인코딩 변환
Y = tf.keras.utils.to_categorical(y)

# 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=42)
print(X_train.shape, y_train.shape)

# 모델 생성
model = Sequential()
# 은닉층과 출력층에 모델 추가
model.add(Dense(100,activation='sigmoid',input_shape=(4,)))
model.add(Dense(100,activation='sigmoid'))# 층 추가로 성능 향상
# 합성곱층을 여러 개 추가해도 학습할 모델 파라미터의 개수가 크게 늘지 않기 때무에 계산 효율성 좋음
model.add(Dense(3,activation='softmax'))
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer(type)     Output Shape       Param  #
# =================================================================
# dense(Dense)    (None, 100)         500
# 4*100+100: input_shape=(4,)라서
# dense_1(Dense)  (None, 100)         10100
# 100*100+100
# dense_2(Dense)  (None, 3)           303
# 100*3+3
# =================================================================
# Total params: 10,903
# Trainable params: 10,903
# Non-trainable params: 0
# _________________________________________________________________

# 최적화 알고리즘과 손실 함수 지정
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', # 최적화 알고리즘으로 적응적 학습률 알고리즘 아담 사용
              metrics=['accuracy'])
# 아담은 손실 함수의 값이 최적값에 가까워질수록 학습률을 낮춰 손실 함수의 값이 안정적으로 수렴되게 함

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200)

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'test_accuracy'])
plt.show()