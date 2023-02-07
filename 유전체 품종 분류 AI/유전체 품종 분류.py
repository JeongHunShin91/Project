import pandas as pd
import random
import os
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from numpy.random import randint
import matplotlib.pyplot as plt
from numpy import argmax

class CFG:
    SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.SEED) # Seed 고정.

train = pd.read_csv('./open/train.csv')
test = pd.read_csv('./open/test.csv')

def get_x_y(df):
    if 'class' in df.columns:
        df_x = df.drop(columns=['id', 'class'])
        df_y = df['class']
        return df_x, df_y
    else:
        df_x = df.drop(columns=['id'])
        return df_x

train_x, train_y = get_x_y(train)
test_x = get_x_y(test)

class_le = preprocessing.LabelEncoder()
snp_le = preprocessing.LabelEncoder()
snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1,16)]

snp_data = []
for col in snp_col:
    snp_data += list(train_x[col].values)

train_y = class_le.fit_transform(train_y)
snp_le.fit(snp_data)

for col in train_x.columns:
    if col in snp_col:
        train_x[col] = snp_le.transform(train_x[col])
        test_x[col] = snp_le.transform(test_x[col])

train_x = train_x.drop(columns=['father','mother','gender'])
test_x = test_x.drop(columns=['father','mother','gender'])

# 원핫코딩
train_y_encoded = tf.keras.utils.to_categorical(train_y)

# 정규화
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
scaler.fit(test_x)
test_x = scaler.transform(test_x)

# 데이터셋 나누기
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y_encoded, test_size = 0.2,
                         stratify = train_y, random_state=CFG.SEED)
early_stopping = EarlyStopping(min_delta =0.001, patience =100, restore_best_weights=True)

h = tf.keras.losses.CategoricalCrossentropy()
model = keras.Sequential()
x = tf.keras.layers.Input(shape=[16])
y = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.models.Model(x,y)
model.compile(loss=h,metrics='accuracy',optimizer='adam')

history = model.fit(train_x,train_y,epochs=3000,validation_data=(val_x,val_y),callbacks=[early_stopping])
predict = model.predict(test_x)

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred_labels = []
for num in range(predict.shape[0]) :
    tmp = predict[num]/max(predict[num])
    tmp = tmp.astype('int')
    pred_labels.append(tmp)

pred_labels2= argmax(pred_labels, axis=1)

pred_labels1 = np.rint(pred_labels2)
pred_labels1 = class_le.inverse_transform(pred_labels1.astype(int))

submit = pd.read_csv('./open/sample_submission.csv')
submit['class'] = pred_labels1
submit.to_csv('./open/submit.csv', index=False)