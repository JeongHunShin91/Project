
# import pandas as pd
# from SyncRNG import SyncRNG
# from xgboost import XGBClassifier

import pandas as pd
import numpy as np
from SyncRNG import SyncRNG
from sklearn.neural_network import MLPClassifier
import os
os.getcwd()
os.chdir('C:/Google driver/pythonProject_study')

#데이터로드
raw_data = pd.read_csv('diabetes.csv')


# 데이터 셋  7:3 으로 분할
v=list(range(1,len(raw_data)+1))
s=SyncRNG(seed=42)
ord=s.shuffle(v)
idx=ord[:round(len(raw_data)*0.7)]

# R에서는 데이터프레임이 1부터 시작하기 때문에
# python에서 0행과 R에서 1행이 같은 원리로
# 같은 인덱스 번호를 가진다면 -1을 해주어 같은 데이터를 가지고 오게 한다.
# 인덱스 수정-R이랑 같은 데이터 가져오려고
for i in range(0,len(idx)):
    idx[i]=idx[i]-1

# 학습데이터, 테스트데이터 생성
train=raw_data.loc[idx] # 70%
test=raw_data.drop(idx) # 30%

#전처리

y_train=pd.DataFrame(train.Outcome)
x_train=pd.DataFrame(train.drop('Outcome',axis=1))
y_test=pd.DataFrame(test.Outcome)
x_test=pd.DataFrame(test.drop('Outcome',axis=1))

# neural network 모델
# clf = MLPClassifier(solver = "lbfgs", alpha = 1e-5, hidden_layer_sizes = (5,2), random_state = 1)
nn_model = MLPClassifier(random_state = 1)
nn_model.fit(x_train,y_train)

nn_pred = nn_model.predict(x_test)


# neural network tune 모델
nn_tune_model = MLPClassifier(solver="adam", max_iter=5000, activation = "relu",
                    hidden_layer_sizes = (12),
                    alpha = 0.05,
                    batch_size = 64,
                    learning_rate_init = 0.001,
                    random_state=1)
nn_tune_model.fit(x_train,y_train)

nn_tune_pred = nn_tune_model.predict(x_test)


# 평가
from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, nn_pred))
metrics.f1_score(y_test, nn_pred)
####################
# 0.5040650406504065

# 평가
print(classification_report(y_test, nn_tune_pred))
metrics.f1_score(y_test, nn_tune_pred)
# 0.5815602836879432
"""
R
#평가
caret::confusionMatrix(as.factor(nnet_pred),test$Outcome)$byClass
######## 
# 0.83
"""



