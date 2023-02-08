import pandas as pd
import numpy as np
from SyncRNG import SyncRNG
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
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

# xgboost
xgb_model  = XGBClassifier()
xgb_model .fit(x_train, y_train)

confusion_matrix(y_test, xgb_model.predict(x_test))
xgb_pred  = xgb_model .predict(x_test)

xgb_model .score(x_test, y_test)

# xgboost_tune 모델
xgb_tune_model = xgb.XGBClassifier(booster = 'gbtree',
                                   learning_rate = 0.02,
                                   n_estimators=1000,
                                   min_child_weight=3 ,
                                   max_depth=8)
xgb_tune_model.fit(x_train, y_train,
                   early_stopping_rounds=30,
                   eval_set=[(x_test, y_test)],
                   verbose=False )
confusion_matrix(y_test, xgb_tune_model.predict(x_test))
xgb_tune_pred = xgb_tune_model.predict(x_test)

from sklearn.metrics import classification_report
from sklearn import metrics
print(classification_report(y_test, xgb_pred ))
metrics.f1_score(y_test, xgb_pred )
print(classification_report(y_test, xgb_tune_pred))
metrics.f1_score(y_test, xgb_tune_pred)
#####################
# 0.6754966887417219

"""
R
# f1 score
caret::confusionMatrix(as.factor(test_y), as.factor(XGB_pred))$byClass[7]
#########
# 0.81
"""



