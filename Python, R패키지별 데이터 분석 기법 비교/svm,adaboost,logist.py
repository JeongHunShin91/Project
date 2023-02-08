# 시험환경 세팅 (코드 변경 X)
import pandas as pd
from SyncRNG import SyncRNG
import os
os.getcwd()
os.chdir('E:/GoogleDrive/2022년 빅데이터&AI 강의/workplace/Python/work')

#데이터로드
raw_data = pd.read_csv('diabetes.csv')

# 데이터 셋  7:3 으로 분할
v=list(range(1,len(raw_data)+1))
s=SyncRNG(seed=42)
ord=s.shuffle(v)
idx=ord[:round(len(raw_data)*0.7)]

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

#모델 train
from sklearn.linear_model import  LinearRegression
help
line_fitter = LinearRegression()
line_fitter.fit(x_train, y_train)
pred = line_fitter.predict(x_test)
########################################support vector machine
#svm모델
from sklearn import svm
svm_model = svm.SVC()
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)

svm_tune_model = svm.SVC(kernel='rbf',cache_size=40,gamma=1/1000)
svm_tune_model.fit(x_train, y_train)
svm_tune_pred = svm_tune_model.predict(x_test)

#adaboost모델
from sklearn.ensemble import AdaBoostClassifier
adaboost_model = AdaBoostClassifier()
adaboost_model.fit(x_train, y_train)
adaboost_pred = adaboost_model.predict(x_test)

#adaboost 튜닝 모델
from sklearn.tree import DecisionTreeClassifier
base_model = DecisionTreeClassifier(max_depth = 5)
adaboost_tune_model = AdaBoostClassifier(base_estimator = base_model,
                                         n_estimators = 100,
                                         random_state = 10,
                                         learning_rate = 0.01)
adaboost_tune_model.fit(x_train, y_train)
adaboost_tune_pred = adaboost_tune_model.predict(x_test)

#로지스틱 모델
from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
logit_model.fit(x_train, y_train)
logit_pred = logit_model.predict(x_test)

from sklearn.linear_model import LogisticRegression
logit_tune_model = LogisticRegression(solver="lbfgs",max_iter=200)
logit_tune_model.fit(x_train, y_train)
logit_tune_pred = logit_tune_model.predict(x_test)
##################################################
#결과
output = pd.DataFrame({'idx': x_test.index, 'Outcome': svm_pred})
output.head()

#평가
from sklearn.metrics import classification_report
from sklearn import metrics
#svm평가
print(classification_report(y_test, svm_pred))
metrics.f1_score(y_test, svm_pred)
#svm튜닝 평가
print(classification_report(y_test, svm_tune_pred))
metrics.f1_score(y_test, svm_tune_pred)
#adaboost평가
print(classification_report(y_test, adaboost_pred))
metrics.f1_score(y_test, adaboost_pred)
#adaboost 튜닝 평가
print(classification_report(y_test, adaboost_tune_pred))
metrics.f1_score(y_test, adaboost_tune_pred)
#로지스틱 평가
print(classification_report(y_test, logit_pred))
metrics.f1_score(y_test, logit_pred)
#로지스틱 튜닝평가
print(classification_report(y_test, logit_tune_pred))
print(metrics.f1_score(y_test, logit_tune_pred))