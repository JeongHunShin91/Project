#의사결정 나무,랜덤포레스트
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from SyncRNG import SyncRNG
import os
os.getcwd()
os.chdir('diabetes.csv')

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
#train=train.sort_index(ascending=True)
test=raw_data.drop(idx) # 30%

#전처리

y_train=pd.DataFrame(train.Outcome)
x_train=pd.DataFrame(train.drop('Outcome',axis=1))
y_test=pd.DataFrame(test.Outcome)
x_test=pd.DataFrame(test.drop('Outcome',axis=1))

#랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier
random_model = RandomForestClassifier(n_estimators=100,max_features=4,max_depth=5)
random_model.fit(x_train,y_train)
random_pred = random_model.predict(x_test)
#결과
output = pd.DataFrame({'idx': x_test.index, 'Outcome': random_pred})
output.head()
#평가
from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, random_pred))
metrics.f1_score(y_test, random_pred)
#0.6474820143884891

#의사결정 나무
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth=3,min_samples_leaf=3,min_samples_split=2)
tree_model.fit(x_train, y_train)
tree_pred = tree_model.predict(x_test)

#결과
output = pd.DataFrame({'idx': x_test.index, 'Outcome': tree_pred})
output.head()
#평가
from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, tree_pred))
metrics.f1_score(y_test, tree_pred)
