# 선형회귀
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SyncRNG import SyncRNG
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
raw_data = pd.read_csv('product.csv',encoding='cp949')
v=list(range(1,len(raw_data)+1))
s=SyncRNG(seed=42)
ord=s.shuffle(v)
idx=ord[:round(len(raw_data)*0.7)]
for i in range(0,len(idx)):
    idx[i]=idx[i]-1
train=raw_data.loc[idx]
test=raw_data.drop(idx)

x_train = train[["제품_적절성"]]
y_train = train["제품_만족도"]
x_test = test[["제품_적절성"]]
y_test = test["제품_만족도"]

import math
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# 기본
line_fitter = LinearRegression()
# 튜닝
line_fitter = LinearRegression(fit_intercept=26)
line_fitter.fit(x_train, y_train)
pred = line_fitter.predict(x_train)
mse=mean_squared_error(pred,y_train)
print('rmse',math.sqrt(mse))

# ==========================================
# 다중 선형회귀
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import random

import matplotlib.pyplot as plt
data = pd.read_csv('BostonHousing.csv',encoding='cp949')

x = data.drop(['medv'],axis=1)
y = data[['medv']]

var = x.columns.tolist()

selected = var
sl_remove = 0.05

sv_per_step = []
adjusted_r_squared = []
steps = []
step = 0

while len(selected) > 0:
    X = sm.add_constant(data[selected])
    p_vals = sm.OLS(y, X).fit().pvalues[1:]
    max_pval = p_vals.max()

    if max_pval >= sl_remove:
        remove_variable = p_vals.idxmax()
        selected.remove(remove_variable)
        step += 1
        steps.append(step)
        adj_r_squared = sm.OLS(y, sm.add_constant(data[selected])).fit().rsquared_adj
        adjusted_r_squared.append(adj_r_squared)
        sv_per_step.append(selected.copy())
    else:
        break

selected
from sklearn.metrics import classification_report
# x = data.drop(["indus","age","medv"], axis=1)
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.7, test_size=0.3,random_state=295)
v=list(range(1,len(raw_data)+1))
s=SyncRNG(seed=42)
ord=s.shuffle(v)
idx=ord[:round(len(raw_data)*0.7)]
for i in range(0,len(idx)):
    idx[i]=idx[i]-1
train=raw_data.loc[idx]
test=raw_data.drop(idx)

xtrain=data[['crim','zn','chas','nox','rm','dis','rad','tax','ptratio','b','lstat']]
ytrain=data['medv']
xtest=data.drop(["indus","age",'medv'],axis=1)
ytest=data['medv']

mlr = LinearRegression()
mlr.fit(xtrain, ytrain)
pred = mlr.predict(xtest)
mlr.fit(pred.reshape(-1, 1), ytest)

import math
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# 기본
line_fitter = LinearRegression()
# 튜닝
line_fitter = LinearRegression(fit_intercept=26)
line_fitter.fit(xtrain, ytrain)
pred = line_fitter.predict(xtrain)
mse=mean_squared_error(pred,ytrain)
print('rmse',math.sqrt(mse))