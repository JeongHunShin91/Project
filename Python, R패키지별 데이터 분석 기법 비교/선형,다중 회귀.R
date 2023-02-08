# 선형회귀
library(SyncRNG) 
library(caret)
library(ModelMetrics)
df<-read.csv("product.csv")

#데이터분할
v <- 1:nrow(df)
s <- SyncRNG(seed=42)
idx <- s$shuffle(v)[1:round(nrow(df)*0.7)] 
idx[1:length(idx)]
train_df <- df[idx,] 
test_df <- df[-idx,]
help(lm)

#R
m_lm<- lm(제품_만족도~제품_적절성, train)
p_lm<-predict(m_lm,test[,-1])
RMSE(p_lm,test[,3])
R2(p_lm,test[,3])
MAE(p_lm,test[,3])


# caret
caret_lm_model <- train(제품_만족도~제품_적절성, data = train_df,method = "lm")
caret_lm_model
p_lm<-predict(caret_lm_model,test[,-1])
RMSE(p_lm,test[,3])
R2(p_lm,test[,3])
MAE(p_lm,test[,3])


# caret 튜닝
nn_Grid <-  expand.grid(intercept=26)
gbmFit1 <- train(제품_만족도~제품_적절성, 
                 data = train,method = "lm",
                 tuneGrid=nn_Grid)
p_lm<-predict(caret_lm_model,test[,-1])
RMSE(p_lm,test[,3])
R2(p_lm,test[,3])
MAE(p_lm,test[,3])
help(train)
help(MAE)
################################


# =====================================
# 다중선형회귀
library(mlbench)
library(car)
library(ggplot2)

df<-read.csv("BostonHousing.csv")
v <- 1:nrow(df)
s <- SyncRNG(seed=42)
idx <- s$shuffle(v)[1:round(nrow(df)*0.7)]
idx[1:length(idx)]
train <- df[idx,]
test <- df[-idx,]
train
test

model <- lm(formula = medv ~ ., data = train)
test1 = test[,length(test)]
p_lm<-predict(model,test)
RMSE(p_lm,test1)

# caret
gbmFit1 <- train(medv~ ., data = train,method = "lm")
gbmFit1
test1 = test[,length(test)]
p_lm<-predict(model,test)
RMSE(p_lm,test1)

# caret tuning
nn_Grid <-  expand.grid(intercept=26)
gbmFit1 <- train(medv~ ., data = train,method = "lm",
                 tuneGrid=nn_Grid))
test1 = test[,length(test)]
p_lm<-predict(model,test)
RMSE(p_lm,test1)

#4$$$$$$$$$$$$$$$$
# 10-1 앙상블(부스팅)
# 데이터 로딩
library(mlbench)
library(adabag)
data(BreastCancer)
data <- BreastCancer
data <- na.omit(data)
data <- data[,-1]

# seed 고정, test/train 분류
library(SyncRNG)
library(fastAdaboost)
v <- 1:nrow(data)
s <- SyncRNG(seed=42)
s=s$shuffle(v)
idx <- s[1:round(nrow(data)*0.7)]

head(data[-idx[1:length(idx)],])
tr <- data[idx[1:length(idx)],]
te <- data[-idx[1:length(idx)],]
###$$$$$$$$$$$$$
# 튜닝
gbmGrid <-  expand.grid(nIter=200, method='adaboost')
set.seed(42)
model_en <- train(Class~.,data=tr,method='adaboost', tuneGrid=gbmGrid)
cpred <- predict(object=model_en, te)
sum(cpred==te$Class)/length(cpred)
